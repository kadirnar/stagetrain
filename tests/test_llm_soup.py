"""`llm.soup`: the average of several LoRA runs' updates, written as one adapter. Checked for
real on a tiny random model built in the test; nothing is downloaded."""

import json

import pytest

torch = pytest.importorskip("torch")
peft = pytest.importorskip("peft")

from decision_tiny import build_model_dir  # noqa: E402

from stagetrain.core import Pipeline, Stage, StageParams, StageResult, build_config  # noqa: E402
from stagetrain.core.soup import average_adapters, lora_scale  # noqa: E402
from stagetrain.core.stage import _REGISTRY  # noqa: E402


def make_adapter(base, out, seed, **lora):
    """A LoRA adapter on the tiny model with random (untrained, non-zero) A and B, and a head trained in full."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_pretrained(str(base))
    spec = {"r": 4, "lora_alpha": 8, "target_modules": ["q_proj", "v_proj", "up_proj"], "init_lora_weights": False,
            "modules_to_save": ["lm_head"], **lora}
    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", **spec))
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "modules_to_save" in name:
                param.add_(torch.randn_like(param) * 0.05)
    model.save_pretrained(str(out))
    return out


def merged_weights(base, adapter):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(str(base)), str(adapter)).merge_and_unload()
    return {k: v.clone() for k, v in model.state_dict().items()}


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    return build_model_dir(tmp_path_factory.mktemp("soup") / "base")


def test_the_soup_is_the_exact_average_of_the_members_updates(base, tmp_path):
    members = [
        make_adapter(base, tmp_path / "a", seed=1),
        make_adapter(base, tmp_path / "b", seed=2, r=8, lora_alpha=4),  # another rank and scale
        make_adapter(base, tmp_path / "c", seed=3, use_rslora=True),  # alpha / sqrt(r)
    ]
    out = average_adapters(members, tmp_path / "soup")
    config = json.loads((out / "adapter_config.json").read_text())
    assert config["r"] == 16 and config["lora_alpha"] == 16 and config["use_rslora"] is False
    assert lora_scale(config) == 1.0

    singles = [merged_weights(base, m) for m in members]
    soup = merged_weights(base, out)
    changed = 0
    for name, value in soup.items():
        mean = sum(s[name] for s in singles) / len(singles)
        assert torch.allclose(value, mean, atol=1e-5), name
        changed += not torch.allclose(value, singles[0][name], atol=1e-5)
    assert changed == 7  # 2 layers x 3 projections, and the head: the members really differ

    # Averaging the A and B matrices on their own, which is the naive soup, is a different model.
    from safetensors.torch import load_file

    files = [load_file(m / "adapter_model.safetensors") for m in (members[0], members[2])]  # same shapes
    name = next(k for k in files[0] if "lora_A" in k and "q_proj" in k)
    a, b = [f[name] for f in files], [f[name.replace("lora_A", "lora_B")] for f in files]
    naive = ((b[0] + b[1]) / 2) @ ((a[0] + a[1]) / 2)
    exact = (b[0] @ a[0] + b[1] @ a[1]) / 2
    assert not torch.allclose(naive, exact, atol=1e-3)


def test_weights_and_refusals(base, tmp_path):
    a, b = make_adapter(base, tmp_path / "a", seed=1), make_adapter(base, tmp_path / "b", seed=2)
    out = average_adapters([a, b], tmp_path / "weighted", weights=[3, 1])
    wa, wb, soup = merged_weights(base, a), merged_weights(base, b), merged_weights(base, out)
    name = next(k for k in soup if "q_proj" in k)
    assert torch.allclose(soup[name], 0.75 * wa[name] + 0.25 * wb[name], atol=1e-5)

    with pytest.raises(ValueError, match="weights"):
        average_adapters([a, b], tmp_path / "x", weights=[1])
    with pytest.raises(ValueError, match="nothing"):
        average_adapters([], tmp_path / "x")
    with pytest.raises(FileNotFoundError, match="not a LoRA adapter"):
        average_adapters([a, base], tmp_path / "x")
    other = make_adapter(base, tmp_path / "other", seed=3, target_modules=["q_proj"])
    with pytest.raises(ValueError, match="different modules"):
        average_adapters([a, other], tmp_path / "x")
    dora = make_adapter(base, tmp_path / "dora", seed=4, use_dora=True)
    with pytest.raises(ValueError, match="use_dora"):
        average_adapters([a, dora], tmp_path / "x")
    config = json.loads((b / "adapter_config.json").read_text())
    (b / "adapter_config.json").write_text(json.dumps({**config, "base_model_name_or_path": "another/model"}))
    with pytest.raises(ValueError, match="different base models"):
        average_adapters([a, b], tmp_path / "x")


class TrainParams(StageParams):
    base: str
    train: dict = {}


class FakeTrain(Stage):
    """Stands in for `llm.sft`: writes a real adapter and a score that depends on the learning rate."""

    Params = TrainParams

    def run(self, ctx):
        lr = self.params.train.get("learning_rate", 1e-4)
        out = make_adapter(self.params.base, ctx.path("model"), seed=self.seed(ctx))
        return StageResult(artifacts={"model": out}, metrics={"eval/score": 0.9 - abs(lr - 2e-4) * 500 + 0.001 * self.seed(ctx)})


def sweep(tmp_path, base, **soup):
    _REGISTRY["llm.fake_train"] = FakeTrain
    return Pipeline(build_config({
        "track": "llm", "name": "soup", "output_dir": str(tmp_path),
        "stages": [
            {"name": "sft", "kind": "llm.fake_train", "params": {"base": str(base)},
             "matrix": {"train.learning_rate": [1e-4, 2e-4], "seed": [1, 2, 3]}},
            {"name": "best", "kind": "llm.select", "params": {"group": "sft", "metric": "eval/score", "average_over": ["seed"]}},
            {"name": "soup", "kind": "llm.soup", "params": {"group": "sft", **soup}},
        ],
    }))


def test_soup_of_the_selected_configurations_seeds(base, tmp_path):
    p = sweep(tmp_path, base, selected_by="best")
    assert set(p.config.stage("soup").needs) == {f"sft-{i}" for i in range(1, 7)} | {"best"}
    p.run()
    m = p.manifest("soup")
    assert [name.rsplit("/", 2)[-2] for name in m.notes["members"]] == ["sft-4", "sft-5", "sft-6"]  # lr 2e-4, three seeds
    assert m.metrics == {"soup/members": 3.0, "soup/rank": 12.0}
    soup = merged_weights(base, m.artifact("model", p.run_dir))
    singles = [merged_weights(base, p.manifest(f"sft-{i}").artifact("model", p.run_dir)) for i in (4, 5, 6)]
    name = next(k for k in soup if "v_proj" in k)
    assert torch.allclose(soup[name], sum(s[name] for s in singles) / 3, atol=1e-5)


def test_soup_parameters(base, tmp_path):
    p = sweep(tmp_path / "all", base)  # without selected_by: every run of the matrix
    p.run()
    assert p.manifest("soup").metrics["soup/members"] == 6.0
    for bad, message in (
        ({"group": None}, "models, or the group"),
        ({"group": None, "selected_by": "best"}, "models, or the group"),
        ({"weights": [1.0, 2.0]}, "weights go with models"),
    ):
        with pytest.raises(Exception, match=message):
            sweep(tmp_path / "bad", base, **bad).validate()
    lone = Pipeline(build_config({
        "track": "llm", "name": "lone", "output_dir": str(tmp_path),
        "stages": [{"name": "soup", "kind": "llm.soup", "params": {"models": [str(make_adapter(base, tmp_path / "a", 1))]}}],
    }))
    with pytest.raises(ValueError, match="at least two"):
        lone.run()
