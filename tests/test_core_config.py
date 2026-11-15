import pytest

from stagetrain.core.config import ConfigError, build_config, parse_overrides, resolve_refs, stage_refs


def raw(**over):
    cfg = {
        "track": "decision",
        "name": "t",
        "vars": {"base": "org/model", "lr": 1e-4},
        "defaults": {"decision.train": {"lora": {"r": 8, "alpha": 16}, "epochs": 1}},
        "stages": [
            {"name": "data", "kind": "decision.build_data", "params": {"n": 3}},
            {"name": "align", "kind": "decision.train",
             "params": {"init_from": "${vars.base}", "lr": "${vars.lr}", "data": "${data.train}", "lora": {"r": 4}}},
            {"name": "full", "kind": "decision.train",
             "params": {"init_from": "${align.model}", "note": "from ${vars.base} at ${run_dir}"},
             "gate": {"metric": "eval/accuracy", "baseline": "align.eval/accuracy", "max_drop": 0.01}},
        ],
    }
    cfg.update(over)
    return cfg


def test_vars_defaults_and_inferred_needs():
    cfg = build_config(raw())
    align, full = cfg.stage("align"), cfg.stage("full")
    assert align.params["init_from"] == "org/model"
    assert align.params["lr"] == 1e-4  # a whole-string reference keeps the value's type
    assert align.params["lora"] == {"r": 4, "alpha": 16}  # defaults merge under the stage's own params
    assert align.params["epochs"] == 1
    assert align.params["data"] == "${data.train}"  # stage artifacts stay for run time
    assert align.needs == ["data"]
    assert full.needs == ["align"]
    assert full.params["note"] == "from org/model at ${run_dir}"
    assert cfg.run_dir.as_posix() == "runs/decision/t"


def test_gate_baseline_adds_a_dependency():
    cfg = raw()
    cfg["stages"][2]["params"] = {}
    assert build_config(cfg).stage("full").needs == ["align"]


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda c: c["stages"].append({"name": "data", "kind": "decision.x"}), "duplicate stage names"),
        (lambda c: c["stages"][1].update(kind="llm.sft"), "tracks are independent"),
        (lambda c: c["stages"][1]["params"].update(data="${nope.train}"), "not an enabled stage"),
        (lambda c: c["stages"][0].update(needs=["full"]), "cycle"),
        (lambda c: c["stages"][1]["params"].update(x="${vars.missing}"), "not defined under vars"),
        (lambda c: c["stages"][2].update(gate={"metric": "m"}), "no bound"),
        (lambda c: c["stages"][2].update(gate={"metric": "m", "max_drop": 0.1}), "need a baseline"),
        (lambda c: c["stages"][0].update(name="Bad.Name"), "must match"),
        (lambda c: c.update(track="vision"), "track"),
        (lambda c: c["stages"][0].update(enabled=False), "not an enabled stage"),
    ],
)
def test_invalid_configs(mutate, message):
    cfg = raw()
    mutate(cfg)
    with pytest.raises(ConfigError, match=message):
        build_config(cfg)


def test_disabled_stage_is_dropped():
    cfg = raw()
    cfg["stages"][2]["enabled"] = False
    assert [s.name for s in build_config(cfg).stages] == ["data", "align"]


def test_overrides_address_vars_top_level_and_stage_params():
    over = parse_overrides(["vars.base=other/model", "seed=7", "align.lora.r=64", "full.extra=[1, 2]"])
    cfg = build_config(raw(), over)
    assert cfg.seed == 7
    assert cfg.stage("align").params["init_from"] == "other/model"
    assert cfg.stage("align").params["lora"]["r"] == 64
    assert cfg.stage("full").params["extra"] == [1, 2]
    with pytest.raises(ConfigError):
        parse_overrides(["novalue"])


def test_reference_helpers():
    params = {"a": "${s1.model}", "b": ["x ${s2.data} y", {"c": "${vars.k}"}], "d": "${run_dir}/z", "e": 3}
    assert stage_refs(params) == {"s1", "s2"}
    got = resolve_refs(params, lambda ref: {"s1.model": "/m", "s2.data": "/d", "run_dir": "/r"}.get(ref))
    assert got == {"a": "/m", "b": ["x /d y", {"c": "${vars.k}"}], "d": "/r/z", "e": 3}
