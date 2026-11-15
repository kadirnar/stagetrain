"""The llm stages wired through `Pipeline` with a fake backend: no Unsloth, no weights, no network."""

import json
from pathlib import Path

import pytest

from stagetrain.core import (
    GateFailed,
    Pipeline,
    PipelineError,
    build_config,
    load_config,
    read_jsonl,
    write_jsonl,
)
from stagetrain.tracks.llm import set_backend
from stagetrain.tracks.llm.backend import TrainOutput, is_adapter_dir
from stagetrain.tracks.llm.evaluate import score

pytest.importorskip("datasets")  # llm.prepare_data reads its sources through `datasets`

CONFIGS = sorted((Path(__file__).parent.parent / "configs" / "llm").glob("*.yaml"))


class FakeTokenizer:
    eos_token = "</s>"
    chat_template = "fake"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert tokenize is False and add_generation_prompt is False
        return "".join(f"<|{m['role']}|>{m['content']}</s>" for m in messages)


class FakeModel:
    def __init__(self, name, adapter, full):
        self.name, self.adapter, self.full, self.lora = name, adapter, full, None


class FakeBackend:
    """Records what the stages ask for and writes the files a real backend would leave behind."""

    def __init__(self, eval_loss=None):
        self.calls = []
        self.eval_loss = eval_loss or {}  # substring of the model's name -> loss

    def _loss(self, model):
        for key, value in self.eval_loss.items():
            if key in model.name:
                return value
        return 1.0

    def load(self, spec):
        self.calls.append(("load", spec.model_name, spec.load_in, spec.full_finetuning))
        return FakeModel(spec.model_name, is_adapter_dir(spec.model_name), spec.full_finetuning), FakeTokenizer()

    def has_adapter(self, model):
        return model.adapter

    def attach_lora(self, model, kwargs):
        self.calls.append(("attach_lora", kwargs))
        model.adapter, model.lora = True, kwargs
        return model

    def set_chat_template(self, tokenizer, name):
        self.calls.append(("chat_template", name))
        tokenizer.chat_template = name
        return tokenizer

    def bf16_supported(self):
        return True

    def release(self):
        self.calls.append(("release",))

    def train_text(self, model, tokenizer, train_texts, eval_texts, config, *, responses_only=None, resume=True):
        self.calls.append(("train_text", {
            "model": model.name, "train": list(train_texts), "eval": list(eval_texts), "config": config,
            "responses_only": responses_only, "resume": resume,
        }))
        metrics = {"train/loss": 0.9}
        if eval_texts:
            metrics["eval/loss"] = self._loss(model)
        return TrainOutput(metrics, [{"step": 1, "loss": 0.9}])

    def train_dpo(self, model, tokenizer, train_rows, eval_rows, config, *, resume=True):
        self.calls.append(("train_dpo", {"model": model.name, "train": list(train_rows), "config": config}))
        return TrainOutput({"train/loss": 0.6, "eval/loss": 0.5} if eval_rows else {"train/loss": 0.6}, [])

    def evaluate_loss(self, model, tokenizer, texts, config, *, responses_only=None):
        self.calls.append(("evaluate_loss", {"model": model.name, "texts": list(texts), "responses_only": responses_only}))
        return {"eval/loss": self._loss(model), "eval/runtime": 1.0}

    def generate(self, model, tokenizer, conversations, *, max_new_tokens=256, temperature=0.0, top_p=1.0):
        self.calls.append(("generate", len(conversations)))
        out = []
        for i, messages in enumerate(conversations):
            assert messages[-1]["role"] == "user"  # the reference answer is never shown to the model
            asked = messages[-1]["content"]
            truth = asked.replace("Question", "Answer").rstrip("?") if asked.startswith("Question") else asked.replace("Do", "Done")
            # the first answer is exactly right, the second only contains it, the rest are wrong
            out.append([f"<think>hmm</think> {truth}", f"Sure: {truth.lower()}!", "no idea"][min(i, 2)])
        return out

    def save(self, model, tokenizer, out_dir):
        self.calls.append(("save", str(out_dir)))
        name = "adapter_config.json" if model.adapter else "config.json"
        saved = (model.lora or {}).get("modules_to_save")
        Path(out_dir, name).write_text(json.dumps({"base_model_name_or_path": model.name, "modules_to_save": saved}))

    def save_merged(self, model, tokenizer, out_dir, method="merged_16bit"):
        self.calls.append(("save_merged", str(out_dir), method))
        Path(out_dir, "config.json").write_text(json.dumps({"merged_from": model.name, "method": method}))

    def save_gguf(self, model, tokenizer, out_dir, quantization):
        self.calls.append(("save_gguf", str(out_dir), list(quantization)))
        directory = Path(f"{out_dir}_gguf")
        directory.mkdir(parents=True, exist_ok=True)
        files = [str(directory / f"model.{q}.gguf") for q in quantization]
        for f in files:
            Path(f).write_bytes(b"gguf")
        return str(directory), files

    def push(self, model, tokenizer, repo_id, method, *, private=True, quantization=None):
        self.calls.append(("push", repo_id, method, private))

    def of(self, name):
        return [c for c in self.calls if c[0] == name]


@pytest.fixture
def backend():
    fake = FakeBackend()
    set_backend(fake)
    yield fake
    set_backend(None)


@pytest.fixture
def sources(tmp_path):
    chat = [{"messages": [{"role": "user", "content": f"Question {i}?"}, {"role": "assistant", "content": f"Answer {i}"}]}
            for i in range(60)]
    chat += [chat[0], {"messages": [{"role": "assistant", "content": "orphan"}]}]  # a duplicate and a bad row
    write_jsonl(tmp_path / "chat.jsonl", chat)
    write_jsonl(tmp_path / "alpaca.jsonl", [{"instruction": f"Do {i}", "input": "", "output": f"Done {i}"} for i in range(20)])
    write_jsonl(tmp_path / "corpus.jsonl", [{"text": f"Document {i}. " + "Domain sentence. " * 30} for i in range(12)])
    write_jsonl(tmp_path / "prefs.jsonl", [{"prompt": f"Pick {i}", "chosen": f"good {i}", "rejected": f"bad {i}"} for i in range(30)])
    return tmp_path


def pipeline(tmp_path, **over):
    stages = [
        {"name": "data", "kind": "llm.prepare_data", "params": {
            "chat": [{"source": {"path": str(tmp_path / "chat.jsonl")}, "system": "Be brief."},
                     {"source": {"path": str(tmp_path / "alpaca.jsonl")}, "format": "alpaca"}],
            "text": [{"source": {"path": str(tmp_path / "corpus.jsonl")}, "chunk_chars": 200}],
            "preference": [{"source": {"path": str(tmp_path / "prefs.jsonl")}}],
            "eval_fraction": 0.1,
        }},
        {"name": "eval_base", "kind": "llm.eval",
         "params": {"init_from": "org/base", "eval_data": "${data.chat_eval}", "generate_samples": 0}},
        {"name": "cpt", "kind": "llm.cpt",
         "params": {"init_from": "org/base", "train_data": "${data.text_train}", "eval_data": "${data.text_eval}"}},
        {"name": "sft_broad", "kind": "llm.sft", "params": {
            "init_from": "${cpt.model}", "on_adapter": "merge", "train_data": "${data.chat_train}",
            "eval_data": "${data.chat_eval}", "lora": {"r": 8}, "train": {"num_train_epochs": 2}},
         "gate": {"metric": "eval/loss", "baseline": "eval_base.eval/loss", "max_rise": 0.0}},
        {"name": "sft_domain", "kind": "llm.sft", "params": {
            "init_from": "${sft_broad.model}", "train_data": "${data.chat_train}", "max_train_samples": 5,
            "responses_only": False}},
        {"name": "dpo", "kind": "llm.dpo", "params": {
            "init_from": "${sft_domain.model}", "train_data": "${data.preference_train}",
            "eval_data": "${data.preference_eval}", "beta": 0.2}},
        {"name": "eval", "kind": "llm.eval", "params": {"init_from": "${dpo.model}", "eval_data": "${data.chat_eval}",
                                                       "generate_samples": 4}},
        {"name": "export", "kind": "llm.export", "needs": ["eval"], "params": {
            "init_from": "${dpo.model}", "formats": ["gguf", "merged_16bit", "lora"], "gguf_quantization": ["q4_k_m", "q8_0"]}},
    ]
    for s in stages:
        s["params"].update(over.get(s["name"], {}))
        if f"{s['name']}_gate" in over:
            s["gate"] = over[f"{s['name']}_gate"]
    return Pipeline(build_config({"track": "llm", "name": "t", "output_dir": str(tmp_path / "runs"), "seed": 5, "stages": stages}))


def test_prepare_data_normalises_dedupes_and_splits(sources, backend):
    p = pipeline(sources)
    m = p.run(only=["data"])[0]
    run = sources / "runs" / "llm" / "t" / "data"
    train, held_out = list(read_jsonl(run / "chat.train.jsonl")), list(read_jsonl(run / "chat.eval.jsonl"))
    assert len(train) + len(held_out) == 80 and held_out  # 60 + 20, the duplicate and the orphan dropped
    assert m.metrics["chat/duplicates"] == 1 and m.metrics["chat/rejected"] == 1
    assert m.metrics["chat/train_rows"] == len(train) and m.metrics["chat/eval_rows"] == len(held_out)
    systems = [r["messages"][0] for r in train + held_out if r["messages"][0]["role"] == "system"]
    assert len(systems) == 60 and systems[0]["content"] == "Be brief."  # only the source that asked for one
    text = list(read_jsonl(run / "text.train.jsonl")) + list(read_jsonl(run / "text.eval.jsonl"))
    assert len(text) > 12 and all(len(r["text"]) <= 200 for r in text)  # documents were chunked
    pref = next(read_jsonl(run / "preference.train.jsonl"))
    assert pref["prompt"][0]["role"] == "user" and pref["chosen"][0]["role"] == "assistant"
    stats = json.loads((run / "stats.json").read_text())
    assert stats["chat"]["rejected"] == {"conversation does not start with a user message": 1}
    assert set(m.artifacts) == {"chat_train", "chat_eval", "text_train", "text_eval", "preference_train",
                                "preference_eval", "stats"}
    assert backend.calls == []  # no model is touched


def test_prepare_data_rejects_bad_configs(sources, backend):
    with pytest.raises(PipelineError, match="at least one source"):
        pipeline(sources, data={"chat": [], "text": [], "preference": []}).validate()
    with pytest.raises(ValueError, match="no usable row"):
        pipeline(sources, data={"min_chars": 10**6}).run(only=["data"])


def test_full_pipeline_chains_models_and_reports_metrics(sources, backend):
    backend.eval_loss = {"org/base": 2.0, "sft_broad": 1.2, "dpo": 1.1}
    p = pipeline(sources)
    manifests = {m.name: m for m in p.run()}
    run = sources / "runs" / "llm" / "t"

    # cpt: a new adapter on the base model, embeddings trained at a tenth of the learning rate, raw text + EOS
    cpt_lora = backend.of("attach_lora")[0][1]
    assert cpt_lora["r"] == 128 and cpt_lora["modules_to_save"] == ["embed_tokens", "lm_head"] and cpt_lora["random_state"] == 5
    cpt_call = backend.of("train_text")[0][1]
    assert cpt_call["model"] == "org/base" and cpt_call["responses_only"] is None
    assert all(t.endswith("</s>") and "<|user|>" not in t for t in cpt_call["train"])
    assert cpt_call["config"]["learning_rate"] == 5e-5 and cpt_call["config"]["embedding_learning_rate"] == pytest.approx(5e-6)
    assert cpt_call["config"]["output_dir"] == str(run / "cpt" / "checkpoints")
    assert manifests["cpt"].notes["mode"] == "adapter:new" and is_adapter_dir(run / "cpt" / "model")

    # sft_broad (merge): the cpt adapter is folded into a 16-bit base, then a fresh small adapter is attached
    assert ("save_merged", str(run / "sft_broad" / "merged_base"), "merged_16bit") in backend.calls
    assert ("load", str(run / "sft_broad" / "merged_base"), "4bit", False) in backend.calls
    sft_lora = backend.of("attach_lora")[1][1]
    assert sft_lora["r"] == 8 and sft_lora["lora_alpha"] == 16 and sft_lora["modules_to_save"] is None
    sft_call = backend.of("train_text")[1][1]
    assert sft_call["train"][0].startswith("<|system|>Be brief.</s><|user|>") or sft_call["train"][0].startswith("<|user|>")
    assert sft_call["responses_only"] is not None and sft_call["responses_only"].instruction_part is None
    assert sft_call["config"]["num_train_epochs"] == 2 and sft_call["config"]["learning_rate"] == 2e-4
    assert "embedding_learning_rate" not in sft_call["config"]
    assert manifests["sft_broad"].notes["mode"] == "adapter:merge"
    assert manifests["sft_broad"].artifacts["base"] == "run://sft_broad/merged_base"
    assert manifests["sft_broad"].metrics["eval/loss"] == 1.2
    assert manifests["sft_broad"].metrics["eval/perplexity"] == pytest.approx(3.3201169)

    # sft_domain (continue): the adapter is loaded and trained on; no new LoRA, no merge
    assert manifests["sft_domain"].notes["mode"] == "adapter:continue"
    assert [c[1] for c in backend.of("save_merged")][:2] == [str(run / "sft_broad" / "merged_base"), str(run / "dpo" / "merged_base")]
    domain_call = backend.of("train_text")[2][1]
    assert domain_call["model"] == str(run / "sft_broad" / "model") and len(domain_call["train"]) == 5
    assert domain_call["responses_only"] is None and domain_call["eval"] == []
    assert "eval/loss" not in manifests["sft_domain"].metrics and manifests["sft_domain"].metrics["train/rows"] == 5

    # dpo: merges by default, so the reference policy is the new adapter's base; conversational rows
    dpo_call = backend.of("train_dpo")[0][1]
    assert dpo_call["model"] == str(run / "dpo" / "merged_base") and manifests["dpo"].notes["mode"] == "adapter:merge"
    assert backend.of("attach_lora")[2][1]["r"] == 16
    assert dpo_call["config"]["beta"] == 0.2 and dpo_call["config"]["learning_rate"] == 5e-6 and "loss_type" not in dpo_call["config"]
    assert set(dpo_call["train"][0]) == {"prompt", "chosen", "rejected"}

    # eval: loss, perplexity and generation scores, with samples on disk
    final = manifests["eval"].metrics
    assert final["eval/loss"] == 1.1 and final["eval/generated"] == 4
    assert final["eval/exact_match"] == 0.25 and final["eval/contains"] == 0.5
    samples = list(read_jsonl(run / "eval" / "samples.jsonl"))
    assert samples[0]["exact"] and samples[1]["contains"] and not samples[1]["exact"] and not samples[2]["contains"]
    assert samples[0]["prompt"][-1]["role"] == "user"
    assert backend.of("generate") == [("generate", 4)]  # eval_base asked for none

    # export: every format, in a safe order, nothing pushed
    saves = [c[0] for c in backend.calls if c[0] in ("save", "save_merged", "save_gguf")][-3:]
    assert saves == ["save", "save_merged", "save_gguf"]
    export = manifests["export"]
    assert export.artifacts["model"] == export.artifacts["merged_16bit"] == "run://export/merged_16bit"
    assert export.artifacts["gguf"] == "run://export/gguf_gguf" and export.metrics["export/gguf_files"] == 2
    assert backend.of("push") == []
    # every loaded model is released again
    assert len(backend.of("release")) == len(backend.of("load"))


def test_finished_stages_are_skipped_and_a_changed_stage_reruns_downstream(sources, backend):
    pipeline(sources).run()
    backend.calls.clear()
    pipeline(sources).run()
    assert backend.calls == []
    p = pipeline(sources, dpo={"beta": 0.5})
    states = {s.name: s.state for s in p.status()}
    assert states["sft_domain"] == "cached" and states["dpo"] == "stale" and states["export"] == "stale"
    p.run()
    assert [c[0] for c in backend.calls if c[0].startswith("train")] == ["train_dpo"]


def test_gate_against_the_base_model_stops_the_pipeline(sources, backend):
    backend.eval_loss = {"org/base": 1.0, "sft_broad": 1.5}  # fine-tuning made held-out loss worse
    with pytest.raises(GateFailed, match="rose more than 0"):
        pipeline(sources).run()
    assert backend.of("train_dpo") == [] and len(backend.of("train_text")) == 2


def test_continuing_an_adapter_with_trained_embeddings_keeps_the_slow_embedding_rate(sources, backend):
    p = pipeline(sources, sft_broad={"on_adapter": "continue", "train": {"learning_rate": 1e-4}})
    p.run(until="sft_broad")
    call = backend.of("train_text")[1][1]
    assert call["config"]["embedding_learning_rate"] == pytest.approx(1e-5)
    assert len(backend.of("attach_lora")) == 1 and backend.of("save_merged") == []
    explicit = pipeline(sources, sft_broad={"on_adapter": "continue", "embedding_learning_rate": 3e-6})
    explicit.run(until="sft_broad")
    assert backend.of("train_text")[-1][1]["config"]["embedding_learning_rate"] == 3e-6


def test_full_finetuning_merges_an_adapter_and_attaches_no_lora(sources, backend):
    p = pipeline(sources, sft_broad={"full_finetuning": True, "load_in": "16bit", "on_adapter": "continue"})
    m = p.run(until="sft_broad")[-1]
    assert m.notes["mode"] == "full" and len(backend.of("attach_lora")) == 1  # only cpt's
    run = sources / "runs" / "llm" / "t" / "sft_broad"
    assert ("load", str(run / "merged_base"), "16bit", True) in backend.calls
    assert (run / "model" / "config.json").is_file() and not is_adapter_dir(run / "model")


def test_export_pushes_only_when_asked_and_needs_an_adapter_for_lora(sources, backend):
    p = pipeline(sources, export={"push_to_hub": "user/model", "private": False})
    p.run()
    assert backend.of("push") == [
        ("push", "user/model-lora", "lora", False), ("push", "user/model", "merged_16bit", False),
        ("push", "user/model-GGUF", "gguf", False),
    ]
    with pytest.raises(ValueError, match="no 'lora' to export"):
        pipeline(sources, export={"init_from": "org/base", "formats": ["lora"]}).run(only=["export"], allow_stale=True)


def test_chat_template_and_missing_data(sources, backend):
    p = pipeline(sources, sft_domain={"chat_template": "chatml", "train_data": str(sources / "nope.jsonl")})
    with pytest.raises(FileNotFoundError, match="no data file"):
        p.run(until="sft_domain")
    assert ("chat_template", "chatml") not in backend.calls  # data is read before any model is loaded
    p = pipeline(sources, sft_domain={"chat_template": "chatml"})
    p.run(until="sft_domain")
    assert ("chat_template", "chatml") in backend.calls


@pytest.mark.parametrize(
    "stage, params, message",
    [
        ("sft_broad", {"instruction_part": "<|user|>"}, "both instruction_part and response_part"),
        ("sft_broad", {"full_finetuning": True}, "needs load_in: 16bit"),
        ("sft_broad", {"on_adapter": "restart"}, "on_adapter"),
        ("sft_broad", {"lora": {"rank": 8}}, "rank"),
        ("dpo", {"embedding_learning_rate": 1e-6}, "one learning rate"),
        ("export", {"formats": []}, "at least one format"),
        ("export", {"formats": ["gguf"], "gguf_quantization": []}, "gguf_quantization"),
        ("eval", {"temperature": -1}, "temperature"),
    ],
)
def test_bad_parameters_fail_before_anything_runs(sources, backend, stage, params, message):
    with pytest.raises(PipelineError, match=message):
        pipeline(sources, **{stage: params}).validate()
    assert backend.calls == []


def test_score():
    assert score("Paris", "  paris ") == {"exact": True, "contains": True}
    assert score("Paris", "It is Paris.") == {"exact": False, "contains": True}
    assert score("Paris", "<think>Paris?</think>London") == {"exact": False, "contains": False}
    assert score("Paris", "<think>x</think>London", strip_thinking=False) == {"exact": False, "contains": False}
    assert score("", "anything") == {"exact": False, "contains": False}


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_shipped_configs_validate(path):
    p = Pipeline(load_config(path))
    p.validate()
    assert p.config.track == "llm" and [s.state for s in p.status()]


def test_configs_are_shipped():
    assert CONFIGS


def test_importing_the_track_needs_no_training_library():
    import subprocess
    import sys

    code = (
        "import sys, stagetrain.tracks.llm\n"
        "from stagetrain.core import kinds\n"
        "assert set(kinds('llm')) == {'llm.prepare_data', 'llm.cpt', 'llm.sft', 'llm.dpo', 'llm.eval', 'llm.export'}\n"
        "heavy = [m for m in ('torch', 'transformers', 'trl', 'unsloth', 'datasets', 'peft') if m in sys.modules]\n"
        "assert not heavy, heavy\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
