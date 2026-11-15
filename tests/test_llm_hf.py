"""The llm stages on the `hf` backend, for real: a tiny random causal LM and a word-level
tokenizer built in the test, trained on CPU with transformers, PEFT and TRL. Nothing is downloaded."""

import json
import os
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("trl")
pytest.importorskip("peft")

from decision_tiny import NEGATIVE, POSITIVE, build_model_dir, build_tokenizer  # noqa: E402

from stagetrain.core import LaunchSpec, Pipeline, PipelineError, build_config, read_jsonl, write_jsonl  # noqa: E402
from stagetrain.core.history import load_history  # noqa: E402
from stagetrain.core.launch import in_worker, is_main_process, read_result, write_result  # noqa: E402
from stagetrain.core.stage import StageResult  # noqa: E402
from stagetrain.tracks.llm.backend import get_backend, is_adapter_dir  # noqa: E402
from stagetrain.tracks.llm.hf import CausalCollator, HfBackend, message_spans, response_labels  # noqa: E402


@pytest.fixture(autouse=True)
def _cpu(monkeypatch):
    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")  # the Trainer would otherwise pick this machine's MPS device


CPU = {"extra": {"use_cpu": True}, "precision": "fp32", "logging_steps": 5}


def conversations(n, seed=0):
    """`the movie was <word>` -> `positive` / `negative`: learnable by a two-layer model in seconds."""
    import random

    rng = random.Random(seed)
    rows = []
    for i in range(n):
        good = rng.random() < 0.5
        word = rng.choice(POSITIVE if good else NEGATIVE)
        filler = rng.choice(["the movie was", "the film was really", "the story was quite", "the acting was"])
        # a running number makes every question a different one (the tokenizer reads most of them as unknown)
        rows.append({"messages": [{"role": "user", "content": f"{filler} {word} {i}"},
                                  {"role": "assistant", "content": "positive" if good else "negative"}]})
    return rows


# ------------------------------------------------------------- loss masking


def test_loss_covers_assistant_turns_and_their_end_token():
    tok = build_tokenizer()
    messages = [
        {"role": "system", "content": "Answer briefly"},
        {"role": "user", "content": "the movie was great"},
        {"role": "assistant", "content": "positive"},
        {"role": "user", "content": "the film was awful"},
        {"role": "assistant", "content": "negative"},
    ]
    text = tok.apply_chat_template(messages, tokenize=False)
    spans = message_spans(text, messages)
    assert [text[a:b] for a, b, _ in spans] == [m["content"] for m in messages]
    assert [role for _, _, role in spans] == ["system", "user", "assistant", "user", "assistant"]

    ids, labels = response_labels(tok, text, messages, max_length=128)
    learned = [tok.decode([i]) for i, label in zip(ids, labels, strict=True) if label != -100]
    assert learned == ["positive", "<|im_end|>", "negative", "<|im_end|>"]  # the answers and where they stop
    assert all(label in (-100, i) for i, label in zip(ids, labels, strict=True))
    _, last = response_labels(tok, text, messages, max_length=128, last_response_only=True)
    assert [tok.decode([i]) for i, label in zip(ids, last, strict=True) if label != -100] == ["negative", "<|im_end|>"]
    # "positive" also occurs in a user turn here: only the assistant's occurrence carries loss
    tricky = [{"role": "user", "content": "is it positive"}, {"role": "assistant", "content": "positive"}]
    rendered = tok.apply_chat_template(tricky, tokenize=False)
    ids, labels = response_labels(tok, rendered, tricky, max_length=128)
    assert sum(label != -100 for label in labels) == 2 and labels[ids.index(tok.convert_tokens_to_ids("positive"))] == -100
    # a row cut before its answer has nothing to learn from
    assert all(label == -100 for label in response_labels(tok, text, messages, max_length=6)[1])

    batch = CausalCollator(tok.pad_token_id)([{"input_ids": [5, 6, 7], "labels": [-100, 6, 7]}, {"input_ids": [5], "labels": [5]}])
    assert batch["input_ids"].tolist()[1] == [5, tok.pad_token_id, tok.pad_token_id]
    assert batch["labels"].tolist()[1] == [5, -100, -100] and batch["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]


# ------------------------------------------------------------- the pipeline


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    os.environ["ACCELERATE_USE_CPU"] = "true"
    tmp = tmp_path_factory.mktemp("llm_hf")
    base = build_model_dir(tmp / "base")
    write_jsonl(tmp / "chat.jsonl", conversations(400))
    model = {"backend": "hf", "max_seq_length": 48, "load_in": "16bit"}
    answers = {**model, "max_new_tokens": 2}  # how every stage generates and judges an answer
    judge = {**answers, "measure_loss": False, "max_samples": 60}
    stages = [
        {"name": "data", "kind": "llm.prepare_data", "params": {
            "chat": [{"source": {"path": str(tmp / "chat.jsonl")}}], "eval_fraction": 0.2, "test_fraction": 0.2,
            "dedupe": False}},
        {"name": "eval_base", "kind": "llm.eval", "params": {**judge, "init_from": str(base), "eval_data": "${data.chat_test}"}},
        # full fine-tuning: every weight of the tiny model, a checkpoint per epoch
        {"name": "sft", "kind": "llm.sft", "params": {
            **model, "init_from": str(base), "full_finetuning": True, "train_data": "${data.chat_train}",
            "eval_data": "${data.chat_eval}",
            "train": {**CPU, "learning_rate": 3e-3, "num_train_epochs": 3, "per_device_train_batch_size": 16,
                      "gradient_accumulation_steps": 1, "warmup": 0.0}}},
        {"name": "pick", "kind": "llm.select_checkpoint", "params": {
            **answers, "init_from": "${sft.checkpoints}", "eval_data": "${data.chat_eval}", "max_samples": 40}},
        {"name": "eval_sft", "kind": "llm.eval", "params": {
            **judge, "init_from": "${pick.model}", "eval_data": "${data.chat_test}", "measure_loss": True,
            "compare_to": "${eval_base.scores}"},
         "gate": {"metric": "eval/delta_ci_low", "min": 0.0}},
        # a LoRA adapter on top of the fine-tuned model, in a process of its own
        {"name": "lora", "kind": "llm.sft", "params": {
            **model, "init_from": "${pick.model}", "train_data": "${data.chat_train}", "max_train_samples": 64,
            "lora": {"r": 4, "alpha": 8}, "launch": {"subprocess": True, "launcher": "python"},
            "train": {**CPU, "learning_rate": 1e-4, "num_train_epochs": 1, "per_device_train_batch_size": 16,
                      "gradient_accumulation_steps": 1, "logging_steps": 2}}},
        {"name": "selfplay", "kind": "llm.generate_data", "params": {
            **model, "init_from": "${pick.model}", "data": "${data.chat_train}", "max_samples": 24,
            "samples_per_prompt": 3, "temperature": 0.5, "max_new_tokens": 2}},
        {"name": "grpo", "kind": "llm.grpo", "params": {
            **model, "init_from": "${lora.model}", "train_data": "${data.chat_train}", "max_train_samples": 16,
            "num_generations": 2, "max_completion_length": 4, "lora": {"r": 4, "alpha": 8},
            "rewards": [{"kind": "correct"}, {"kind": "length", "max_chars": 20, "weight": 0.1}],
            "train": {**CPU, "per_device_train_batch_size": 4, "max_steps": 3, "save_steps": None, "learning_rate": 1e-5,
                      "logging_steps": 1}}},
        {"name": "export", "kind": "llm.export", "params": {**model, "init_from": "${grpo.model}", "formats": ["lora", "merged_16bit"]}},
    ]
    pipeline = Pipeline(build_config({"track": "llm", "name": "hf", "output_dir": str(tmp / "runs"), "seed": 3, "stages": stages}))
    pipeline.run()
    return pipeline


def test_sft_teaches_the_task_and_the_paired_eval_shows_it(run):
    base, tuned = run.manifest("eval_base").metrics, run.manifest("eval_sft").metrics
    assert base["eval/score"] < 0.3  # a random model does not answer in the label vocabulary
    assert tuned["eval/score"] >= 0.9 and tuned["eval/delta"] >= 0.6 and tuned["eval/delta_ci_low"] > 0
    assert tuned["eval/delta_n"] == tuned["eval/generated"] == 60  # every question compared with itself
    assert tuned["eval/loss"] < 3.0 and tuned["eval/no_answer"] == 0.0
    sft = run.manifest("sft")
    assert sft.notes["mode"] == "full" and sft.notes["backend"] == "hf"
    assert sft.metrics["train/loss"] < 2.5 and sft.metrics["eval/loss"] < 1.0 and sft.metrics["train/dropped_overlength"] == 0
    history = load_history(run.run_dir / "sft")
    losses = [r["train/loss"] for r in history if "train/loss" in r]
    assert len(losses) >= 5 and losses[-1] < losses[0] / 2  # the curve the report draws goes down
    assert len([r for r in history if "eval/loss" in r]) >= 3  # one evaluation per epoch


def test_checkpoints_are_judged_by_generated_answers(run):
    pick = run.manifest("pick")
    table = pick.notes["checkpoints"]
    assert len(table) == 3 and [row["step"] for row in table] == sorted(row["step"] for row in table)
    assert pick.metrics["eval/score"] == max(row["score"] for row in table) >= 0.9
    best = [row["step"] for row in table if row["score"] == pick.metrics["eval/score"]]
    assert pick.metrics["select/chosen_step"] == min(best)  # among equally good checkpoints, the least trained
    assert (run.run_dir / "pick" / "model" / "config.json").is_file()  # a full model was saved again


def test_a_lora_stage_runs_in_a_worker_process(run):
    lora = run.manifest("lora")
    assert lora.completed and lora.notes["launch"] == {"num_processes": 1, "launcher": "python"}
    assert lora.notes["mode"] == "adapter:new" and is_adapter_dir(run.run_dir / "lora" / "model")
    job = json.loads((run.run_dir / "lora" / "job.json").read_text())
    assert job["kind"] == "llm.sft" and job["params"]["backend"] == "hf" and job["ctx"]["stage"] == "lora"
    assert lora.metrics["train/rows"] == 64 and lora.artifacts["model"] == "run://lora/model"
    assert load_history(run.run_dir / "lora")  # the worker wrote the history the report reads


def test_selfplay_and_grpo_run(run):
    gen = run.manifest("selfplay").metrics
    assert gen["generate/samples"] == 72 and 0 <= gen["generate/mixed_fraction"] <= 1 and gen["generate/pass_at_k"] >= 0.7
    rows = list(read_jsonl(run.run_dir / "selfplay" / "sft.jsonl"))
    assert rows and rows[0]["messages"][-1]["content"] in ("positive", "negative")
    grpo = run.manifest("grpo")
    assert grpo.notes["mode"] == "adapter:continue"  # it went on training the LoRA of the stage before
    assert {"train/reward", "train/reward_start", "train/reward_gain"} <= set(grpo.metrics)
    logged = load_history(run.run_dir / "grpo")
    assert len(logged) >= 3 and "train/rewards/correct/mean" in logged[0] and "train/rewards/length/mean" in logged[0]
    assert 0.0 <= logged[0]["train/rewards/correct/mean"] <= 1.0


def test_export_and_a_second_run(run):
    export = run.manifest("export")
    assert is_adapter_dir(run.run_dir / "export" / "lora") and (run.run_dir / "export" / "merged_16bit" / "config.json").is_file()
    assert export.artifacts["model"] == "run://export/merged_16bit"
    assert {s.state for s in Pipeline(run.config).status()} == {"cached"}


# ------------------------------------------------------- backend and launch


def test_backend_choice_and_its_limits(tmp_path):
    assert isinstance(get_backend("hf"), HfBackend) and get_backend("hf") is get_backend("hf")
    with pytest.raises(ValueError, match="backend must be one of"):
        get_backend("other")
    hf = get_backend("hf")
    with pytest.raises(ValueError, match="unsloth backend"):
        hf.save_gguf(None, None, tmp_path, ["q4_k_m"])
    with pytest.raises(ValueError, match="merged_16bit only"):
        hf.save_merged(None, None, tmp_path, "merged_4bit")
    with pytest.raises(ValueError, match="Unsloth's named templates"):
        hf.set_chat_template(None, "chatml")

    def config(**params):
        return build_config({"track": "llm", "name": "x", "output_dir": str(tmp_path), "stages": [
            {"name": "grpo", "kind": "llm.grpo", "params": {"init_from": "m", "train_data": "d.jsonl", **params}}]})

    with pytest.raises(PipelineError, match="needs backend: unsloth"):
        Pipeline(config(backend="hf", fast_inference=True)).validate()
    # eight completions per prompt need a step of 8, 16, ...: four GPUs with a batch of 2 each fit
    Pipeline(config(train={"per_device_train_batch_size": 2}, launch={"num_processes": 4})).validate()
    with pytest.raises(PipelineError, match="processes = 6 must be a multiple"):
        Pipeline(config(train={"per_device_train_batch_size": 2}, launch={"num_processes": 3})).validate()


def test_launch_commands():
    python = sys.executable
    assert not LaunchSpec().active and LaunchSpec(subprocess=True).active and LaunchSpec(num_processes=2).active
    assert LaunchSpec(num_processes=4).command("job.json") == [  # torchrun by default, as TRL documents
        python, "-m", "torch.distributed.run", "--nproc_per_node", "4", "-m", "stagetrain.worker", "job.json"]
    assert LaunchSpec(num_processes=4, launcher="accelerate", gpu_ids="0,1,2,3").command("job.json") == [
        python, "-m", "accelerate.commands.accelerate_cli", "launch", "--num_processes", "4", "--multi_gpu",
        "--gpu_ids", "0,1,2,3", "-m", "stagetrain.worker", "job.json"]
    assert LaunchSpec(num_processes=2, launcher="accelerate", extra_args=["--mixed_precision", "bf16"]).command("j")[4:] == [
        "--num_processes", "2", "--multi_gpu", "--mixed_precision", "bf16", "-m", "stagetrain.worker", "j"]
    assert LaunchSpec(subprocess=True, launcher="python").command("j") == [python, "-m", "stagetrain.worker", "j"]
    with pytest.raises(ValueError, match="starts one process"):
        LaunchSpec(num_processes=2, launcher="python").command("j")
    env = LaunchSpec(num_processes=2, gpu_ids="2,3", env={"X": "1"}).environment()
    assert env["STAGETRAIN_WORKER"] == "1" and env["CUDA_VISIBLE_DEVICES"] == "2,3" and env["X"] == "1"
    assert "STAGETRAIN_FSDP" not in env and LaunchSpec(num_processes=2, fsdp=True).environment()["STAGETRAIN_FSDP"] == "1"


def test_fsdp_is_switched_on_in_the_training_arguments(monkeypatch, tmp_path):
    from transformers import TrainingArguments

    base = {"output_dir": str(tmp_path), "report_to": "none"}
    plain = HfBackend._arguments(dict(base), TrainingArguments)
    assert plain.ddp_find_unused_parameters is False and not plain.fsdp and plain.use_cpu
    monkeypatch.setenv("STAGETRAIN_WORKER", "1")
    monkeypatch.setenv("STAGETRAIN_FSDP", "1")
    sharded = HfBackend._arguments(dict(base), TrainingArguments)
    assert sharded.fsdp and sharded.fsdp_config["version"] == 2


def test_only_the_main_process_writes(tmp_path, monkeypatch):
    assert not in_worker() and is_main_process()
    monkeypatch.setenv("STAGETRAIN_WORKER", "1")
    assert in_worker() and is_main_process()  # a single-process worker
    monkeypatch.setenv("RANK", "0")
    assert is_main_process()
    monkeypatch.setenv("RANK", "1")
    assert not is_main_process()  # a secondary process of a distributed run
    write_result(tmp_path, StageResult(artifacts={"model": tmp_path / "model"}, metrics={"train/loss": 1}, notes={"a": 1}))
    got = read_result(tmp_path)
    assert got.artifacts == {"model": str(tmp_path / "model")} and got.metrics == {"train/loss": 1.0} and got.notes == {"a": 1}
    with pytest.raises(RuntimeError, match="left no result.json"):
        read_result(tmp_path / "missing")
