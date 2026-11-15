"""`llm.axolotl` and `llm.swift`: the config a stage writes, the command it runs, and what it
reads back. The libraries themselves are not installed here; a small script stands in for their
commands, so this checks the stages' side of the contract, not the libraries'."""

import json
import sys
from pathlib import Path

import pytest
import yaml

from stagetrain.core import Pipeline, PipelineError, build_config, read_jsonl, write_jsonl
from stagetrain.core.history import load_history
from stagetrain.tracks.llm.backend import is_adapter_dir
from stagetrain.tracks.llm.external import last_checkpoint, split_init, trained_model

FAKE = [sys.executable, str(Path(__file__).with_name("fake_trainer_cli.py"))]


@pytest.fixture
def data(tmp_path):
    write_jsonl(tmp_path / "chat.jsonl", [
        {"messages": [{"role": "user", "content": f"Question {i}?"}, {"role": "assistant", "content": f"Answer {i}"}]}
        for i in range(8)])
    write_jsonl(tmp_path / "text.jsonl", [{"text": f"Document {i}."} for i in range(4)])
    write_jsonl(tmp_path / "pref.jsonl", [
        {"prompt": [{"role": "system", "content": "Be kind."}, {"role": "user", "content": f"Pick {i}"}],
         "chosen": [{"role": "assistant", "content": f"good {i}"}], "rejected": [{"role": "assistant", "content": f"bad {i}"}]}
        for i in range(5)])
    adapter = tmp_path / "earlier" / "model"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "org/base"}))
    return tmp_path


def run(data, kind, **params):
    base = {"init_from": "org/base", "train_data": str(data / "chat.jsonl"), "executable": FAKE}
    p = Pipeline(build_config({"track": "llm", "name": "ext", "output_dir": str(data / "runs"), "seed": 9, "stages": [
        {"name": "train", "kind": kind, "params": {**base, **params}}]}))
    manifest = p.run()[0]
    stage = p.run_dir / "train"
    return manifest, json.loads((stage / "seen.json").read_text()), stage


def test_axolotl_sft_on_four_gpus(data):
    m, seen, stage = run(data, "llm.axolotl", num_processes=4, gpu_ids="0,1,2,3", eval_data=str(data / "chat.jsonl"),
                         lora={"r": 64, "alpha": 32}, train={"learning_rate": 1e-4, "num_train_epochs": 2, "warmup": 0.1},
                         config={"sample_packing": True, "learning_rate": 5e-5}, env={"EXTRA": "x"})
    cfg = seen["config"]
    assert seen["argv"] == ["train", str(stage / "config.yaml"), "--num-processes", "4"] and seen["cwd"] == str(stage)
    assert seen["env"] == {"AXOLOTL_DO_NOT_TRACK": "1", "NPROC_PER_NODE": None, "CUDA_VISIBLE_DEVICES": "0,1,2,3",
                           "HF_HUB_DISABLE_TELEMETRY": "1", "EXTRA": "x"}
    assert cfg["base_model"] == "org/base" and cfg["adapter"] == "lora" and cfg["lora_r"] == 64 and cfg["lora_alpha"] == 32
    assert cfg["lora_target_modules"] == ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    assert cfg["datasets"] == [{
        "path": str(data / "chat.jsonl"), "ds_type": "json", "type": "chat_template", "field_messages": "messages",
        "message_property_mappings": {"role": "role", "content": "content"}}]
    assert cfg["chat_template"] == "tokenizer_default" and cfg["sequence_len"] == 2048 and cfg["seed"] == 9
    assert cfg["num_epochs"] == 2 and cfg["warmup_ratio"] == 0.1 and cfg["micro_batch_size"] == 2
    assert cfg["sample_packing"] is True and cfg["learning_rate"] == 5e-5  # `config` is written as it is and wins
    assert cfg["test_datasets"][0]["path"] == str(data / "chat.jsonl") and cfg["val_set_size"] == 0
    assert "rl" not in cfg and "lora_model_dir" not in cfg and "load_in_4bit" not in cfg
    assert yaml.safe_load((stage / "config.yaml").read_text()) == cfg

    assert m.artifacts["model"] == "run://train/output" and is_adapter_dir(stage / "output")
    assert m.artifacts["checkpoints"] == "run://train/output" and m.notes["tool"] == "axolotl"
    # train/loss is the run's mean, as the other training stages report it; the curve is in the history
    assert m.metrics["train/loss"] == 1.3 and m.metrics["eval/loss"] == 1.2 and m.metrics["train/runtime"] == 3.0
    history = load_history(stage)  # the same file the report reads for every other training stage
    assert [r["train/loss"] for r in history if "train/loss" in r] == [1.5, 1.1] and history[-1]["eval/loss"] == 1.2


def test_axolotl_continues_an_adapter_and_formats_dpo_and_raw_text(data):
    _, seen, stage = run(data, "llm.axolotl", method="dpo", init_from=str(data / "earlier" / "model"),
                         train_data=str(data / "pref.jsonl"), load_in="4bit")
    cfg = seen["config"]
    assert cfg["rl"] == "dpo" and cfg["base_model"] == "org/base" and cfg["adapter"] == "qlora" and cfg["load_in_4bit"] is True
    assert cfg["lora_model_dir"] == str((data / "earlier" / "model").resolve())
    assert cfg["datasets"][0]["type"] == "chat_template.default" and cfg["datasets"][0]["field_chosen"] == "chosen"
    rows = list(read_jsonl(cfg["datasets"][0]["path"]))
    assert rows[0] == {"messages": [{"role": "system", "content": "Be kind."}, {"role": "user", "content": "Pick 0"}],
                       "chosen": {"role": "assistant", "content": "good 0"}, "rejected": {"role": "assistant", "content": "bad 0"}}
    assert "chat_template" not in cfg and seen["argv"][-1] == "1" and seen["env"]["CUDA_VISIBLE_DEVICES"] is None

    _, seen, _ = run(data, "llm.axolotl", method="cpt", train_data=str(data / "text.jsonl"), full_finetuning=True,
                     train={"max_steps": 50, "warmup": 100})
    cfg = seen["config"]
    assert cfg["datasets"] == [{"path": str(data / "text.jsonl"), "ds_type": "json", "type": "completion"}]
    assert "adapter" not in cfg and "lora_r" not in cfg and cfg["max_steps"] == 50 and cfg["warmup_steps"] == 100


def test_swift_sft_dpo_and_pretraining(data):
    m, seen, stage = run(data, "llm.swift", num_processes=2, gpu_ids="0,1", eval_data=str(data / "chat.jsonl"),
                         lora={"r": 16, "alpha": 32, "target_modules": "all-linear"}, load_in="4bit",
                         train={"save_steps": 100, "save_total_limit": 2})
    cfg = seen["config"]
    assert seen["argv"] == ["sft", str(stage / "config.yaml")]
    assert seen["env"]["NPROC_PER_NODE"] == "2" and seen["env"]["CUDA_VISIBLE_DEVICES"] == "0,1"  # DDP, not a split model
    assert seen["env"]["AXOLOTL_DO_NOT_TRACK"] is None
    assert cfg["model"] == "org/base" and cfg["use_hf"] is True and cfg["tuner_type"] == "lora"
    assert cfg["dataset"] == [str(data / "chat.jsonl")] and cfg["val_dataset"] == [str(data / "chat.jsonl")]
    assert cfg["lora_rank"] == 16 and cfg["target_modules"] == ["all-linear"] and cfg["quant_method"] == "bnb" and cfg["quant_bits"] == 4
    assert cfg["add_version"] is False and cfg["create_checkpoint_symlink"] is True and cfg["max_length"] == 2048
    assert cfg["save_steps"] == cfg["eval_steps"] == 100 and cfg["save_total_limit"] == 2 and cfg["warmup_ratio"] == 0.05
    # ms-swift leaves checkpoints and a `last` link: that is the model
    assert m.artifacts["model"] == "run://train/output/checkpoint-20" and is_adapter_dir(stage / "output" / "checkpoint-20")
    assert m.metrics["train/loss"] == 1.3 and m.notes["tool"] == "ms-swift"

    _, seen, _ = run(data, "llm.swift", method="dpo", init_from=str(data / "earlier" / "model"), train_data=str(data / "pref.jsonl"))
    cfg = seen["config"]
    adapter = str((data / "earlier" / "model").resolve())
    assert seen["argv"][0] == "rlhf" and cfg["rlhf_type"] == "dpo" and cfg["adapters"] == [adapter] and cfg["ref_adapters"] == [adapter]
    assert seen["env"]["NPROC_PER_NODE"] is None and "quant_method" not in cfg
    row = next(read_jsonl(cfg["dataset"][0]))
    assert row["messages"][-1] == {"role": "assistant", "content": "good 0"} and row["rejected_response"] == "bad 0"

    _, seen, _ = run(data, "llm.swift", method="cpt", train_data=str(data / "text.jsonl"), full_finetuning=True)
    assert seen["argv"][0] == "pt" and seen["config"]["tuner_type"] == "full" and "lora_rank" not in seen["config"]
    assert next(read_jsonl(seen["config"]["dataset"][0])) == {"messages": [{"role": "assistant", "content": "Document 0."}]}


def test_failures_are_reported(data):
    with pytest.raises(RuntimeError, match="axolotl exited with code 3"):
        run(data, "llm.axolotl", config={"fail": True})
    with pytest.raises(RuntimeError, match="not part of this environment"):
        run(data, "llm.swift", executable=["definitely-not-installed-swift"])
    with pytest.raises(PipelineError, match="method"):
        run(data, "llm.axolotl", method="grpo")


def test_output_discovery(tmp_path):
    assert split_init("org/base") == ("org/base", None)
    out = tmp_path / "out"
    (out / "checkpoint-5").mkdir(parents=True)
    (out / "checkpoint-40").mkdir()
    (out / "checkpoint-final").mkdir()
    assert last_checkpoint(out) == out / "checkpoint-40" and trained_model(out) == out / "checkpoint-40"
    (out / "config.json").write_text("{}")
    assert trained_model(out) == out  # a model in the output directory itself comes first
    with pytest.raises(RuntimeError, match="neither a model nor a checkpoint"):
        trained_model(tmp_path)
