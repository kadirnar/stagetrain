"""`llm.axolotl` and `llm.swift`: a training stage run by another library's command line.

Axolotl and ms-swift each bring multi-GPU training (DDP, FSDP2, DeepSpeed) and methods of their
own behind a config file and a CLI. A stage here writes that config from the same parameters the
other training stages take, converts the data to the library's row format, runs the command, and
reads the result back: the adapter or model it wrote, and its training log.

Each library pins its own versions of transformers and torch, so it lives in an environment of
its own. `executable` names its command there, e.g. `[/opt/axolotl/bin/axolotl]`.

The configs are written from the libraries' documentation (Axolotl 0.21, ms-swift 4.6). Neither
library was run here: anything it rejects can be corrected through `config`, whose keys are
written as they are and win over the generated ones.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field

from stagetrain.core import Stage, StageContext, StageParams, StageResult, read_jsonl, register, write_jsonl
from stagetrain.core.history import save_history
from stagetrain.core.training import final_metrics

from .backend import LlmLora, SftTrainArgs, adapter_base, is_adapter_dir


class ExternalParams(StageParams):
    method: Literal["sft", "cpt", "dpo"] = "sft"
    init_from: str  # a Hub id, a full model directory, or the LoRA adapter of an earlier stage
    train_data: str  # the same JSONL files the other stages read: chat, raw text, or preference pairs
    eval_data: str | None = None
    max_seq_length: int = 2048
    load_in: Literal["4bit", "8bit", "16bit"] = "16bit"
    full_finetuning: bool = False
    lora: LlmLora = Field(default_factory=LlmLora)
    train: SftTrainArgs = Field(default_factory=SftTrainArgs)
    num_processes: int = Field(default=1, ge=1)  # one per GPU
    gpu_ids: str | None = None  # CUDA_VISIBLE_DEVICES for the run, e.g. "0,1,2,3"
    config: dict[str, Any] = Field(default_factory=dict)  # the library's own keys; written as they are, and they win
    executable: list[str] | None = None  # the library's command; give its path when it lives in another environment
    env: dict[str, str] = Field(default_factory=dict)


def split_init(init_from: str) -> tuple[str, str | None]:
    """`(base model, adapter directory or None)`."""
    if is_adapter_dir(init_from):
        base = adapter_base(init_from)
        if not base:
            raise ValueError(f"{init_from}: adapter_config.json names no base model")
        return base, str(Path(init_from).resolve())
    return init_from, None


def last_checkpoint(output_dir: Path) -> Path | None:
    found = []
    for path in output_dir.glob("checkpoint-*"):
        step = path.name.rsplit("-", 1)[-1]
        if path.is_dir() and step.isdigit():
            found.append((int(step), path))
    return max(found)[1] if found else None


def trained_model(output_dir: Path) -> Path:
    """Where the run left its result: the output directory itself when it holds the adapter or
    the model, otherwise its `last` link or its latest checkpoint."""
    for name in ("adapter_config.json", "config.json"):
        if (output_dir / name).is_file():
            return output_dir
    last = output_dir / "last"
    if last.exists():
        return last.resolve()
    checkpoint = last_checkpoint(output_dir)
    if checkpoint is None:
        raise RuntimeError(f"the run left neither a model nor a checkpoint in {output_dir}")
    return checkpoint


def log_history(output_dir: Path) -> list[dict[str, Any]]:
    """The `Trainer` log of the latest checkpoint, when there is one."""
    for root in (last_checkpoint(output_dir), output_dir):
        state = root / "trainer_state.json" if root else None
        if state and state.is_file():
            return list(json.loads(state.read_text(encoding="utf-8")).get("log_history") or [])
    return []


def preference_rows(path: str) -> list[dict[str, Any]]:
    return [r for r in read_jsonl(path) if r.get("chosen") and r.get("rejected")]


class ExternalStage(Stage):
    """Shared steps of a stage that hands training to another library's CLI."""

    Params = ExternalParams
    tool = ""

    def convert(self, p: ExternalParams, source: str, target: Path) -> str:
        """The data file in the library's row format; by default the file as it is."""
        return str(Path(source).resolve())

    def build_config(self, p: ExternalParams, data: str, eval_data: str | None, out: Path, seed: int) -> dict[str, Any]:
        raise NotImplementedError

    def command(self, p: ExternalParams, config: Path) -> list[str]:
        raise NotImplementedError

    def environment(self, p: ExternalParams) -> dict[str, str]:
        env = {**os.environ, "HF_HUB_DISABLE_TELEMETRY": "1", **p.env}
        if p.gpu_ids:
            env["CUDA_VISIBLE_DEVICES"] = p.gpu_ids
        return env

    def run(self, ctx: StageContext) -> StageResult:
        p: ExternalParams = self.params
        out = ctx.out_dir / "output"
        data = self.convert(p, p.train_data, ctx.path("data", "train.jsonl"))
        eval_data = self.convert(p, p.eval_data, ctx.path("data", "eval.jsonl")) if p.eval_data else None
        config = {**self.build_config(p, data, eval_data, out, self.seed(ctx)), **p.config}
        config_path = ctx.path("config.yaml")
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        command = self.command(p, config_path)
        ctx.log.info("stage %s: %s on %d process(es): %s", ctx.stage, self.tool, p.num_processes, " ".join(command))
        try:
            done = subprocess.run(command, env=self.environment(p), cwd=ctx.out_dir, check=False)
        except FileNotFoundError as e:
            raise RuntimeError(
                f"{command[0]!r} was not found. {self.tool} is not part of this environment: install it in one of "
                f"its own and set `executable` to its command there"
            ) from e
        if done.returncode != 0:
            raise RuntimeError(f"stage {ctx.stage!r}: {self.tool} exited with code {done.returncode}; see its output above")
        model = trained_model(out)
        history = log_history(out)
        save_history(ctx.out_dir, history)
        metrics: dict[str, float] = {}
        losses = [e["loss"] for e in history if isinstance(e.get("loss"), (int, float))]
        evals = [e["eval_loss"] for e in history if isinstance(e.get("eval_loss"), (int, float))]
        if losses:
            metrics["train/loss"] = float(losses[-1])
        if evals:
            metrics["eval/loss"] = float(evals[-1])
        summary = next((e for e in reversed(history) if "train_runtime" in e), {})
        metrics.update({k: v for k, v in final_metrics(summary).items() if k.startswith("train/")})
        artifacts: dict[str, Any] = {"model": model, "config": config_path}
        if last_checkpoint(out) is not None:
            artifacts["checkpoints"] = out
        return StageResult(artifacts=artifacts, metrics=metrics, notes={"tool": self.tool, "command": command})


# --------------------------------------------------------------------------- #
# Axolotl
# --------------------------------------------------------------------------- #


@register("llm.axolotl")
class Axolotl(ExternalStage):
    """Train with Axolotl's CLI (SFT, continued pretraining, DPO), on one GPU or several."""

    tool = "axolotl"

    def convert(self, p: ExternalParams, source: str, target: Path) -> str:
        if p.method != "dpo":
            return str(Path(source).resolve())
        # Axolotl's `chat_template.default` preference rows: the conversation, then one chosen and
        # one rejected assistant message.
        rows = [{"messages": r["prompt"], "chosen": r["chosen"][-1], "rejected": r["rejected"][-1]} for r in preference_rows(source)]
        write_jsonl(target, rows)
        return str(target.resolve())

    def dataset(self, p: ExternalParams, path: str) -> dict[str, Any]:
        mapping = {"role": "role", "content": "content"}
        if p.method == "cpt":
            return {"path": path, "ds_type": "json", "type": "completion"}
        if p.method == "dpo":
            return {
                "path": path, "ds_type": "json", "split": "train", "type": "chat_template.default",
                "field_messages": "messages", "field_chosen": "chosen", "field_rejected": "rejected",
                "message_property_mappings": mapping,
                "roles": {"user": ["user"], "assistant": ["assistant"], "system": ["system"]},
            }
        return {"path": path, "ds_type": "json", "type": "chat_template", "field_messages": "messages",
                "message_property_mappings": mapping}

    def build_config(self, p: ExternalParams, data: str, eval_data: str | None, out: Path, seed: int) -> dict[str, Any]:
        base, adapter = split_init(p.init_from)
        t = p.train
        config: dict[str, Any] = {
            "base_model": base,
            "datasets": [self.dataset(p, data)],
            "output_dir": str(out),
            "dataset_prepared_path": str(out.parent / "prepared"),
            "sequence_len": p.max_seq_length,
            "micro_batch_size": t.per_device_train_batch_size,
            "gradient_accumulation_steps": t.gradient_accumulation_steps,
            "num_epochs": t.num_train_epochs,
            "learning_rate": t.learning_rate,
            "optimizer": t.optim or "adamw_torch",
            "lr_scheduler": t.lr_scheduler_type,
            "weight_decay": t.weight_decay,
            "logging_steps": t.logging_steps,
            "saves_per_epoch": 1,
            "seed": seed,
            "bf16": "auto",
            "val_set_size": 0,
        }
        config["warmup_ratio" if t.warmup < 1 else "warmup_steps"] = t.warmup if t.warmup < 1 else int(t.warmup)
        if t.max_steps > 0:
            config["max_steps"] = t.max_steps
        if p.method == "sft":
            config["chat_template"] = "tokenizer_default"  # the tokenizer's own template, as the other stages use
        if p.method == "dpo":
            config["rl"] = "dpo"
        if not p.full_finetuning:
            config.update({
                "adapter": "qlora" if p.load_in == "4bit" else "lora",
                "lora_r": p.lora.r, "lora_alpha": p.lora.alpha, "lora_dropout": p.lora.dropout,
                "lora_target_modules": p.lora.target_modules,
            })
            if adapter:
                config["lora_model_dir"] = adapter  # go on from the earlier stage's adapter
        if p.load_in != "16bit" and not p.full_finetuning:
            config[f"load_in_{p.load_in}"] = True
        if eval_data:
            config["test_datasets"] = [{**self.dataset(p, eval_data), "split": "train"}]
            config["evals_per_epoch"] = 1
        return config

    def command(self, p: ExternalParams, config: Path) -> list[str]:
        return [*(p.executable or ["axolotl"]), "train", str(config), "--num-processes", str(p.num_processes)]

    def environment(self, p: ExternalParams) -> dict[str, str]:
        return {"AXOLOTL_DO_NOT_TRACK": "1", **super().environment(p)}  # its telemetry is on unless told otherwise


# --------------------------------------------------------------------------- #
# ms-swift
# --------------------------------------------------------------------------- #

SWIFT_COMMANDS = {"sft": "sft", "cpt": "pt", "dpo": "rlhf"}


@register("llm.swift")
class Swift(ExternalStage):
    """Train with ms-swift's CLI (SFT, continued pretraining, DPO), on one GPU or several."""

    tool = "ms-swift"

    def convert(self, p: ExternalParams, source: str, target: Path) -> str:
        if p.method == "sft":
            return str(Path(source).resolve())
        if p.method == "cpt":
            # ms-swift reads raw text as a conversation that holds one assistant message.
            rows = [{"messages": [{"role": "assistant", "content": r["text"]}]} for r in read_jsonl(source) if r.get("text")]
        else:
            # A preference pair is the conversation with the chosen answer, plus `rejected_response`.
            rows = [
                {"messages": [*r["prompt"], r["chosen"][-1]], "rejected_response": r["rejected"][-1]["content"]}
                for r in preference_rows(source)
            ]
        write_jsonl(target, rows)
        return str(target.resolve())

    def build_config(self, p: ExternalParams, data: str, eval_data: str | None, out: Path, seed: int) -> dict[str, Any]:
        base, adapter = split_init(p.init_from)
        t = p.train
        config: dict[str, Any] = {
            "model": base,
            "use_hf": True,  # ms-swift reads model ids from ModelScope unless told to use the Hugging Face Hub
            "dataset": [data],
            "tuner_type": "full" if p.full_finetuning else "lora",
            "torch_dtype": "bfloat16",
            "max_length": p.max_seq_length,
            "num_train_epochs": t.num_train_epochs,
            "per_device_train_batch_size": t.per_device_train_batch_size,
            "per_device_eval_batch_size": t.per_device_eval_batch_size or t.per_device_train_batch_size,
            "gradient_accumulation_steps": t.gradient_accumulation_steps,
            "learning_rate": t.learning_rate,
            "lr_scheduler_type": t.lr_scheduler_type,
            "weight_decay": t.weight_decay,
            "logging_steps": t.logging_steps,
            "seed": seed,
            "output_dir": str(out),
            "add_version": False,  # otherwise it appends a version and a timestamp to output_dir
            "create_checkpoint_symlink": True,  # output_dir/last points at the latest checkpoint
        }
        config["warmup_ratio" if t.warmup < 1 else "warmup_steps"] = t.warmup if t.warmup < 1 else int(t.warmup)
        if t.max_steps > 0:
            config["max_steps"] = t.max_steps
        if t.save_steps:
            config["save_steps"] = t.save_steps
            config["eval_steps"] = t.eval_steps or t.save_steps
        if t.save_total_limit is not None:
            config["save_total_limit"] = t.save_total_limit
        if not p.full_finetuning:
            config.update({
                "lora_rank": p.lora.r, "lora_alpha": p.lora.alpha, "lora_dropout": p.lora.dropout,
                "target_modules": p.lora.target_modules if isinstance(p.lora.target_modules, list) else [p.lora.target_modules],
            })
            if p.load_in != "16bit":
                config.update({"quant_method": "bnb", "quant_bits": 4 if p.load_in == "4bit" else 8})
            if adapter:
                config["adapters"] = [adapter]
                if p.method == "dpo":
                    config["ref_adapters"] = [adapter]  # the reference policy is the model DPO starts from
        if p.method == "dpo":
            config["rlhf_type"] = "dpo"
        if eval_data:
            config["val_dataset"] = [eval_data]
        return config

    def command(self, p: ExternalParams, config: Path) -> list[str]:
        return [*(p.executable or ["swift"]), SWIFT_COMMANDS[p.method], str(config)]

    def environment(self, p: ExternalParams) -> dict[str, str]:
        env = super().environment(p)
        if p.num_processes > 1:
            # Without this, several visible GPUs mean one model split across them, not one process per GPU.
            env["NPROC_PER_NODE"] = str(p.num_processes)
        return env
