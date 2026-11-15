"""Training settings shared by the tracks that train with Hugging Face `Trainer`s."""

from __future__ import annotations

import random
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class TrainArgs(BaseModel):
    """The subset of `transformers.TrainingArguments` a stage usually sets, under stable names.
    Anything else goes through `extra` unchanged."""

    model_config = ConfigDict(extra="forbid")

    learning_rate: float = 2e-5
    num_train_epochs: float = 1.0
    max_steps: int = -1  # > 0 overrides num_train_epochs
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int | None = None
    gradient_accumulation_steps: int = 1
    warmup: float = 0.05  # below 1: a ratio of the total steps; otherwise a number of steps
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0
    optim: str | None = None
    # Adam's moments and epsilon, the library's defaults (0.9, 0.999, 1e-8) when unset. No
    # fine-tuning study found a value that beats them; they are here to be set, not to be swept.
    adam_beta1: float | None = Field(default=None, ge=0.0, lt=1.0)
    adam_beta2: float | None = Field(default=None, ge=0.0, lt=1.0)
    adam_epsilon: float | None = Field(default=None, gt=0.0)
    lr_scheduler_kwargs: dict[str, Any] = Field(default_factory=dict)  # e.g. {min_lr_rate: 0.1} for cosine_with_min_lr
    precision: Literal["auto", "bf16", "fp16", "fp32"] = "auto"
    gradient_checkpointing: bool = False
    group_by_length: bool = False
    logging_steps: int = 20
    eval_steps: int | None = None  # None: evaluate once per epoch when there is an eval set
    save_steps: int | None = None  # None: save when evaluating (or once per epoch)
    save_total_limit: int | None = 2
    load_best_model_at_end: bool = False
    metric_for_best_model: str | None = None
    greater_is_better: bool | None = None
    dataloader_num_workers: int = 0
    report_to: list[str] = Field(default_factory=list)  # e.g. [wandb], [tensorboard]
    extra: dict[str, Any] = Field(default_factory=dict)

    def to_kwargs(self, output_dir: str, seed: int, has_eval: bool, run_name: str | None = None) -> dict[str, Any]:
        """Keyword arguments for `TrainingArguments` (or a subclass such as
        `SentenceTransformerTrainingArguments`)."""
        precision = resolve_precision(self.precision)
        if has_eval:
            eval_strategy = "steps" if self.eval_steps else "epoch"
        else:
            eval_strategy = "no"
        save_steps = self.save_steps or self.eval_steps
        save_strategy = "steps" if save_steps else "epoch"
        if self.load_best_model_at_end and not has_eval:
            raise ValueError("load_best_model_at_end needs an eval set")
        kwargs: dict[str, Any] = {
            "output_dir": output_dir,
            "seed": seed,
            "learning_rate": self.learning_rate,
            "num_train_epochs": self.num_train_epochs,
            "max_steps": self.max_steps,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "per_device_eval_batch_size": self.per_device_eval_batch_size or self.per_device_train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "warmup_steps": self.warmup if self.warmup < 1 else int(self.warmup),
            "weight_decay": self.weight_decay,
            "lr_scheduler_type": self.lr_scheduler_type,
            "max_grad_norm": self.max_grad_norm,
            "bf16": precision == "bf16",
            "fp16": precision == "fp16",
            "train_sampling_strategy": "group_by_length" if self.group_by_length else "random",
            "logging_steps": self.logging_steps,
            "eval_strategy": eval_strategy,
            "save_strategy": save_strategy,
            "save_total_limit": self.save_total_limit,
            "load_best_model_at_end": self.load_best_model_at_end,
            "dataloader_num_workers": self.dataloader_num_workers,
            "report_to": self.report_to or "none",
        }
        if self.gradient_checkpointing:
            kwargs[_checkpointing_key()] = True
        if eval_strategy == "steps":
            kwargs["eval_steps"] = self.eval_steps
        if save_strategy == "steps":
            kwargs["save_steps"] = save_steps
        if self.lr_scheduler_kwargs:
            kwargs["lr_scheduler_kwargs"] = dict(self.lr_scheduler_kwargs)
        for key in ("optim", "adam_beta1", "adam_beta2", "adam_epsilon", "metric_for_best_model", "greater_is_better"):
            if getattr(self, key) is not None:
                kwargs[key] = getattr(self, key)
        if run_name:
            kwargs["run_name"] = run_name
        kwargs.update(self.extra)
        return kwargs


def _checkpointing_key() -> str:
    """transformers 5 renamed `gradient_checkpointing` to `activation_checkpointing`."""
    try:
        import dataclasses

        from transformers import TrainingArguments
    except ImportError:
        return "gradient_checkpointing"
    names = {f.name for f in dataclasses.fields(TrainingArguments)}
    return "activation_checkpointing" if "activation_checkpointing" in names else "gradient_checkpointing"


class LoraSpec(BaseModel):
    """A LoRA adapter. `target_modules` is a list of module names or `all-linear`."""

    model_config = ConfigDict(extra="forbid")

    r: int = 16
    alpha: int = 32
    dropout: float = 0.0
    target_modules: list[str] | str = "all-linear"
    use_rslora: bool = False
    modules_to_save: list[str] | None = None
    bias: Literal["none", "all", "lora_only"] = "none"


def resolve_precision(precision: str) -> str:
    """`auto` is bf16 on a CUDA device that supports it, and fp32 anywhere else."""
    if precision != "auto":
        return precision
    try:
        import torch
    except ImportError:
        return "fp32"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return "bf16"
    return "fp32"


def torch_dtype(precision: str):
    import torch

    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[resolve_precision(precision)]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def final_metrics(metrics: dict[str, Any], prefix: str = "") -> dict[str, float]:
    """A `Trainer` metrics dict as stage metrics: numbers only, `eval_x` renamed to `eval/x`."""
    out = {}
    for k, v in metrics.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        for head in ("eval_", "train_", "test_"):
            if k.startswith(head):
                k = f"{head[:-1]}/{k[len(head):]}"
                break
        out[f"{prefix}{k}"] = float(v)
    return out
