"""Every Unsloth and TRL call of the llm track, behind one small class.

The stages talk to a backend object: `UnslothBackend` on a training machine, a fake in tests.
The functions that turn stage parameters into library keyword arguments are plain functions
that import nothing heavy, so they are tested without Unsloth installed.

Written against unsloth 2026.11.4 and unsloth_zoo 2026.11.4. That release pins
transformers <= 5.17, trl <= 1.15 and datasets < 5, while this repository's other tracks are
developed on newer ones, so `fit_config_kwargs` renames the few training arguments that differ
between those versions instead of assuming one of them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import Field

from stagetrain.core.training import LoraSpec, TrainArgs, final_metrics

# The projections Unsloth has fast paths for; other names work but train slower.
UNSLOTH_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
# Trained as full matrices through `modules_to_save` (LoRA on embeddings does not train them).
EMBEDDING_MODULES = ["embed_tokens", "lm_head"]

MERGE_METHODS = ("merged_16bit", "merged_4bit")


class LlmTrainArgs(TrainArgs):
    """`TrainArgs` with the defaults of Unsloth's LoRA recipes."""

    learning_rate: float = 2e-4
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    lr_scheduler_type: str = "linear"
    optim: str | None = "adamw_8bit"
    logging_steps: int = 10


class CptTrainArgs(LlmTrainArgs):
    learning_rate: float = 5e-5


class DpoTrainArgs(LlmTrainArgs):
    learning_rate: float = 5e-6


class LlmLora(LoraSpec):
    """`LoraSpec` with Unsloth's defaults, so a config that sets only `r` keeps the fast modules."""

    r: int = 16
    alpha: int = 16
    target_modules: list[str] | str = Field(default_factory=lambda: list(UNSLOTH_TARGET_MODULES))


class CptLora(LlmLora):
    """The wide, rank-stabilised adapter of Unsloth's continued-pretraining recipe."""

    r: int = 128
    alpha: int = 32
    use_rslora: bool = True


@dataclass(frozen=True)
class LoadSpec:
    model_name: str  # a Hub id, a full checkpoint directory, or a LoRA adapter directory
    max_seq_length: int = 2048
    load_in: str = "4bit"  # 4bit (QLoRA), 8bit, or 16bit
    full_finetuning: bool = False
    trust_remote_code: bool = False


@dataclass(frozen=True)
class ResponsesOnly:
    """Mask everything but the assistant turns out of the loss. With both parts `None`, Unsloth
    reads the markers off the tokenizer's chat template."""

    instruction_part: str | None = None
    response_part: str | None = None
    last_response_only: bool = False


@dataclass
class TrainOutput:
    metrics: dict[str, float] = field(default_factory=dict)
    log_history: list[dict[str, Any]] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# parameters -> library keyword arguments (no heavy imports)
# --------------------------------------------------------------------------- #


def load_kwargs(spec: LoadSpec) -> dict[str, Any]:
    """Keyword arguments of `FastLanguageModel.from_pretrained`."""
    if spec.load_in not in ("4bit", "8bit", "16bit"):
        raise ValueError(f"load_in must be 4bit, 8bit or 16bit, got {spec.load_in!r}")
    if spec.full_finetuning and spec.load_in != "16bit":
        raise ValueError("full_finetuning trains every weight, so it needs load_in: 16bit")
    return {
        "model_name": spec.model_name,
        "max_seq_length": spec.max_seq_length,
        "dtype": None,  # Unsloth picks bf16 or fp16 for the device
        "load_in_4bit": spec.load_in == "4bit",
        "load_in_8bit": spec.load_in == "8bit",
        # With full fine-tuning the precision flags stay off: it is its own mode.
        "load_in_16bit": spec.load_in == "16bit" and not spec.full_finetuning,
        "full_finetuning": spec.full_finetuning,
        "trust_remote_code": spec.trust_remote_code,
    }


def peft_kwargs(lora: LoraSpec, seed: int, train_embeddings: bool = False) -> dict[str, Any]:
    """Keyword arguments of `FastLanguageModel.get_peft_model`. With `train_embeddings`, the input
    embeddings and the output head are added to `modules_to_save`."""
    modules_to_save = list(lora.modules_to_save or [])
    if train_embeddings:
        modules_to_save += [m for m in EMBEDDING_MODULES if m not in modules_to_save]
    return {
        "r": lora.r,
        "target_modules": lora.target_modules,
        "lora_alpha": lora.alpha,
        "lora_dropout": lora.dropout,
        "bias": lora.bias,
        "use_rslora": lora.use_rslora,
        "modules_to_save": modules_to_save or None,
        "use_gradient_checkpointing": "unsloth",
        "random_state": seed,
        "loftq_config": None,
    }


def _trainer_kwargs(
    train: TrainArgs, output_dir: str | Path, seed: int, has_eval: bool, run_name: str | None, bf16: bool | None
) -> dict[str, Any]:
    kwargs = train.to_kwargs(str(output_dir), seed=seed, has_eval=has_eval, run_name=run_name)
    if train.precision == "auto" and bf16 is not None:
        # Unsloth's rule: bf16 where the device has it, fp16 everywhere else.
        kwargs["bf16"], kwargs["fp16"] = bf16, not bf16
    return kwargs


def sft_config_kwargs(
    train: TrainArgs,
    *,
    output_dir: str | Path,
    seed: int,
    has_eval: bool,
    max_seq_length: int,
    run_name: str | None = None,
    packing: bool = False,
    embedding_learning_rate: float | None = None,
    bf16: bool | None = None,
) -> dict[str, Any]:
    """Keyword arguments of `UnslothTrainingArguments`, which on a GPU install is TRL's `SFTConfig`
    plus `embedding_learning_rate`. The dataset is always one rendered `text` column."""
    kwargs = _trainer_kwargs(train, output_dir, seed, has_eval, run_name, bf16)
    kwargs.setdefault("dataset_text_field", "text")
    kwargs.setdefault("max_length", max_seq_length)
    kwargs.setdefault("packing", packing)
    if embedding_learning_rate is not None:
        kwargs["embedding_learning_rate"] = embedding_learning_rate
    return kwargs


def dpo_config_kwargs(
    train: TrainArgs,
    *,
    output_dir: str | Path,
    seed: int,
    has_eval: bool,
    max_seq_length: int,
    beta: float = 0.1,
    loss_type: str | list[str] | None = None,
    run_name: str | None = None,
    bf16: bool | None = None,
) -> dict[str, Any]:
    """Keyword arguments of TRL's `DPOConfig`. `loss_type` is left to TRL's default (sigmoid)
    unless set: its type changed from a string to a list between TRL releases."""
    kwargs = _trainer_kwargs(train, output_dir, seed, has_eval, run_name, bf16)
    kwargs.setdefault("max_length", max_seq_length)
    kwargs.setdefault("beta", beta)
    if loss_type is not None:
        kwargs.setdefault("loss_type", loss_type)
    return kwargs


def fit_config_kwargs(kwargs: dict[str, Any], config_cls: type) -> dict[str, Any]:
    """`kwargs` under the names the installed `config_cls` (a `TrainingArguments` dataclass)
    declares. `TrainArgs.to_kwargs` speaks transformers 5.2x; this maps back what older releases
    spell differently and leaves everything else alone, so a real typo still raises in the library.
    """
    import dataclasses

    names = {f.name for f in dataclasses.fields(config_cls)}
    out = dict(kwargs)
    if "train_sampling_strategy" not in names and "train_sampling_strategy" in out:
        if out.pop("train_sampling_strategy") == "group_by_length" and "group_by_length" in names:
            out["group_by_length"] = True
    warmup = out.get("warmup_steps")
    if isinstance(warmup, float) and 0 < warmup < 1 and "warmup_ratio" in names:
        out["warmup_ratio"] = out.pop("warmup_steps")  # only newer releases read a ratio from warmup_steps
    if "eval_strategy" not in names and "evaluation_strategy" in names and "eval_strategy" in out:
        out["evaluation_strategy"] = out.pop("eval_strategy")
    if "max_length" not in names and "max_seq_length" in names and "max_length" in out:
        out["max_seq_length"] = out.pop("max_length")
    for old, new in (("gradient_checkpointing", "activation_checkpointing"), ("activation_checkpointing", "gradient_checkpointing")):
        if old in out and old not in names and new in names:
            out[new] = out.pop(old)
    return out


def is_adapter_dir(path: str | Path) -> bool:
    """Whether `path` is a directory holding a saved LoRA adapter."""
    return (Path(str(path)).expanduser() / "adapter_config.json").is_file()


def adapter_base(path: str | Path) -> str | None:
    """The base model a saved adapter was trained on."""
    try:
        config = json.loads((Path(str(path)).expanduser() / "adapter_config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return config.get("base_model_name_or_path")


def adapter_trains_embeddings(path: str | Path) -> bool:
    """Whether a saved adapter carries trained copies of the embeddings or the output head."""
    try:
        config = json.loads((Path(str(path)).expanduser() / "adapter_config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    saved = config.get("modules_to_save") or []
    return any(str(m).split(".")[-1] in EMBEDDING_MODULES for m in saved)


def has_checkpoint(output_dir: str | Path) -> bool:
    out = Path(output_dir)
    return out.is_dir() and any(p.is_dir() for p in out.glob("checkpoint-*"))


# --------------------------------------------------------------------------- #
# the real backend
# --------------------------------------------------------------------------- #


class UnslothBackend:
    """Unsloth for loading, LoRA, SFT and export; TRL for DPO. `import unsloth` comes first in
    every method: it patches transformers and trl, and it must be imported before them."""

    name = "unsloth"

    # ------------------------------------------------------------------ model

    def load(self, spec: LoadSpec):
        """`(model, tokenizer)`. A LoRA adapter directory loads its base model with the adapter
        on top, trainable, so training can simply continue."""
        import unsloth  # noqa: F401
        from unsloth import FastLanguageModel

        self.release()  # whatever an earlier stage of this process left behind
        return FastLanguageModel.from_pretrained(**load_kwargs(spec))

    def has_adapter(self, model) -> bool:
        return hasattr(model, "peft_config")

    def attach_lora(self, model, kwargs: dict[str, Any]):
        import unsloth  # noqa: F401
        from unsloth import FastLanguageModel

        return FastLanguageModel.get_peft_model(model, **kwargs)

    def set_chat_template(self, tokenizer, name: str):
        """Give the tokenizer one of Unsloth's named chat templates (`chatml`, `llama-3.1`, ...)."""
        import unsloth  # noqa: F401
        from unsloth.chat_templates import get_chat_template

        return get_chat_template(tokenizer, chat_template=name)

    def bf16_supported(self) -> bool | None:
        import unsloth  # noqa: F401
        from unsloth import is_bfloat16_supported

        return bool(is_bfloat16_supported())

    def release(self) -> None:
        """Give GPU memory back. The caller drops its own references to the model first; a model
        that is still referenced anywhere cannot be freed."""
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --------------------------------------------------------------- training

    def _text_trainer(self, model, tokenizer, train_texts, eval_texts, config_kwargs, responses_only):
        import unsloth  # noqa: F401
        from datasets import Dataset
        from unsloth import UnslothTrainer, UnslothTrainingArguments

        trainer = UnslothTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=Dataset.from_dict({"text": list(train_texts)}),
            eval_dataset=Dataset.from_dict({"text": list(eval_texts)}) if eval_texts else None,
            args=UnslothTrainingArguments(**fit_config_kwargs(config_kwargs, UnslothTrainingArguments)),
        )
        if responses_only is not None:
            from unsloth.chat_templates import train_on_responses_only

            trainer = train_on_responses_only(
                trainer,
                instruction_part=responses_only.instruction_part,
                response_part=responses_only.response_part,
                last_response_only=responses_only.last_response_only,
            )
        return trainer

    def train_text(
        self,
        model,
        tokenizer,
        train_texts: list[str],
        eval_texts: list[str],
        config_kwargs: dict[str, Any],
        *,
        responses_only: ResponsesOnly | None = None,
        resume: bool = True,
    ) -> TrainOutput:
        """Causal-LM training on rendered text: continued pretraining (`responses_only=None`) and
        chat fine-tuning (assistant turns only)."""
        trainer = self._text_trainer(model, tokenizer, train_texts, eval_texts, config_kwargs, responses_only)
        return self._fit(trainer, config_kwargs["output_dir"], bool(eval_texts), resume)

    def train_dpo(
        self,
        model,
        tokenizer,
        train_rows: list[dict[str, Any]],
        eval_rows: list[dict[str, Any]],
        config_kwargs: dict[str, Any],
        *,
        resume: bool = True,
    ) -> TrainOutput:
        """Direct preference optimisation on prompt / chosen / rejected rows. No reference model is
        passed: with a LoRA adapter TRL recovers the starting policy from the adapter itself (a
        frozen copy when the adapter was already trained, the bare base model when it is new)."""
        import unsloth  # noqa: F401
        from unsloth import PatchDPOTrainer

        PatchDPOTrainer()  # a no-op in current Unsloth; older releases needed it before the import below
        from datasets import Dataset
        from trl import DPOConfig, DPOTrainer

        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=DPOConfig(**fit_config_kwargs(config_kwargs, DPOConfig)),
            train_dataset=Dataset.from_list(list(train_rows)),
            eval_dataset=Dataset.from_list(list(eval_rows)) if eval_rows else None,
            processing_class=tokenizer,
        )
        return self._fit(trainer, config_kwargs["output_dir"], bool(eval_rows), resume)

    @staticmethod
    def _fit(trainer, output_dir: str, has_eval: bool, resume: bool) -> TrainOutput:
        result = trainer.train(resume_from_checkpoint=True if resume and has_checkpoint(output_dir) else None)
        metrics = dict(result.metrics)
        if has_eval:
            metrics.update(trainer.evaluate())
        return TrainOutput(final_metrics(metrics), list(trainer.state.log_history))

    # ------------------------------------------------------------- evaluation

    def evaluate_loss(
        self, model, tokenizer, texts: list[str], config_kwargs: dict[str, Any], *,
        responses_only: ResponsesOnly | None = None,
    ) -> dict[str, float]:
        """Held-out loss of `texts`, with the same masking as training. TRL's trainer needs a train
        set to be constructed, so it is handed the eval rows; nothing is trained."""
        trainer = self._text_trainer(model, tokenizer, texts, texts, config_kwargs, responses_only)
        return final_metrics(trainer.evaluate())

    def generate(
        self, model, tokenizer, conversations: list[list[dict[str, str]]], *, max_new_tokens: int = 256,
        temperature: float = 0.0, top_p: float = 1.0,
    ) -> list[str]:
        """The model's reply to each conversation, one at a time, greedy when `temperature` is 0."""
        import torch
        import unsloth  # noqa: F401
        from unsloth import FastLanguageModel

        FastLanguageModel.for_inference(model)
        sampling = {"do_sample": True, "temperature": temperature, "top_p": top_p} if temperature > 0 else {"do_sample": False}
        device = next(model.parameters()).device
        out = []
        for messages in conversations:
            inputs = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True
            ).to(device)
            with torch.no_grad():
                ids = model.generate(**inputs, max_new_tokens=max_new_tokens, use_cache=True, **sampling)
            out.append(tokenizer.decode(ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip())
        return out

    # ----------------------------------------------------------------- saving

    def save(self, model, tokenizer, out_dir: str | Path) -> None:
        """What was trained: the LoRA adapter, or the whole model after full fine-tuning."""
        model.save_pretrained(str(out_dir))
        tokenizer.save_pretrained(str(out_dir))

    def save_merged(self, model, tokenizer, out_dir: str | Path, method: str = "merged_16bit") -> None:
        """The adapter folded into the base weights, as a standalone checkpoint."""
        if method not in MERGE_METHODS:
            raise ValueError(f"merge method must be one of {MERGE_METHODS}, got {method!r}")
        model.save_pretrained_merged(str(out_dir), tokenizer, save_method=method)

    def save_gguf(self, model, tokenizer, out_dir: str | Path, quantization: list[str]) -> tuple[str, list[str]]:
        """`(directory, files)` of the GGUF export. Unsloth writes the 16-bit merge it converts
        from into `out_dir` and the GGUF files into the sibling `<out_dir>_gguf`."""
        result = model.save_pretrained_gguf(str(out_dir), tokenizer, quantization_method=list(quantization))
        directory = f"{out_dir}_gguf"
        files: list[str] = []
        if isinstance(result, dict):
            directory = result.get("gguf_directory") or directory
            files = [str(f) for f in result.get("gguf_files") or []]
        if not files and Path(directory).is_dir():
            files = sorted(str(p) for p in Path(directory).glob("*.gguf"))
        return str(directory), files

    def push(
        self, model, tokenizer, repo_id: str, method: str, *, private: bool = True, quantization: list[str] | None = None,
    ) -> None:
        if method == "gguf":
            model.push_to_hub_gguf(repo_id, tokenizer, quantization_method=list(quantization or []), private=private)
        else:
            model.push_to_hub_merged(repo_id, tokenizer, save_method=method, private=private)


_BACKEND: Any = None


def get_backend():
    global _BACKEND
    if _BACKEND is None:
        _BACKEND = UnslothBackend()
    return _BACKEND


def set_backend(backend) -> None:
    """Replace the backend (tests pass a fake; `None` restores Unsloth)."""
    global _BACKEND
    _BACKEND = backend
