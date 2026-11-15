"""Stage `decision.train`: fine-tune a causal LM to answer at the answer slot.

The same stage serves every step of the curriculum (format alignment on short inputs, the full
mixture on long inputs, one run per seed and data mixture for the soup): only its parameters
change. It always writes a full merged checkpoint, so the next stage starts from plain weights.
"""

from __future__ import annotations

import json
import random
import shutil
from typing import Any

from pydantic import Field

from stagetrain.core import (
    LoraSpec,
    Stage,
    StageContext,
    StageParams,
    StageResult,
    TrainArgs,
    final_metrics,
    read_jsonl,
    register,
    seed_everything,
)

from .metrics import summarize
from .modeling import (
    AugmentConfig,
    DecisionCollator,
    LossConfig,
    decision_loss,
    load_model,
    load_records,
    load_tokenizer,
    read_meta,
    score_records,
    slot_logits,
    write_meta,
)
from .prompt import PromptStyle


class TrainParams(StageParams):
    init_from: str  # a Hub id, or an earlier stage's `model`
    train_data: str
    eval_data: str | None = None
    max_eval_samples: int | None = 1000
    # Sampling rate per source: 0 drops it, 0.5 keeps half, 2 shows every row twice (each time with
    # its own option order). Different mixtures give the soup its diversity.
    mixture: dict[str, float] = Field(default_factory=dict)
    mixture_default: float = 1.0
    max_length: int = 2048
    prompt: PromptStyle | None = None  # None: the style `init_from` was trained in, else the default
    lora: LoraSpec | None = Field(default_factory=LoraSpec)  # None: full fine-tuning
    train: TrainArgs = Field(default_factory=lambda: TrainArgs(learning_rate=1e-4, per_device_train_batch_size=16))
    loss: LossConfig = Field(default_factory=LossConfig)
    augment: AugmentConfig = Field(default_factory=AugmentConfig)
    eval_batch_size: int = 16
    resume: bool = True  # continue from the stage's last checkpoint after an interruption
    keep_checkpoints: bool = False
    trust_remote_code: bool = False


def apply_mixture(rows: list[dict[str, Any]], mixture: dict[str, float], default: float, seed: int) -> list[dict]:
    if not mixture and default == 1.0:
        return rows
    rng = random.Random(seed)
    out = []
    for row in rows:
        rate = mixture.get(row.get("source", "default"), default)
        copies = int(rate) + (rng.random() < rate - int(rate))
        out.extend([row] * copies)
    return out


def as_dataset(rows: list[dict[str, Any]]):
    """Records as a `datasets.Dataset` of JSON strings (states and criteria have no fixed schema)
    with a `length` column for length-grouped batches."""
    from datasets import Dataset

    blobs = [json.dumps(r, ensure_ascii=False) for r in rows]
    return Dataset.from_dict({"record": blobs, "length": [len(b) for b in blobs]})


def trainer_class():
    from transformers import Trainer

    class DecisionTrainer(Trainer):
        def __init__(self, *args, loss_config: LossConfig, evaluator=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.loss_config = loss_config
            self.evaluator = evaluator

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            slot = slot_logits(model, inputs["input_ids"], inputs["attention_mask"])
            loss, scores = decision_loss(slot, inputs, self.loss_config)
            return (loss, {"scores": scores}) if return_outputs else loss

        def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
            """Accuracy, NLL and calibration of the answers, instead of a token-level loss."""
            if self.evaluator is None:
                return {}
            metrics = {f"{metric_key_prefix}_{k}": v for k, v in self.evaluator(self.model).items()}
            self.log(metrics)
            self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
            return metrics

    return DecisionTrainer


@register("decision.train")
class Train(Stage):
    """Fine-tune a causal LM on decision records with the answer-slot loss (LoRA or full)."""

    Params = TrainParams

    def run(self, ctx: StageContext) -> StageResult:
        from transformers import TrainingArguments
        from transformers.trainer_utils import get_last_checkpoint

        p = self.params
        seed = self.seed(ctx)
        seed_everything(seed)

        tokenizer = load_tokenizer(p.init_from, p.trust_remote_code)
        style = p.prompt or PromptStyle.model_validate(read_meta(p.init_from).get("prompt") or {})
        # Under LoRA the frozen base can sit in half precision. Full fine-tuning keeps fp32 weights
        # (mixed precision still applies to the forward pass): small updates vanish in bf16 weights.
        model = load_model(p.init_from, p.train.precision if p.lora is not None else "fp32", p.trust_remote_code)
        if p.lora is not None:
            from peft import LoraConfig, get_peft_model

            model = get_peft_model(model, LoraConfig(
                task_type="CAUSAL_LM", r=p.lora.r, lora_alpha=p.lora.alpha, lora_dropout=p.lora.dropout,
                target_modules=p.lora.target_modules, use_rslora=p.lora.use_rslora,
                modules_to_save=p.lora.modules_to_save, bias=p.lora.bias,
            ))

        rows = apply_mixture(list(read_jsonl(p.train_data)), p.mixture, p.mixture_default, seed)
        if not rows:
            raise ValueError(f"{p.train_data}: no training records left after the mixture")
        eval_records = load_records(p.eval_data, p.max_eval_samples) if p.eval_data else []
        evaluator = None
        if eval_records:
            def evaluator(m):
                return summarize(score_records(m, tokenizer, eval_records, style, p.max_length, p.eval_batch_size))

        checkpoints = ctx.out_dir / "checkpoints"
        kwargs = p.train.to_kwargs(str(checkpoints), seed, has_eval=bool(eval_records), run_name=f"{ctx.pipeline}-{ctx.stage}")
        kwargs.setdefault("remove_unused_columns", False)  # the collator reads the `record` column
        collator = DecisionCollator(tokenizer, style, p.max_length, p.augment, seed)
        trainer = trainer_class()(
            model=model, args=TrainingArguments(**kwargs), data_collator=collator,
            train_dataset=as_dataset(rows),
            eval_dataset=as_dataset([r.to_json() for r in eval_records]) if eval_records else None,
            loss_config=p.loss, evaluator=evaluator,
        )
        # `ctx.resumed`: the directory holds an unfinished attempt of this same run, nothing older.
        last = get_last_checkpoint(str(checkpoints)) if p.resume and ctx.resumed and checkpoints.is_dir() else None
        if last:
            ctx.log.info("stage %s: resuming from %s", ctx.stage, last)
        result = trainer.train(resume_from_checkpoint=last)

        metrics = final_metrics({"train_loss": result.training_loss, "train_steps": trainer.state.global_step})
        metrics["train/records"] = float(len(rows))
        metrics["train/truncated_states"] = float(collator.truncated)
        if evaluator is not None:
            metrics.update({f"eval/{k}": v for k, v in evaluator(trainer.model).items()})

        artifacts = {}
        model = trainer.model
        if p.lora is not None:
            artifacts["adapter"] = ctx.out_dir / "adapter"
            model.save_pretrained(str(artifacts["adapter"]))
            model = model.merge_and_unload()
        artifacts["model"] = ctx.out_dir / "model"
        model.save_pretrained(str(artifacts["model"]))
        tokenizer.save_pretrained(str(artifacts["model"]))
        write_meta(artifacts["model"], style, p.max_length, extra={"init_from": p.init_from})
        if not p.keep_checkpoints:
            shutil.rmtree(checkpoints, ignore_errors=True)
        return StageResult(artifacts=artifacts, metrics=metrics)
