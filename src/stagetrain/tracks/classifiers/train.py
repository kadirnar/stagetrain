"""`classifiers.mlm_adapt` and `classifiers.finetune`: the two stages that update weights."""

from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from stagetrain.core import (
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

from .labels import LABELS_FILE, LabelSpace, source_counts


def _paths(v: Any) -> Any:
    return [v] if isinstance(v, str) else v


def resume_point(checkpoints: Path, resume: bool) -> str | None:
    """The last checkpoint to continue from, or None to start over with an empty directory.

    `resume` is the stage's `resume` parameter and the pipeline's `ctx.resumed`, which is only
    true when the stage directory is left from an unfinished attempt of this same run. Without
    it, checkpoints found here belong to another configuration and are deleted.
    """
    from transformers.trainer_utils import get_last_checkpoint

    last = get_last_checkpoint(str(checkpoints)) if resume and checkpoints.is_dir() else None
    if last is None:
        shutil.rmtree(checkpoints, ignore_errors=True)
    return last


def _resume_note(last: str | None) -> dict[str, str]:
    return {"resumed_from": Path(last).name} if last else {}


def _finish(trainer, tokenizer, model_dir: Path, checkpoints: Path, keep_checkpoints: bool) -> None:
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    if not keep_checkpoints:
        shutil.rmtree(checkpoints, ignore_errors=True)


# --------------------------------------------------------------------------- #
# masked-LM adaptation
# --------------------------------------------------------------------------- #


class MlmParams(StageParams):
    init_from: str  # a Hub id or a model directory with a masked-LM head
    data: list[str]  # JSONL files with `text`; labels are ignored
    eval_data: str | None = None
    eval_fraction: float = 0.05  # held out of `data` when `eval_data` is not given
    max_length: int = 256
    chunk_long_texts: bool = True  # split texts longer than max_length into windows instead of truncating
    stride: int = 32
    mlm_probability: float = 0.15  # ModernBERT was pretrained with 0.3
    max_samples: int | None = None
    train: TrainArgs = Field(default_factory=lambda: TrainArgs(learning_rate=5e-5, num_train_epochs=3))
    resume: bool = True  # continue from the last checkpoint when the pipeline resumes this same run
    keep_checkpoints: bool = False
    trust_remote_code: bool = False

    @field_validator("data", mode="before")
    @classmethod
    def _data(cls, v: Any) -> Any:
        return _paths(v)


def mlm_features(tokenizer, texts: list[str], max_length: int, chunk: bool, stride: int) -> list[dict[str, list[int]]]:
    """Token ids of `texts`; with `chunk`, a long text gives several overlapping windows."""
    if not texts:
        return []
    kwargs: dict[str, Any] = {"truncation": True, "max_length": max_length, "return_special_tokens_mask": True}
    if chunk:
        kwargs.update(return_overflowing_tokens=True, stride=min(stride, max(0, max_length // 4)))
    enc = tokenizer(texts, **kwargs)
    keys = [k for k in ("input_ids", "attention_mask", "token_type_ids", "special_tokens_mask") if k in enc]
    return [{k: enc[k][i] for k in keys} for i in range(len(enc["input_ids"]))]


@register("classifiers.mlm_adapt")
class MlmAdapt(Stage):
    """Adapt the encoder to the domain with masked-language modelling on unlabeled text (DAPT/TAPT)."""

    Params = MlmParams

    def run(self, ctx: StageContext) -> StageResult:
        import random

        from transformers import (
            AutoModelForMaskedLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )

        from .modeling import RowDataset

        p: MlmParams = self.params
        seed = self.seed(ctx)
        seed_everything(seed)
        tokenizer = AutoTokenizer.from_pretrained(p.init_from, trust_remote_code=p.trust_remote_code)
        if tokenizer.mask_token_id is None:
            raise ValueError(f"{p.init_from}: the tokenizer has no mask token, so it cannot do masked-LM")
        texts = sorted({str(r["text"]) for path in p.data for r in read_jsonl(path) if r.get("text")})
        random.Random(seed).shuffle(texts)
        if p.max_samples:
            texts = texts[: p.max_samples]
        if p.eval_data:
            eval_texts = [str(r["text"]) for r in read_jsonl(p.eval_data) if r.get("text")]
        else:
            n_eval = min(len(texts) - 1, max(1, round(len(texts) * p.eval_fraction))) if p.eval_fraction > 0 else 0
            eval_texts, texts = texts[:n_eval], texts[n_eval:]
        if not texts:
            raise ValueError("mlm_adapt has no text to train on")

        train_set = RowDataset(mlm_features(tokenizer, texts, p.max_length, p.chunk_long_texts, p.stride))
        eval_set = RowDataset(mlm_features(tokenizer, eval_texts, p.max_length, p.chunk_long_texts, p.stride))
        model = AutoModelForMaskedLM.from_pretrained(p.init_from, trust_remote_code=p.trust_remote_code)
        checkpoints = ctx.out_dir / "checkpoints"
        has_eval = len(eval_set) > 0
        args = TrainingArguments(**p.train.to_kwargs(str(checkpoints), seed, has_eval, f"{ctx.pipeline}-{ctx.stage}"))
        trainer = Trainer(
            model=model, args=args, train_dataset=train_set, eval_dataset=eval_set if has_eval else None,
            processing_class=tokenizer,
            # A seeded collator: the masks, and so the losses, are the same for the same stage seed.
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm_probability=p.mlm_probability, seed=seed),
        )
        metrics: dict[str, float] = {"n_train": float(len(train_set)), "n_eval": float(len(eval_set))}
        if has_eval:
            metrics["mlm/loss_before"] = float(trainer.evaluate()["eval_loss"])
        last = resume_point(checkpoints, p.resume and ctx.resumed)
        out = trainer.train(resume_from_checkpoint=last)
        metrics["train/loss"] = float(out.training_loss)
        if has_eval:
            loss = float(trainer.evaluate()["eval_loss"])
            metrics.update({
                "eval/loss": loss,
                "eval/perplexity": math.exp(min(loss, 50.0)),
                "mlm/loss_drop": metrics["mlm/loss_before"] - loss,
            })
        model_dir = ctx.out_dir / "model"
        _finish(trainer, tokenizer, model_dir, checkpoints, p.keep_checkpoints)
        return StageResult(artifacts={"model": model_dir}, metrics=metrics, notes=_resume_note(last))


# --------------------------------------------------------------------------- #
# supervised fine-tuning
# --------------------------------------------------------------------------- #


class ExtraData(BaseModel):
    """More training rows, e.g. the output of `classifiers.pseudo_label`."""

    model_config = ConfigDict(extra="forbid")

    path: str
    weight: float = 1.0  # multiplies the loss of every row of this file
    max_samples: int | None = None


class FinetuneParams(StageParams):
    init_from: str  # a Hub id, or the `model` artifact of mlm_adapt or of an earlier finetune
    labels: str  # labels.json from prepare_data
    train_data: str
    eval_data: str | None = None
    extra_train_data: list[ExtraData] = Field(default_factory=list)
    max_length: int = 256

    loss: Literal["ce", "focal"] = "ce"  # for multi-label: BCE, or focal BCE
    focal_gamma: float = 2.0
    class_weights: Literal["none", "balanced"] = "none"  # balanced: the weights prepare_data computed
    label_smoothing: float = 0.0  # gold rows only; soft-labelled rows are used as they are
    soft_label_weight: float = 1.0  # loss weight of rows that carry `probs`

    llrd: float = 1.0  # layer-wise lr decay per layer from the top; 1.0 turns it off
    head_lr_multiplier: float = 1.0  # the classification head's lr relative to learning_rate
    adam_betas: tuple[float, float] = (0.9, 0.999)
    adam_eps: float = 1e-8
    head_warmup_epochs: float = 0.0  # LP-FT: train only the head this long, then everything
    head_warmup_lr: float = 1e-3
    classifier_dropout: float | None = None

    train: TrainArgs = Field(default_factory=lambda: TrainArgs(learning_rate=3e-5, num_train_epochs=3))
    resume: bool = True  # as in mlm_adapt
    keep_checkpoints: bool = False
    trust_remote_code: bool = False

    @field_validator("extra_train_data", mode="before")
    @classmethod
    def _extra(cls, v: Any) -> Any:
        return [{"path": x} if isinstance(x, str) else x for x in (_paths(v) or [])]


def _keep_columns(kwargs: dict[str, Any]) -> dict[str, Any]:
    """The Trainer drops batch keys the model's `forward` does not name; `sample_weight` is one."""
    kwargs["remove_unused_columns"] = False
    return kwargs


def read_rows(path: str, labels: LabelSpace, weight: float = 1.0, max_samples: int | None = None) -> list[dict[str, Any]]:
    """Training rows from a JSONL file, checked against the label space."""
    key = "labels" if labels.multi else "label"
    rows = []
    for row in read_jsonl(path):
        if key not in row and row.get("probs") is None:
            raise KeyError(f"{path}: a row has neither {key!r} nor 'probs'; keys: {sorted(row)}")
        if weight != 1.0:
            row["weight"] = float(row.get("weight", 1.0)) * weight
        rows.append(row)
        if max_samples and len(rows) >= max_samples:
            break
    return rows


def make_compute_metrics(labels: LabelSpace):
    """Trainer metrics on the eval split: `eval/accuracy`, `eval/macro_f1`, `eval/micro_f1`, ..."""
    from .metrics import classification_report, decide, probabilities

    def compute(pred) -> dict[str, float]:
        logits = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
        targets = pred.label_ids
        # Eval rows are gold: the argmax of a (smoothed) one-hot row is its class.
        y = targets >= 0.5 if labels.multi else targets.argmax(-1)
        guess = decide(probabilities(logits, labels.problem_type), labels.problem_type)
        return classification_report(guess, y, labels.names, labels.problem_type)["metrics"]

    return compute


@register("classifiers.finetune")
class Finetune(Stage):
    """Supervised fine-tuning on gold (and optionally soft-labelled) rows."""

    Params = FinetuneParams

    def run(self, ctx: StageContext) -> StageResult:
        import torch
        from transformers import TrainingArguments

        from .modeling import (
            ClassifierTrainer,
            Collator,
            build_dataset,
            classification_loss,
            freeze_backbone,
            llrd_groups,
            load_classifier,
        )

        p: FinetuneParams = self.params
        seed = self.seed(ctx)
        seed_everything(seed)
        labels = LabelSpace.load(p.labels)
        rows = read_rows(p.train_data, labels)
        for extra in p.extra_train_data:
            rows += read_rows(extra.path, labels, extra.weight, extra.max_samples)
        eval_rows = read_rows(p.eval_data, labels) if p.eval_data else []

        model, tokenizer = load_classifier(p.init_from, labels, p.trust_remote_code, p.classifier_dropout)
        train_set = build_dataset(tokenizer, rows, labels, p.max_length, p.label_smoothing, p.soft_label_weight)
        eval_set = build_dataset(tokenizer, eval_rows, labels, p.max_length)
        has_eval = len(eval_set) > 0

        weights = None
        if p.class_weights == "balanced":
            if not labels.class_weights:
                raise ValueError(f"{p.labels} carries no class weights; they are written by classifiers.prepare_data")
            weights = torch.tensor(labels.class_weights, dtype=torch.float32)

        def loss_fn(logits, targets, sample_weight):
            return classification_loss(
                logits, targets, sample_weight, labels.problem_type,
                weights.to(logits.device) if weights is not None else None,
                p.focal_gamma if p.loss == "focal" else 0.0,
            )

        common = {
            "model": model, "train_dataset": train_set, "processing_class": tokenizer,
            "data_collator": Collator(tokenizer), "loss_fn": loss_fn,
        }
        metrics: dict[str, float] = {"n_train": float(len(train_set)), "n_eval": float(len(eval_set))}
        soft = sum(1 for r in rows if r.get("probs") is not None)
        metrics["n_soft"] = float(soft)

        checkpoints = ctx.out_dir / "checkpoints"
        last = resume_point(checkpoints, p.resume and ctx.resumed)
        if p.head_warmup_epochs > 0 and last is None:  # a resumed run is already past its warm-up
            # Linear probing first: a randomly initialised head otherwise sends large gradients
            # through the pretrained encoder in the first steps.
            trainable = freeze_backbone(model, True)
            ctx.log.info("finetune: head warm-up, %d trainable parameters", trainable)
            warm = p.train.model_copy(update={
                "learning_rate": p.head_warmup_lr, "num_train_epochs": p.head_warmup_epochs, "max_steps": -1,
                "load_best_model_at_end": False, "eval_steps": None, "save_steps": None, "lr_scheduler_type": "linear",
            })
            kwargs = _keep_columns(warm.to_kwargs(str(ctx.out_dir / "warmup"), seed, False))
            kwargs["save_strategy"] = "no"
            out = ClassifierTrainer(args=TrainingArguments(**kwargs), **common).train()
            metrics["warmup/loss"] = float(out.training_loss)
            shutil.rmtree(ctx.out_dir / "warmup", ignore_errors=True)
            freeze_backbone(model, False)

        args = TrainingArguments(
            **_keep_columns(p.train.to_kwargs(str(checkpoints), seed, has_eval, f"{ctx.pipeline}-{ctx.stage}"))
        )
        optimizers: tuple[Any, Any] = (None, None)
        if p.llrd != 1.0 or p.head_lr_multiplier != 1.0:
            groups = llrd_groups(
                model.named_parameters(), p.train.learning_rate, p.llrd, p.train.weight_decay,
                p.head_lr_multiplier, model.base_model_prefix,
            )
            for g in groups:
                g.pop("name")
            # The Trainer builds its scheduler on top of this optimizer; it scales every group alike.
            optimizers = (torch.optim.AdamW(groups, betas=p.adam_betas, eps=p.adam_eps), None)
            metrics["lr/lowest"] = min(g["lr"] for g in groups)
        trainer = ClassifierTrainer(
            args=args, eval_dataset=eval_set if has_eval else None, optimizers=optimizers,
            compute_metrics=make_compute_metrics(labels) if has_eval else None, **common,
        )
        out = trainer.train(resume_from_checkpoint=last)
        metrics["train/loss"] = float(out.training_loss)
        if has_eval:
            metrics.update(final_metrics(trainer.evaluate()))

        model_dir = ctx.out_dir / "model"
        _finish(trainer, tokenizer, model_dir, checkpoints, p.keep_checkpoints)
        labels.save(model_dir / LABELS_FILE)
        return StageResult(
            artifacts={"model": model_dir}, metrics=metrics,
            notes={"train_sources": source_counts(rows), **_resume_note(last)},
        )
