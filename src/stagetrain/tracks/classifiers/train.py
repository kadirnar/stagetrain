"""`classifiers.mlm_adapt` and `classifiers.finetune`: the two stages that update weights."""

from __future__ import annotations

import hashlib
import math
import shutil
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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
    write_jsonl,
)
from stagetrain.core.history import history_callback, save_history

from .labels import LABELS_FILE, LabelSpace, source_counts, text_key


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
    # 0.15 is the rate to start from when the goal is classification. ModernBERT was pretrained
    # at 0.3, but the two newest pretraining studies find classification falls as the rate rises.
    # Losses at different rates are not comparable: decide by the downstream metric.
    mlm_probability: float = 0.15
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
        trainer.add_callback(history_callback(ctx.out_dir))
        out = trainer.train(resume_from_checkpoint=last)
        save_history(ctx.out_dir, trainer.state.log_history)
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
    """Supervised fine-tuning. What decides the result, in this order: the learning rate, the
    number of epochs, the encoder and its size, and the seed. Sweep the first three with a
    `matrix` and judge each setting by its mean over at least three seeds; the defaults below
    are a starting point for a ModernBERT-base-sized encoder, not a tuned recipe."""

    init_from: str  # a Hub id, or the `model` artifact of mlm_adapt or of an earlier finetune
    labels: str  # labels.json from prepare_data
    train_data: str
    eval_data: str | None = None
    extra_train_data: list[ExtraData] = Field(default_factory=list)

    # `auto`: the length that fits `length_percentile` percent of the training rows, up to the
    # model's limit (and `max_length_cap`, to bound the cost on long documents).
    max_length: int | Literal["auto"] = "auto"
    length_percentile: float = Field(default=99.0, gt=0.0, le=100.0)
    max_length_cap: int | None = None
    truncation: Literal["head", "head_tail"] = "head"  # head_tail: keep the start and the end of an over-long text
    pooling: str | None = None  # `classifier_pooling` of ModernBERT-family models: mean or cls; None keeps the model's

    # ce: plain cross-entropy (BCE for multi-label), the default the evidence supports.
    # logit_adjusted: for single-label class imbalance, see `logit_adjustment_tau`.
    loss: Literal["ce", "focal", "logit_adjusted"] = "ce"
    focal_gamma: float = 2.0
    logit_adjustment_tau: float = Field(default=1.0, gt=0.0)  # strength of the prior correction
    class_weights: Literal["none", "balanced"] = "none"  # balanced: the weights prepare_data computed
    label_smoothing: float = 0.0  # gold rows only. Leave at 0 for a teacher whose confidence is thresholded or distilled
    soft_label_weight: float = 1.0  # loss weight of rows that carry `probs`

    # Search dimensions, not defaults: none of these has controlled support on current encoders.
    llrd: float = 1.0  # layer-wise lr decay per layer from the top; 1.0 turns it off
    head_lr_multiplier: float = 1.0  # the classification head's lr relative to learning_rate
    adam_betas: tuple[float, float] = (0.9, 0.999)
    adam_eps: float = 1e-8
    head_warmup_epochs: float = 0.0  # LP-FT: train only the head this long, then everything
    head_warmup_lr: float = 1e-3
    classifier_dropout: float | None = None

    # Stop when the eval metric has not improved for this many evaluations, and keep the best
    # checkpoint. Evaluate several times per epoch (`train.eval_steps`) for this to be useful.
    early_stopping_patience: int | None = Field(default=None, ge=1)

    # Cross-validation: hold fold `fold` of `folds` out of the training rows (by a hash of the
    # text), train on the rest, and write the held-out rows' predictions as the `oof` artifact.
    # Run it as a matrix over `fold` and hand the runs to classifiers.audit_labels.
    fold: int | None = Field(default=None, ge=0)
    folds: int = Field(default=5, ge=2)

    train: TrainArgs = Field(default_factory=lambda: TrainArgs(learning_rate=5e-5, num_train_epochs=3))
    resume: bool = True  # as in mlm_adapt
    keep_checkpoints: bool = False
    trust_remote_code: bool = False

    @field_validator("extra_train_data", mode="before")
    @classmethod
    def _extra(cls, v: Any) -> Any:
        return [{"path": x} if isinstance(x, str) else x for x in (_paths(v) or [])]

    @model_validator(mode="after")
    def _consistent(self) -> FinetuneParams:
        if self.early_stopping_patience and not self.eval_data:
            raise ValueError("early_stopping_patience needs eval_data")
        if self.fold is not None and self.fold >= self.folds:
            raise ValueError(f"fold must be below folds ({self.folds}), got {self.fold}")
        if self.loss == "logit_adjusted" and self.class_weights != "none":
            raise ValueError("logit_adjusted already corrects for class frequency; use it without class_weights")
        return self

    def train_args(self) -> TrainArgs:
        """`train`, with what early stopping needs: the best checkpoint kept, by macro-F1 unless
        another metric is named."""
        if not self.early_stopping_patience:
            return self.train
        update: dict[str, Any] = {"load_best_model_at_end": True}
        if self.train.metric_for_best_model is None:
            update.update(metric_for_best_model="eval_macro_f1", greater_is_better=True)
        return self.train.model_copy(update=update)


def fold_of(row: dict[str, Any], folds: int) -> int:
    """Which fold a training row belongs to: decided by its text alone, so it is the same in
    every run and for every seed."""
    return int(hashlib.sha1(text_key(row).encode("utf-8")).hexdigest()[:8], 16) % folds


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
            auto_max_length,
            build_dataset,
            classification_loss,
            freeze_backbone,
            llrd_groups,
            load_classifier,
            log_prior,
            model_limit,
            token_lengths,
            write_preprocessing,
        )

        p: FinetuneParams = self.params
        seed = self.seed(ctx)
        seed_everything(seed)
        labels = LabelSpace.load(p.labels)
        rows = read_rows(p.train_data, labels)
        held_out: list[dict[str, Any]] = []
        if p.fold is not None:
            held_out = [r for r in rows if fold_of(r, p.folds) == p.fold]
            rows = [r for r in rows if fold_of(r, p.folds) != p.fold]
            if not rows or not held_out:
                raise ValueError(f"fold {p.fold} of {p.folds} leaves {len(rows)} training and {len(held_out)} held-out rows")
        for extra in p.extra_train_data:
            rows += read_rows(extra.path, labels, extra.weight, extra.max_samples)
        eval_rows = read_rows(p.eval_data, labels) if p.eval_data else []

        model, tokenizer, model_notes = load_classifier(
            p.init_from, labels, p.trust_remote_code, p.classifier_dropout, p.pooling
        )
        lengths = token_lengths(tokenizer, rows)
        limit = min(model_limit(tokenizer, model.config), p.max_length_cap or 10**9)
        max_length = auto_max_length(lengths, p.length_percentile, limit) if p.max_length == "auto" else p.max_length
        truncated = sum(n > max_length for n in lengths) / len(lengths)
        if truncated > 0.05:
            ctx.log.warning(
                "finetune: %.0f%% of the training rows are longer than max_length=%d and are cut (%s)",
                100 * truncated, max_length, p.truncation,
            )
        train_set = build_dataset(
            tokenizer, rows, labels, max_length, p.label_smoothing, p.soft_label_weight, p.truncation
        )
        eval_set = build_dataset(tokenizer, eval_rows, labels, max_length, truncation=p.truncation)
        has_eval = len(eval_set) > 0

        weights = None
        if p.class_weights == "balanced":
            if not labels.class_weights:
                raise ValueError(f"{p.labels} carries no class weights; they are written by classifiers.prepare_data")
            weights = torch.tensor(labels.class_weights, dtype=torch.float32)
        bias = None
        if p.loss == "logit_adjusted":
            if labels.multi:
                raise ValueError("loss: logit_adjusted is for single-label tasks; multi-label imbalance is handled by thresholds")
            bias = torch.tensor(log_prior(labels.counts or [], p.logit_adjustment_tau), dtype=torch.float32)

        def loss_fn(logits, targets, sample_weight):
            return classification_loss(
                logits, targets, sample_weight, labels.problem_type,
                weights.to(logits.device) if weights is not None else None,
                p.focal_gamma if p.loss == "focal" else 0.0,
                # Training batches only: evaluation reads the raw logits, as prediction will.
                bias if model.training else None,
            )

        common = {
            "model": model, "train_dataset": train_set, "processing_class": tokenizer,
            "data_collator": Collator(tokenizer), "loss_fn": loss_fn,
        }
        metrics: dict[str, float] = {
            "n_train": float(len(train_set)), "n_eval": float(len(eval_set)),
            "data/max_length": float(max_length), "data/truncated_fraction": truncated,
        }
        soft = sum(1 for r in rows if r.get("probs") is not None)
        metrics["n_soft"] = float(soft)
        train_args = p.train_args()

        checkpoints = ctx.out_dir / "checkpoints"
        last = resume_point(checkpoints, p.resume and ctx.resumed)
        if p.head_warmup_epochs > 0 and last is None:  # a resumed run is already past its warm-up
            # Linear probing first: a randomly initialised head otherwise sends large gradients
            # through the pretrained encoder in the first steps.
            trainable = freeze_backbone(model, True)
            ctx.log.info("finetune: head warm-up, %d trainable parameters", trainable)
            warm = train_args.model_copy(update={
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
            **_keep_columns(train_args.to_kwargs(str(checkpoints), seed, has_eval, f"{ctx.pipeline}-{ctx.stage}"))
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
        trainer.add_callback(history_callback(ctx.out_dir))
        if p.early_stopping_patience:
            from transformers import EarlyStoppingCallback

            trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=p.early_stopping_patience))
        out = trainer.train(resume_from_checkpoint=last)
        save_history(ctx.out_dir, trainer.state.log_history)
        metrics["train/loss"] = float(out.training_loss)
        metrics["train/steps"] = float(trainer.state.global_step)
        if has_eval:
            metrics.update(final_metrics(trainer.evaluate()))

        artifacts: dict[str, Any] = {}
        if held_out:
            artifacts["oof"], oof_metrics = write_out_of_fold(
                ctx, trainer, tokenizer, held_out, labels, max_length, p.truncation, p.fold
            )
            metrics.update(oof_metrics)
        model_dir = ctx.out_dir / "model"
        _finish(trainer, tokenizer, model_dir, checkpoints, p.keep_checkpoints)
        labels.save(model_dir / LABELS_FILE)
        write_preprocessing(model_dir, max_length, p.truncation)
        artifacts["model"] = model_dir
        return StageResult(
            artifacts=artifacts, metrics=metrics,
            notes={"train_sources": source_counts(rows), **model_notes, **_resume_note(last)},
        )


def write_out_of_fold(
    ctx: StageContext, trainer, tokenizer, held_out: list[dict[str, Any]], labels: LabelSpace, max_length: int,
    truncation: str, fold: int,
) -> tuple[Path, dict[str, float]]:
    """Predict the rows this run did not train on and write them with their loss: `oof.jsonl`.
    A row the model gets confidently wrong without ever having seen it is a candidate label error."""
    from .metrics import classification_report, decide, probabilities, row_losses, targets_array
    from .modeling import build_dataset

    pred = trainer.predict(build_dataset(tokenizer, held_out, labels, max_length, truncation=truncation))
    logits = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
    kind = labels.problem_type
    y = targets_array(held_out, len(labels), kind)
    probs = probabilities(logits, kind)
    guess = decide(probs, kind)
    losses = row_losses(logits, y, kind)
    path = ctx.path("oof.jsonl")
    write_jsonl(path, (
        {**{k: row[k] for k in ("text", "text_pair", "label", "labels") if k in row}, "fold": fold,
         "loss": round(float(losses[i]), 6), "probs": [round(float(x), 6) for x in probs[i]],
         "predicted": [c for c in range(len(labels)) if guess[i][c]] if labels.multi else int(guess[i])}
        for i, row in enumerate(held_out)
    ))
    report = classification_report(guess, y, labels.names, kind)["metrics"]
    return path, {
        "n_oof": float(len(held_out)), "oof/loss": float(losses.mean()),
        "oof/accuracy": report["accuracy"], "oof/macro_f1": report["macro_f1"],
    }
