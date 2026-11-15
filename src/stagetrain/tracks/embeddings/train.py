"""Contrastive training of a bi-encoder. `pretrain` and `finetune` share one implementation and
differ in what they are for, and so in their defaults:

    pretrain   large, weakly-supervised pairs; in-batch negatives only, big (cached) batches
    finetune   supervised pairs with mined hard negatives, from an earlier stage's model;
               optionally Matryoshka dimensions and teacher-score distillation
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ...core.data import DataSource, load_source, write_json
from ...core.manifest import digest
from ...core.stage import Stage, StageContext, StageParams, StageResult, register
from ...core.training import TrainArgs, final_metrics, seed_everything
from .evaluate import make_evaluator, retrieval_metrics
from .model import DOCUMENT, QUERY, EncoderSpec, PromptSpec, load_encoder

SCORES = "scores"  # per row: [teacher(query, positive), teacher(query, negative_1), ...]
_NEGATIVE = re.compile(r"^negative(_\d+)?$")


class LossSpec(BaseModel):
    """The training objective.

    `mnrl`: InfoNCE with in-batch negatives (and the row's own hard negatives). `cached_mnrl`: the
    same with gradient caching, so the contrastive batch can be far larger than what fits in
    memory at once; `mini_batch_size` is what is encoded at a time. `margin_mse`: only distil a
    teacher's score margins. `distill_weight` adds the margin distillation term to `mnrl`.
    """

    model_config = ConfigDict(extra="forbid")

    name: Literal["mnrl", "cached_mnrl", "margin_mse"] = "mnrl"
    scale: float = 20.0  # 1 / temperature
    symmetric: bool = False  # also rank queries for each document
    mini_batch_size: int = 32
    matryoshka_dims: list[int] | None = None  # e.g. [768, 512, 256, 128, 64]; include the full dimension
    matryoshka_weights: list[float] | None = None
    distill_weight: float = Field(default=0.0, ge=0.0)
    # The student's score that is matched to the teacher's: distill_scale x (dot product | cosine).
    # A model that ends in Normalize can only produce margins up to 2, so give it a scale that
    # lets its margins reach the teacher's (e.g. 20 for cross-encoder logits).
    distill_similarity: Literal["dot", "cosine"] = "dot"
    distill_scale: float = Field(default=1.0, gt=0.0)

    @model_validator(mode="after")
    def _check(self) -> LossSpec:
        if self.distill_weight > 0 and self.name != "mnrl":
            raise ValueError("distill_weight combines with `mnrl` only; use name: margin_mse to distil alone")
        if self.matryoshka_weights and len(self.matryoshka_weights) != len(self.matryoshka_dims or []):
            raise ValueError("matryoshka_weights must match matryoshka_dims")
        return self

    @property
    def needs_scores(self) -> bool:
        return self.name == "margin_mse" or self.distill_weight > 0


class TrainEmbeddingParams(StageParams):
    init_from: str  # a Hub id, a local path, or an earlier stage's `${stage.model}`
    train_data: str  # jsonl: query, positive[, negative | negative_1..n][, scores]
    eval_pairs: str | None = None  # held-out pairs, for the eval loss
    eval_set: str | None = None  # retrieval eval set, evaluated during and after training
    encoder: EncoderSpec = Field(default_factory=EncoderSpec)
    loss: LossSpec = Field(default_factory=LossSpec)
    batch_sampler: Literal["batch_sampler", "no_duplicates", "no_duplicates_hashed"] = "no_duplicates"
    max_negatives: int | None = None  # use at most this many negative columns
    max_train_samples: int | None = None
    eval_dims: list[int] = Field(default_factory=list)  # truncated dimensions for the final evaluation
    eval_batch_size: int = 64
    resume: bool = True  # continue from the stage's last checkpoint if it was interrupted
    train: TrainArgs = Field(default_factory=TrainArgs)


class PretrainParams(TrainEmbeddingParams):
    loss: LossSpec = Field(default_factory=lambda: LossSpec(name="cached_mnrl"))
    train: TrainArgs = Field(default_factory=lambda: TrainArgs(learning_rate=5e-5, per_device_train_batch_size=256))


class FinetuneParams(TrainEmbeddingParams):
    train: TrainArgs = Field(default_factory=lambda: TrainArgs(learning_rate=1e-5, per_device_train_batch_size=32))


def order_columns(columns: list[str], max_negatives: int | None, keep_scores: bool) -> list[str]:
    """Training columns in the order the losses read them: query, positive, negatives, then the
    label column. Anything else in the file is left out."""
    for col in ("query", "positive"):
        if col not in columns:
            raise KeyError(f"training data has no {col!r} column; columns: {columns}")
    negatives = sorted(
        (c for c in columns if _NEGATIVE.match(c)), key=lambda c: int(c.split("_")[1]) if "_" in c else 0
    )
    if max_negatives is not None:
        negatives = negatives[:max_negatives]
    out = ["query", "positive", *negatives]
    if keep_scores:
        if SCORES not in columns:
            raise KeyError(
                f"this loss distils teacher scores, but the training data has no {SCORES!r} column; "
                "mine with output_scores (and a cross_encoder teacher)"
            )
        out.append(SCORES)
    return out


def trim_scores(row: dict[str, Any], n_negatives: int) -> dict[str, Any]:
    return {SCORES: row[SCORES][: 1 + n_negatives]}


def build_loss(model, spec: LossSpec):
    """The loss module for `spec`, wrapped for Matryoshka dimensions when asked."""
    import torch
    from sentence_transformers import util
    from sentence_transformers.sentence_transformer import losses

    directions = ("query_to_doc", "doc_to_query") if spec.symmetric else ("query_to_doc",)

    def matryoshka(loss):
        if not spec.matryoshka_dims:
            return loss
        return losses.MatryoshkaLoss(model, loss, spec.matryoshka_dims, matryoshka_weights=spec.matryoshka_weights)

    def margin_mse():
        base = util.pairwise_dot_score if spec.distill_similarity == "dot" else util.pairwise_cos_sim
        scale = spec.distill_scale
        return losses.MarginMSELoss(model, similarity_fct=base if scale == 1.0 else lambda a, b: base(a, b) * scale)

    if spec.name == "margin_mse":
        return matryoshka(margin_mse())
    if spec.name == "cached_mnrl":
        return matryoshka(
            losses.CachedMultipleNegativesRankingLoss(
                model, scale=spec.scale, mini_batch_size=spec.mini_batch_size, directions=directions
            )
        )
    contrastive = matryoshka(losses.MultipleNegativesRankingLoss(model, scale=spec.scale, directions=directions))
    if spec.distill_weight <= 0:
        return contrastive

    class ContrastiveWithDistillation(torch.nn.Module):
        """InfoNCE on the texts plus `weight` x MarginMSE on the teacher's scores."""

        def __init__(self, contrastive, distill, weight: float):
            super().__init__()
            self.contrastive, self.distill, self.weight = contrastive, distill, weight

        def forward(self, sentence_features, labels):
            return self.contrastive(sentence_features, labels) + self.weight * self.distill(sentence_features, labels)

        def get_config_dict(self) -> dict[str, Any]:
            return {"contrastive": type(self.contrastive).__name__, "distill": "MarginMSELoss", "weight": self.weight}

    return ContrastiveWithDistillation(contrastive, matryoshka(margin_mse()), spec.distill_weight)


def train_embedding(p: TrainEmbeddingParams, ctx: StageContext, seed: int) -> StageResult:
    from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments

    seed_everything(seed)
    model = load_encoder(p.init_from, p.encoder)

    train = load_source(DataSource(path=p.train_data, max_samples=p.max_train_samples))
    columns = order_columns(train.column_names, p.max_negatives, p.loss.needs_scores)
    train = train.select_columns(columns)
    n_negatives = len(columns) - 2 - int(p.loss.needs_scores)
    if p.loss.needs_scores:
        if n_negatives < 1:
            raise ValueError("distilling teacher scores needs at least one negative column")
        train = train.map(trim_scores, fn_kwargs={"n_negatives": n_negatives})

    # Held-out pairs have no teacher scores, so the eval loss is only defined for the plain contrastive losses.
    eval_pairs = None
    if p.eval_pairs and not p.loss.needs_scores:
        eval_pairs = load_source(DataSource(path=p.eval_pairs)).select_columns(["query", "positive"])
    evaluator = make_evaluator(model, p.eval_set, p.eval_batch_size) if p.eval_set else None
    has_eval = eval_pairs is not None or evaluator is not None

    checkpoints = ctx.out_dir / "checkpoints"
    kwargs = p.train.to_kwargs(str(checkpoints), seed, has_eval, run_name=f"{ctx.pipeline}-{ctx.stage}")
    kwargs["batch_sampler"] = p.batch_sampler
    text_columns = [c for c in columns if c != SCORES]
    prompts = (p.encoder.prompts or _saved_prompts(model)).for_columns(text_columns)
    if prompts:
        kwargs["prompts"] = prompts
    args = SentenceTransformerTrainingArguments(**kwargs)

    trainer = SentenceTransformerTrainer(
        model=model, args=args, train_dataset=train, eval_dataset=eval_pairs, loss=build_loss(model, p.loss),
        evaluator=evaluator,
    )
    resume = claim_checkpoints(checkpoints, digest({"params": p.model_dump(mode="json"), "seed": seed}), p.resume)
    if resume:
        ctx.log.info("%s: resuming from the last checkpoint in %s", ctx.stage, checkpoints)
    output = trainer.train(resume_from_checkpoint=True if resume else None)
    write_json(checkpoints / RUN_MARKER, {**json.loads((checkpoints / RUN_MARKER).read_text()), "finished": True})

    model_dir = ctx.out_dir / "model"
    model.save_pretrained(str(model_dir))
    metrics = final_metrics(output.metrics)
    metrics["train/pairs"] = float(len(train))
    metrics["train/negatives_per_pair"] = float(n_negatives)
    if eval_pairs is not None:
        loss = trainer.evaluate().get("eval_loss")
        if loss is not None:
            metrics["eval/loss"] = float(loss)
    if p.eval_set:
        metrics.update(retrieval_metrics(model, p.eval_set, p.eval_dims, p.eval_batch_size))
    return StageResult(artifacts={"model": model_dir, "checkpoints": Path(checkpoints)}, metrics=metrics)


RUN_MARKER = "run.json"


def claim_checkpoints(checkpoints: Path, run_key: str, resume: bool = True) -> bool:
    """Prepare the checkpoint directory for a run identified by `run_key` (its parameters and seed)
    and say whether to resume.

    Checkpoints are resumed only when they belong to an unfinished run with the same key, i.e. the
    stage was interrupted. Checkpoints of a finished run (a forced rerun) or of other parameters
    are removed, so training never continues from a model trained under another configuration.
    """
    marker = checkpoints / RUN_MARKER
    previous = None
    if marker.is_file():
        try:
            previous = json.loads(marker.read_text())
        except json.JSONDecodeError:
            previous = None
    same_unfinished = bool(previous) and previous.get("key") == run_key and not previous.get("finished")
    if resume and same_unfinished and any(checkpoints.glob("checkpoint-*")):
        return True
    if checkpoints.exists():
        shutil.rmtree(checkpoints)
    write_json(marker, {"key": run_key, "finished": False})
    return False


def _saved_prompts(model) -> PromptSpec:
    """The query / document prompts a loaded checkpoint carries, so later stages keep training with them."""
    saved = getattr(model, "prompts", None) or {}
    return PromptSpec(query=saved.get(QUERY) or None, document=saved.get(DOCUMENT) or None)


@register("embeddings.pretrain")
class Pretrain(Stage):
    """Weakly-supervised contrastive pre-training on pairs with in-batch negatives (cached InfoNCE by default)."""

    Params = PretrainParams

    def run(self, ctx: StageContext) -> StageResult:
        return train_embedding(self.params, ctx, self.seed(ctx))


@register("embeddings.finetune")
class Finetune(Stage):
    """Supervised fine-tuning on mined hard negatives, optionally with Matryoshka dimensions and score distillation."""

    Params = FinetuneParams

    def run(self, ctx: StageContext) -> StageResult:
        return train_embedding(self.params, ctx, self.seed(ctx))
