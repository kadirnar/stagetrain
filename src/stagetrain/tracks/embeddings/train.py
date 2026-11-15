"""Training a bi-encoder. `finetune` and `pretrain` share one implementation and differ in what
they are for, and so in their defaults:

    finetune   a published embedding model (or an earlier stage's) on pairs with mined hard
               negatives: contrastive, distilled from a teacher's scores, or both
    pretrain   a plain encoder on very many weak pairs with in-batch negatives only. Worth it only
               with tens of millions of pairs and contrastive batches in the thousands; below
               that, starting from a published embedder is the better first stage
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ...core.data import DataSource, load_source, write_json
from ...core.history import history_callback, save_history
from ...core.manifest import digest
from ...core.stage import Stage, StageContext, StageParams, StageResult, register
from ...core.training import TrainArgs, final_metrics, seed_everything
from .evaluate import make_evaluator, retrieval_metrics
from .model import DOCUMENT, QUERY, EncoderSpec, PromptSpec, load_encoder

SCORES = "scores"  # per row: [teacher(query, positive), teacher(query, negative_1), ...]
_NEGATIVE = re.compile(r"^negative(_\d+)?$")
CONTRASTIVE = ("mnrl", "cached_mnrl", "gist", "cached_gist")
DISTILLATION = ("kl_div", "margin_mse")


class LossSpec(BaseModel):
    """The training objective.

    Contrastive (`name`):
      mnrl          InfoNCE over in-batch negatives and every row's hard negatives
      cached_mnrl   the same with gradient caching: the contrastive batch can be far larger than
                    what fits in memory; `mini_batch_size` is what is encoded at a time
      gist          InfoNCE in which a `guide` model masks in-batch candidates it scores as high
                    as the positive: likely false negatives stop being pushed away. Reads at
                    most one hard negative per row
      cached_gist   gist with gradient caching; reads every hard negative of a row

    Distillation of a teacher's scores (the `scores` column a mining stage writes):
      kl_div        listwise: the student's ranking of [positive, negatives] matches the teacher's
      margin_mse    pairwise: the student's positive-negative margins match the teacher's

    Use a distillation loss alone (`name: kl_div`), or add it to `mnrl` / `gist` with
    `distill_weight`. Fine-tuning an already supervised embedder with InfoNCE alone often lowers
    it; the combination that gained in the controlled comparison is listwise KL plus a small
    contrastive term: `distill_weight: 1.0, contrastive_weight: 0.1`.
    """

    model_config = ConfigDict(extra="forbid")

    name: Literal["mnrl", "cached_mnrl", "gist", "cached_gist", "kl_div", "margin_mse"] = "mnrl"
    # 1 / temperature of the contrastive loss. Recent embedding models train at a temperature of
    # 0.01-0.02 (scale 50-100), the library defaults to 0.05 (scale 20), and no published sweep
    # settles it: sweep it, starting from the value the checkpoint was trained with if known.
    scale: float = Field(default=50.0, gt=0.0)
    symmetric: bool = False  # mnrl: also rank queries for each document, as a second softmax
    mini_batch_size: int = 32
    matryoshka_dims: list[int] | None = None  # e.g. [768, 512, 256, 128]; include the full dimension
    matryoshka_weights: list[float] | None = None

    guide: str | None = None  # gist: a Hub id or path of the guide embedder
    guide_margin_strategy: Literal["absolute", "relative"] = "absolute"
    guide_margin: float = 0.0  # 0: mask only candidates the guide scores above the positive

    distill_weight: float = Field(default=0.0, ge=0.0)  # weight of the distillation term added to mnrl / gist
    contrastive_weight: float = Field(default=1.0, ge=0.0)  # weight of the contrastive term next to it
    distill_loss: Literal["kl_div", "margin_mse"] = "kl_div"
    # kl_div: the student's scores are similarity / student_temperature, the teacher's are
    # score / teacher_temperature after min-max normalisation of each query's scores to [0, 1]
    # (which makes the teacher's own scale, logits or probabilities, irrelevant).
    student_temperature: float = Field(default=0.05, gt=0.0)
    teacher_temperature: float = Field(default=0.3, gt=0.0)
    normalize_scores: bool = True
    # margin_mse matches distill_scale x similarity to the teacher's raw margins, so the scale must
    # let the student's margins reach the teacher's: a cosine is at most 2 apart, probabilities at
    # most 1 (scale 1 or less), logits several units (scale 10-20).
    distill_similarity: Literal["dot", "cosine"] = "cosine"
    distill_scale: float = Field(default=1.0, gt=0.0)

    @model_validator(mode="after")
    def _check(self) -> LossSpec:
        if self.distill_weight > 0 and self.name not in ("mnrl", "gist"):
            raise ValueError(
                "distill_weight combines with `mnrl` or `gist` only; use name: kl_div or margin_mse to distil alone"
            )
        if self.name in ("gist", "cached_gist") and not self.guide:
            raise ValueError(f"loss {self.name!r} needs a guide model")
        if self.matryoshka_weights and len(self.matryoshka_weights) != len(self.matryoshka_dims or []):
            raise ValueError("matryoshka_weights must match matryoshka_dims")
        return self

    @property
    def distillation(self) -> str | None:
        """The distillation loss in use, alone or as a term, or None."""
        if self.name in DISTILLATION:
            return self.name
        return self.distill_loss if self.distill_weight > 0 else None

    @property
    def needs_scores(self) -> bool:
        return self.distillation is not None


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
    # An already contrastive checkpoint is fine-tuned at 3e-6 to 4e-5 in the published recipes; a
    # plain encoder needs about ten times more. One epoch is the norm.
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


def normalize_scores(row: dict[str, Any]) -> dict[str, Any]:
    """One query's teacher scores stretched to [0, 1]. The ranking and the relative gaps stay; the
    teacher's scale goes, so one teacher temperature fits logits and probabilities alike."""
    scores = row[SCORES]
    low, high = min(scores), max(scores)
    if high == low:
        return {SCORES: [0.5] * len(scores)}
    return {SCORES: [(s - low) / (high - low) for s in scores]}


def build_loss(model, spec: LossSpec):
    """The loss module for `spec`, wrapped for Matryoshka dimensions when asked."""
    import torch
    from sentence_transformers import util
    from sentence_transformers.sentence_transformer import losses

    similarity = util.pairwise_dot_score if spec.distill_similarity == "dot" else util.pairwise_cos_sim
    # Symmetric InfoNCE is two softmaxes, one per direction, not one softmax over both.
    directions = {"directions": ("query_to_doc", "doc_to_query"), "partition_mode": "per_direction"} if spec.symmetric else {}

    def matryoshka(loss):
        if not spec.matryoshka_dims:
            return loss
        return losses.MatryoshkaLoss(model, loss, spec.matryoshka_dims, matryoshka_weights=spec.matryoshka_weights)

    class ListwiseKL(losses.DistillKLDivLoss):
        """KL divergence between the teacher's and the student's ranking of one query's candidates.
        The library multiplies it by student_temperature squared, which at 0.05 makes it 400 times
        smaller than a contrastive term next to it; this undoes that, so the weights mean what
        they say."""

        def compute_loss_from_embeddings(self, embeddings, labels):
            return super().compute_loss_from_embeddings(embeddings, labels) / self.student_temperature**2

    def distillation():
        if spec.distillation == "kl_div":
            return ListwiseKL(
                model, similarity_fct=similarity, student_temperature=spec.student_temperature,
                teacher_temperature=spec.teacher_temperature,
            )
        scale = spec.distill_scale
        fct = similarity if scale == 1.0 else lambda a, b: similarity(a, b) * scale
        return losses.MarginMSELoss(model, similarity_fct=fct)

    def guide():
        from .model import load_encoder as load

        return load(spec.guide, device=str(model.device))

    if spec.name in DISTILLATION:
        return matryoshka(distillation())
    if spec.name == "cached_mnrl":
        return matryoshka(losses.CachedMultipleNegativesRankingLoss(
            model, scale=spec.scale, mini_batch_size=spec.mini_batch_size, **directions
        ))
    if spec.name == "cached_gist":
        return matryoshka(losses.CachedGISTEmbedLoss(
            model, guide(), temperature=1.0 / spec.scale, mini_batch_size=spec.mini_batch_size,
            margin_strategy=spec.guide_margin_strategy, margin=spec.guide_margin,
        ))
    if spec.name == "gist":
        contrastive = losses.GISTEmbedLoss(
            model, guide(), temperature=1.0 / spec.scale, margin_strategy=spec.guide_margin_strategy,
            margin=spec.guide_margin,
        )
    else:
        contrastive = losses.MultipleNegativesRankingLoss(model, scale=spec.scale, **directions)
    contrastive = matryoshka(contrastive)
    if spec.distill_weight <= 0:
        return contrastive

    class ContrastiveWithDistillation(torch.nn.Module):
        """`contrastive_weight` x InfoNCE on the texts plus `distill_weight` x distillation of the teacher's scores."""

        def __init__(self, contrastive, distill, contrastive_weight: float, distill_weight: float):
            super().__init__()
            self.contrastive, self.distill = contrastive, distill
            self.contrastive_weight, self.distill_weight = contrastive_weight, distill_weight

        def forward(self, sentence_features, labels):
            loss = self.distill_weight * self.distill(sentence_features, labels)
            if self.contrastive_weight > 0:
                loss = loss + self.contrastive_weight * self.contrastive(sentence_features, labels)
            return loss

        def get_config_dict(self) -> dict[str, Any]:
            return {
                "contrastive": type(self.contrastive).__name__, "contrastive_weight": self.contrastive_weight,
                "distill": spec.distillation, "distill_weight": self.distill_weight,
            }

    return ContrastiveWithDistillation(contrastive, matryoshka(distillation()), spec.contrastive_weight, spec.distill_weight)


def train_embedding(p: TrainEmbeddingParams, ctx: StageContext, seed: int) -> StageResult:
    from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments

    seed_everything(seed)
    model = load_encoder(p.init_from, p.encoder)

    train = load_source(DataSource(path=p.train_data, max_samples=p.max_train_samples))
    columns = order_columns(train.column_names, p.max_negatives, p.loss.needs_scores)
    train = train.select_columns(columns)
    n_negatives = len(columns) - 2 - int(p.loss.needs_scores)
    if p.loss.name == "gist" and n_negatives > 1:
        raise ValueError(
            f"loss 'gist' reads at most one hard negative per row and the data has {n_negatives}; "
            "set max_negatives: 1, or use cached_gist, which reads them all"
        )
    if p.loss.needs_scores:
        if n_negatives < 1:
            raise ValueError("distilling teacher scores needs at least one negative column")
        train = train.map(trim_scores, fn_kwargs={"n_negatives": n_negatives})
        if p.loss.distillation == "kl_div" and p.loss.normalize_scores:
            train = train.map(normalize_scores)

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
    trainer.add_callback(history_callback(ctx.out_dir))
    output = trainer.train(resume_from_checkpoint=True if resume else None)
    save_history(ctx.out_dir, trainer.state.log_history)
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
    """Contrastive pre-training of a plain encoder on very many weak pairs (cached InfoNCE); not part of the default recipe."""

    Params = PretrainParams

    def run(self, ctx: StageContext) -> StageResult:
        return train_embedding(self.params, ctx, self.seed(ctx))


@register("embeddings.finetune")
class Finetune(Stage):
    """Fine-tune an embedding model on pairs with hard negatives: contrastive, distilled from a teacher's scores, or both."""

    Params = FinetuneParams

    def run(self, ctx: StageContext) -> StageResult:
        return train_embedding(self.params, ctx, self.seed(ctx))
