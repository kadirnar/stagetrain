"""Hard-negative mining: for every (query, positive), documents a retriever ranks high that are
not the positive, filtered so that likely false negatives stay out.

Three choices decide what this stage is worth, in the order of their measured effect
(NV-Retriever, arXiv 2407.15831; see docs/research/embeddings.md):

- **Who mines.** A stronger embedder than the model being trained finds better negatives than
  the model itself (about +3 nDCG@10 in the one controlled comparison); BM25 negatives alone
  are worse than random ones. So `model` is normally a strong published embedder, not the
  student. Self-mining still works and is what you get by pointing `model` at the student.
- **The false-negative filter.** A candidate is kept only if it scores clearly below the
  positive: `relative_margin: 0.05` keeps what scores under 95 % of the positive's score. This
  one filter beat skipping the top ranks, an absolute score cap and an absolute margin.
- **Not skipping ranks on top of it.** The library applies `range_min` after the score filters,
  to the candidates that already passed them, so `range_min: 10` with a margin throws away the
  ten hardest negatives the filter approved. The default is therefore `range_min: 0`.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from ...core.data import DataSource, load_source, read_jsonl, write_json
from ...core.stage import Stage, StageContext, StageParams, StageResult, register
from .model import EncoderSpec, load_encoder, prompt_names
from .train import SCORES


class MineParams(StageParams):
    model: str  # the miner: a strong published embedder, or `${stage.model}` to self-mine
    train_data: str  # jsonl with query, positive
    corpus: str | None = None  # jsonl with `text`: more candidates than the positives of train_data
    num_negatives: int = Field(default=4, ge=1)  # 4-7 for contrastive training; up to ~19 for listwise distillation
    range_min: int = Field(default=0, ge=0)  # ranks dropped after filtering; see the module docstring
    range_max: int | None = 100  # how deep the miner retrieves
    # False-negative filters, on the miner's scores (or the teacher's when there is one). A candidate
    # is kept only if its score is below positive - |positive| * relative_margin, below
    # positive - absolute_margin, and below max_score.
    relative_margin: float | None = 0.05
    absolute_margin: float | None = None
    max_score: float | None = None
    min_score: float | None = None
    sampling_strategy: Literal["top", "random"] = "top"
    cross_encoder: str | None = None  # a teacher that rescores candidates; its scores drive the filters
    # What the teacher's scores are. A reranker with one output is read through a sigmoid by the
    # library, so its scores are probabilities in (0, 1): margins and caps are in those units.
    # `logit` reads the raw output instead (unbounded, typically -10 .. 10).
    teacher_scores: Literal["probability", "logit"] = "probability"
    output_scores: bool = False  # keep [positive, negative_1..n] scores, for distillation
    batch_size: int = 64
    use_faiss: bool = False
    max_samples: int | None = None
    encoder: EncoderSpec = Field(default_factory=EncoderSpec)  # how the miner is loaded (its prompts, max length)

    @model_validator(mode="after")
    def _window(self) -> MineParams:
        if self.range_max is not None and self.range_max - self.range_min < self.num_negatives:
            raise ValueError(
                f"the rank window [{self.range_min}, {self.range_max}) is smaller than num_negatives={self.num_negatives}"
            )
        if self.cross_encoder and self.teacher_scores == "probability":
            # Scores in (0, 1): a candidate must score below positive - margin, which no candidate
            # can when the margin is 1 or more. The same holds for a cap at or below 0.
            if self.absolute_margin is not None and self.absolute_margin >= 1:
                raise ValueError(
                    f"absolute_margin={self.absolute_margin} removes every candidate: the teacher's scores are "
                    "probabilities in (0, 1). Use a margin below 1, relative_margin, or teacher_scores: logit"
                )
            if self.max_score is not None and self.max_score <= 0:
                raise ValueError("max_score must be above 0 for a teacher whose scores are probabilities")
        return self


def load_teacher(path: str, scores: str = "probability", device: str | None = None):
    """A cross-encoder that scores (query, document) pairs, as probabilities or as raw logits."""
    import torch
    from sentence_transformers import CrossEncoder

    extra = {"activation_fn": torch.nn.Identity()} if scores == "logit" else {}
    return CrossEncoder(path, device=device, **extra)


def score_stats(scores: list[list[float]]) -> dict[str, float]:
    """Mean positive score, mean negative score and mean margin of mined rows."""
    if not scores:
        return {}
    pos = [row[0] for row in scores]
    neg = [s for row in scores for s in row[1:]]
    margins = [row[0] - s for row in scores for s in row[1:]]
    return {
        "mine/positive_score": sum(pos) / len(pos),
        "mine/negative_score": sum(neg) / len(neg),
        "mine/margin": sum(margins) / len(margins),
        "mine/min_margin": min(margins),
    }


@register("embeddings.mine_negatives")
class MineNegatives(Stage):
    """Mine hard negatives with a strong embedder over the corpus, keep those clearly below the positive; writes n-tuples."""

    Params = MineParams

    def run(self, ctx: StageContext) -> StageResult:
        from sentence_transformers import mine_hard_negatives

        p: MineParams = self.params
        model = load_encoder(p.model, p.encoder)
        pairs = load_source(DataSource(path=p.train_data, max_samples=p.max_samples)).select_columns(["query", "positive"])
        corpus = [r["text"] for r in read_jsonl(p.corpus)] if p.corpus else None
        teacher = load_teacher(p.cross_encoder, p.teacher_scores, p.encoder.device) if p.cross_encoder else None

        kwargs: dict[str, Any] = {}
        if p.use_faiss:
            kwargs["use_faiss"] = True
        mined = mine_hard_negatives(
            pairs,
            model,
            anchor_column_name="query",
            positive_column_name="positive",
            corpus=corpus,
            cross_encoder=teacher,
            range_min=p.range_min,
            range_max=p.range_max,
            max_score=p.max_score,
            min_score=p.min_score,
            absolute_margin=p.absolute_margin,
            relative_margin=p.relative_margin,
            num_negatives=p.num_negatives,
            sampling_strategy=p.sampling_strategy,
            output_format="n-tuple",  # query, positive, negative_1..n; rows short of n negatives are dropped
            output_scores=True,
            batch_size=p.batch_size,
            verbose=False,
            **prompt_names(model),
            **kwargs,
        )
        if len(mined) == 0:
            raise ValueError(
                "mining kept no row: no query had enough candidates after filtering; raise range_max, "
                "loosen relative_margin / absolute_margin / max_score, lower num_negatives, or add corpus documents"
            )
        metrics = {
            "mine/pairs_in": float(len(pairs)),
            "mine/rows_out": float(len(mined)),
            "mine/kept_fraction": len(mined) / len(pairs),
            "mine/candidates": float(len(corpus) if corpus else 0),
            **score_stats(mined[SCORES]),
        }
        if not p.output_scores:
            mined = mined.remove_columns(SCORES)
        out = ctx.path("train.jsonl")
        mined.to_json(str(out), force_ascii=False)
        stats = write_json(ctx.path("stats.json"), metrics)
        ctx.log.info("mine_negatives: %d of %d pairs kept with %d negatives each", len(mined), len(pairs), p.num_negatives)
        scored_by = f"teacher ({p.teacher_scores})" if teacher is not None else "miner similarity"
        return StageResult(artifacts={"train": out, "stats": stats}, metrics=metrics, notes={"scores": scored_by})
