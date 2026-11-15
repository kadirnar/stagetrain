"""Hard-negative mining: for every (query, positive), documents the current model ranks high that
are not the positive, filtered so that likely false negatives stay out."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from ...core.data import DataSource, load_source, read_jsonl, write_json
from ...core.stage import Stage, StageContext, StageParams, StageResult, register
from .model import EncoderSpec, load_encoder, prompt_names
from .train import SCORES


class MineParams(StageParams):
    model: str  # the miner, usually `${pretrain.model}`
    train_data: str  # jsonl with query, positive
    corpus: str | None = None  # jsonl with `text`: more candidates than the positives of train_data
    num_negatives: int = Field(default=4, ge=1)
    # Rank window: skipping the first ranks avoids the near-duplicates and unlabeled positives that
    # sit right next to the positive.
    range_min: int = Field(default=10, ge=0)
    range_max: int | None = 100
    # False-negative filters. A candidate is kept only if its score is below
    # positive_score - |positive_score| * relative_margin, below positive_score - absolute_margin,
    # and below max_score.
    relative_margin: float | None = 0.05
    absolute_margin: float | None = None
    max_score: float | None = None
    min_score: float | None = None
    sampling_strategy: Literal["top", "random"] = "top"
    cross_encoder: str | None = None  # a teacher that rescores candidates; its scores drive the filters
    output_scores: bool = False  # keep [positive, negative_1..n] scores, for distillation
    batch_size: int = 64
    use_faiss: bool = False
    max_samples: int | None = None
    encoder: EncoderSpec = Field(default_factory=EncoderSpec)

    @model_validator(mode="after")
    def _window(self) -> MineParams:
        if self.range_max is not None and self.range_max - self.range_min < self.num_negatives:
            raise ValueError(
                f"the rank window [{self.range_min}, {self.range_max}) is smaller than num_negatives={self.num_negatives}"
            )
        return self


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
    """Mine hard negatives with a trained model over the corpus, with false-negative filtering; writes n-tuples."""

    Params = MineParams

    def run(self, ctx: StageContext) -> StageResult:
        from sentence_transformers import CrossEncoder, mine_hard_negatives

        p: MineParams = self.params
        model = load_encoder(p.model, p.encoder)
        pairs = load_source(DataSource(path=p.train_data, max_samples=p.max_samples)).select_columns(["query", "positive"])
        corpus = [r["text"] for r in read_jsonl(p.corpus)] if p.corpus else None
        teacher = CrossEncoder(p.cross_encoder, device=p.encoder.device) if p.cross_encoder else None

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
                "mining kept no row: no query had enough candidates after filtering; widen the rank window "
                "or loosen relative_margin / absolute_margin / max_score, or add corpus documents"
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
        return StageResult(artifacts={"train": out, "stats": stats}, metrics=metrics)
