"""Retrieval evaluation on a prepared eval set (queries, corpus, qrels)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field

from ...core.data import write_json
from ...core.stage import Stage, StageContext, StageParams, StageResult, register
from .data import read_eval_set
from .model import EncoderSpec, load_encoder, prompt_names

EVALUATOR_NAME = "ir"  # in-training metrics are `eval_ir_<similarity>_ndcg@10`, etc.


def make_evaluator(
    model,
    eval_set: str | Path,
    batch_size: int = 64,
    truncate_dim: int | None = None,
    recall_at: list[int] | None = None,
    name: str = EVALUATOR_NAME,
):
    """The library's information-retrieval evaluator over an eval set, using the model's own
    query / document prompts when it has them."""
    from sentence_transformers.sentence_transformer.evaluation import InformationRetrievalEvaluator

    queries, corpus, qrels = read_eval_set(eval_set)
    return InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=qrels,
        ndcg_at_k=[10],
        mrr_at_k=[10],
        map_at_k=[100],
        accuracy_at_k=[1],
        precision_recall_at_k=sorted(set(recall_at or [1, 10, 100])),
        batch_size=batch_size,
        truncate_dim=truncate_dim,
        name=name,
        write_csv=False,
        show_progress_bar=False,
        **prompt_names(model),
    )


def stage_metrics(raw: dict[str, float], evaluator, prefix: str = "eval/") -> dict[str, float]:
    """The evaluator's `<name>_<similarity>_<metric>` results as `eval/<metric>`, for the
    similarity function the evaluator ranks by. Precision@k is left out."""
    primary = evaluator.primary_metric  # "<name>_<similarity>_ndcg@10"
    head = primary[: primary.rindex("ndcg@")]
    out = {}
    for key, value in raw.items():
        if key.startswith(head) and not key[len(head):].startswith("precision@"):
            out[f"{prefix}{key[len(head):]}"] = float(value)
    return out


def retrieval_metrics(
    model, eval_set: str | Path, dims: list[int] | None = None, batch_size: int = 64, recall_at: list[int] | None = None
) -> dict[str, float]:
    """`eval/ndcg@10`, `eval/mrr@10`, `eval/recall@k`, `eval/accuracy@1`, `eval/map@100` at the
    model's full dimension, and `eval/dim<d>/...` for each truncated dimension in `dims`."""
    evaluator = make_evaluator(model, eval_set, batch_size, None, recall_at)
    metrics = stage_metrics(evaluator(model), evaluator)
    full = model.get_embedding_dimension()
    for dim in dims or []:
        if full is not None and dim >= full:
            continue
        evaluator = make_evaluator(model, eval_set, batch_size, dim, recall_at)
        metrics.update(stage_metrics(evaluator(model), evaluator, prefix=f"eval/dim{dim}/"))
    return metrics


class EvalParams(StageParams):
    model: str
    eval_set: str
    dims: list[int] = Field(default_factory=list)  # Matryoshka dimensions to evaluate as well
    recall_at: list[int] = Field(default_factory=lambda: [1, 10, 100])
    batch_size: int = 64
    encoder: EncoderSpec = Field(default_factory=EncoderSpec)


@register("embeddings.eval")
class Eval(Stage):
    """Retrieval metrics (nDCG@10, MRR@10, Recall@k) of a model on a prepared eval set, per Matryoshka dimension."""

    Params = EvalParams

    def run(self, ctx: StageContext) -> StageResult:
        p: EvalParams = self.params
        model = load_encoder(p.model, p.encoder)
        metrics: dict[str, Any] = retrieval_metrics(model, p.eval_set, p.dims, p.batch_size, p.recall_at)
        metrics["eval/dimension"] = float(model.get_embedding_dimension() or 0)
        report = write_json(ctx.path("metrics.json"), metrics)
        return StageResult(artifacts={"metrics": report}, metrics=metrics)
