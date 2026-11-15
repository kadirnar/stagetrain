"""Retrieval evaluation on a prepared eval set (queries, corpus, qrels).

`embeddings.eval` reports the usual retrieval metrics, the 95 % interval of nDCG@10, and each
query's own nDCG@10. With `compare_to` (another evaluation's per-query scores over the same
queries) it also reports the paired difference with its interval, so a gate can ask for "better
than the baseline beyond chance" instead of comparing two means whose difference is within noise.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import Field

from ...core.data import read_jsonl, write_json
from ...core.stage import Stage, StageContext, StageParams, StageResult, register
from .data import read_eval_set
from .model import EncoderSpec, load_encoder, prompt_names
from .ranking import bootstrap_mean, ndcg_at_k, paired_difference

EVALUATOR_NAME = "ir"  # in-training metrics are `eval_ir_<similarity>_ndcg@10`, etc.


def make_evaluator(
    model,
    eval_set: str | Path,
    batch_size: int = 64,
    truncate_dim: int | None = None,
    recall_at: list[int] | None = None,
    name: str = EVALUATOR_NAME,
    write_predictions: bool = False,
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
        write_predictions=write_predictions,
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


def read_rankings(predictions_dir: Path, evaluator) -> dict[str, list[str]]:
    """Each query's ranked document ids, from the predictions file the evaluator wrote for the
    similarity function it ranks by."""
    primary = evaluator.primary_metric  # "<name>_<similarity>_ndcg@10"
    similarity = primary[len(evaluator.name) + 1 : primary.rindex("_ndcg@")]
    path = predictions_dir / f"Information-Retrieval_evaluation_{evaluator.name}_predictions_{similarity}.jsonl"
    return {row["query_id"]: [hit["corpus_id"] for hit in row["results"]] for row in read_jsonl(path)}


def evaluate_model(
    model, eval_set: str | Path, dims: list[int] | None = None, batch_size: int = 64,
    recall_at: list[int] | None = None, predictions_dir: Path | None = None,
) -> tuple[dict[str, float], dict[str, float] | None]:
    """`(metrics, per-query nDCG@10)`: `eval/ndcg@10`, `eval/mrr@10`, `eval/recall@k`,
    `eval/accuracy@1`, `eval/map@100` at the model's full dimension and `eval/dim<d>/...` for
    each truncated dimension in `dims`. Per-query scores need `predictions_dir`, where the
    evaluator writes its rankings; without it the second value is None."""
    evaluator = make_evaluator(model, eval_set, batch_size, None, recall_at, write_predictions=predictions_dir is not None)
    raw = evaluator(model, output_path=str(predictions_dir)) if predictions_dir is not None else evaluator(model)
    metrics = stage_metrics(raw, evaluator)
    per_query = None
    if predictions_dir is not None:
        _, _, qrels = read_eval_set(eval_set)
        rankings = read_rankings(predictions_dir, evaluator)
        per_query = {q: ndcg_at_k(rankings.get(q, ()), relevant) for q, relevant in qrels.items()}
    full = model.get_embedding_dimension()
    for dim in dims or []:
        if full is not None and dim >= full:
            continue
        evaluator = make_evaluator(model, eval_set, batch_size, dim, recall_at)
        metrics.update(stage_metrics(evaluator(model), evaluator, prefix=f"eval/dim{dim}/"))
    return metrics, per_query


def retrieval_metrics(
    model, eval_set: str | Path, dims: list[int] | None = None, batch_size: int = 64, recall_at: list[int] | None = None
) -> dict[str, float]:
    return evaluate_model(model, eval_set, dims, batch_size, recall_at)[0]


def score_statistics(per_query: dict[str, float], compare_to: str | None = None, seed: int = 0) -> dict[str, float]:
    """The interval of nDCG@10 over queries and, with `compare_to`, the paired difference to
    another evaluation of the same queries."""
    stats = bootstrap_mean(list(per_query.values()), seed=seed)
    metrics = {"eval/n": stats["n"], "eval/ndcg@10_ci_low": stats["ci_low"], "eval/ndcg@10_ci_high": stats["ci_high"]}
    if compare_to:
        baseline = json.loads(Path(compare_to).read_text(encoding="utf-8"))
        diff = paired_difference(per_query, baseline, seed=seed)
        if not diff["n"]:
            raise ValueError(f"{compare_to} shares no query with this evaluation; compare runs over the same eval set")
        metrics.update({
            "eval/delta_ndcg@10": diff["delta"], "eval/delta_ci_low": diff["ci_low"],
            "eval/delta_ci_high": diff["ci_high"], "eval/delta_n": diff["n"],
        })
    return metrics


def nanobeir_metrics(model, datasets: list[str] | None, batch_size: int) -> dict[str, float]:
    """Mean nDCG@10 over NanoBEIR: a regression alarm for general retrieval ability, not a score.
    The evaluator downloads its datasets on first use."""
    from sentence_transformers.sentence_transformer.evaluation import NanoBEIREvaluator

    evaluator = NanoBEIREvaluator(dataset_names=datasets, batch_size=batch_size, write_csv=False, show_progress_bar=False)
    raw = evaluator(model)
    return {"eval/nanobeir_ndcg@10": float(raw[evaluator.primary_metric])}


class EvalParams(StageParams):
    model: str
    eval_set: str  # ${data.eval_set} to choose on, ${data.test_set} to report on
    dims: list[int] = Field(default_factory=list)  # Matryoshka dimensions to evaluate as well
    recall_at: list[int] = Field(default_factory=lambda: [1, 10, 100])
    batch_size: int = 64
    compare_to: str | None = None  # the `scores` artifact of another eval (or bm25) stage over the same eval set
    nanobeir: bool = False  # also measure general retrieval ability; downloads the NanoBEIR datasets
    nanobeir_datasets: list[str] | None = None  # None: all of them
    encoder: EncoderSpec = Field(default_factory=EncoderSpec)


@register("embeddings.eval")
class Eval(Stage):
    """Retrieval metrics with their interval, per-query scores, and a paired comparison to another evaluation."""

    Params = EvalParams

    def run(self, ctx: StageContext) -> StageResult:
        p: EvalParams = self.params
        model = load_encoder(p.model, p.encoder)
        metrics: dict[str, Any]
        metrics, per_query = evaluate_model(
            model, p.eval_set, p.dims, p.batch_size, p.recall_at, predictions_dir=ctx.out_dir / "predictions"
        )
        metrics.update(score_statistics(per_query, p.compare_to, self.seed(ctx)))
        metrics["eval/dimension"] = float(model.get_embedding_dimension() or 0)
        if p.nanobeir:
            metrics.update(nanobeir_metrics(model, p.nanobeir_datasets, p.batch_size))
        artifacts = {
            "metrics": write_json(ctx.path("metrics.json"), metrics),
            "scores": write_json(ctx.path("scores.json"), per_query),
        }
        return StageResult(artifacts=artifacts, metrics=metrics)
