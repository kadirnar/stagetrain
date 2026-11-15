"""Per-query retrieval metrics and the statistics that say whether a difference is real.

Plain Python: a ranking is the list of document ids a system returned for one query, best first,
and `relevant` is the set of ids labelled relevant. The same functions score a dense model's
predictions and the BM25 baseline, so the two are compared on equal terms.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence

RESAMPLES = 2000


def ndcg_at_k(ranked: Sequence[str], relevant: set[str], k: int = 10) -> float:
    """nDCG@k with binary relevance: the gain of each relevant document falls with the logarithm
    of its rank, relative to the best ordering possible for this query."""
    if not relevant:
        return 0.0
    dcg = sum(1.0 / math.log2(rank + 2) for rank, doc in enumerate(ranked[:k]) if doc in relevant)
    ideal = sum(1.0 / math.log2(rank + 2) for rank in range(min(k, len(relevant))))
    return dcg / ideal


def mrr_at_k(ranked: Sequence[str], relevant: set[str], k: int = 10) -> float:
    for rank, doc in enumerate(ranked[:k]):
        if doc in relevant:
            return 1.0 / (rank + 1)
    return 0.0


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return sum(1 for doc in ranked[:k] if doc in relevant) / len(relevant)


def summarize_rankings(
    rankings: Mapping[str, Sequence[str]], qrels: Mapping[str, set[str]], recall_at: Sequence[int] = (1, 10, 100)
) -> tuple[dict[str, float], dict[str, float]]:
    """`(metrics, per-query nDCG@10)` over the queries in `qrels`; a query with no ranking scores 0."""
    per_query = {q: ndcg_at_k(rankings.get(q, ()), rel) for q, rel in qrels.items()}
    n = max(len(qrels), 1)
    metrics = {
        "ndcg@10": sum(per_query.values()) / n,
        "mrr@10": sum(mrr_at_k(rankings.get(q, ()), rel) for q, rel in qrels.items()) / n,
        "accuracy@1": sum(recall_at_k(rankings.get(q, ()), rel, 1) > 0 for q, rel in qrels.items()) / n,
    }
    for k in sorted(set(recall_at)):
        metrics[f"recall@{k}"] = sum(recall_at_k(rankings.get(q, ()), rel, k) for q, rel in qrels.items()) / n
    return metrics, per_query


def bootstrap_mean(values: Sequence[float], resamples: int = RESAMPLES, seed: int = 0) -> dict[str, float]:
    """The mean of `values` with a 95 % percentile-bootstrap interval over queries."""
    n = len(values)
    if n == 0:
        return {"n": 0.0}
    mean = sum(values) / n
    if n == 1:
        return {"n": 1.0, "mean": mean, "ci_low": mean, "ci_high": mean}
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(resamples))
    return {
        "n": float(n), "mean": mean,
        "ci_low": means[int(0.025 * resamples)], "ci_high": means[min(int(0.975 * resamples), resamples - 1)],
    }


def paired_difference(
    scores: Mapping[str, float], baseline: Mapping[str, float], resamples: int = RESAMPLES, seed: int = 0
) -> dict[str, float]:
    """This system minus a baseline, query by query, over the queries both were scored on.

    Two systems answering the same queries make correlated errors, so the per-query difference
    has a much tighter interval than two separate means would: a few hundred queries can tell
    which system is better. `ci_low > 0` means better at the 95 % level.
    """
    shared = sorted(set(scores) & set(baseline))
    if not shared:
        return {"n": 0.0}
    diffs = [scores[q] - baseline[q] for q in shared]
    stats = bootstrap_mean(diffs, resamples, seed)
    return {
        "n": stats["n"], "delta": stats["mean"], "ci_low": stats["ci_low"], "ci_high": stats["ci_high"],
        "wins": float(sum(d > 0 for d in diffs)), "losses": float(sum(d < 0 for d in diffs)),
    }
