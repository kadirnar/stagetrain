"""`embeddings.bm25`: the lexical baseline, on the same eval set and with the same metrics.

A dense model is worth deploying over keyword search only where it beats it, and on some corpora
it does not. BM25 needs no training and no model, so it is the first number a pipeline measures.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping
from typing import Any

from pydantic import Field

from ...core.data import write_json
from ...core.stage import Stage, StageContext, StageParams, StageResult, register
from .data import read_eval_set
from .evaluate import score_statistics
from .ranking import summarize_rankings

TOKEN = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


class BM25:
    """Okapi BM25 over a small or medium corpus, with an inverted index held in numpy arrays."""

    def __init__(self, corpus: Mapping[str, str], k1: float = 1.2, b: float = 0.75):
        import numpy as np

        self.ids = list(corpus)
        self.k1, self.b = k1, b
        lengths = np.zeros(len(self.ids), dtype=np.float32)
        postings: dict[str, tuple[list[int], list[int]]] = {}
        for index, doc_id in enumerate(self.ids):
            counts = Counter(tokenize(corpus[doc_id]))
            lengths[index] = sum(counts.values())
            for term, tf in counts.items():
                docs, tfs = postings.setdefault(term, ([], []))
                docs.append(index)
                tfs.append(tf)
        n = len(self.ids)
        mean_length = float(lengths.mean()) if n else 0.0
        self.norm = k1 * (1 - b + b * lengths / mean_length) if mean_length else np.full(n, k1, dtype=np.float32)
        self.index = {}
        for term, (docs, tfs) in postings.items():
            idf = math.log(1 + (n - len(docs) + 0.5) / (len(docs) + 0.5))
            self.index[term] = (np.asarray(docs, dtype=np.int64), np.asarray(tfs, dtype=np.float32), idf)

    def search(self, query: str, k: int = 100) -> list[str]:
        """The ids of the `k` best documents for `query`, best first; ties go to the earlier document."""
        import numpy as np

        scores = np.zeros(len(self.ids), dtype=np.float32)
        for term, weight in Counter(tokenize(query)).items():
            entry = self.index.get(term)
            if entry is None:
                continue
            docs, tfs, idf = entry
            scores[docs] += weight * idf * tfs * (self.k1 + 1) / (tfs + self.norm[docs])
        k = min(k, len(self.ids))
        if k == 0:
            return []
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.lexsort((top, -scores[top]))]
        return [self.ids[i] for i in top if scores[i] > 0]


class Bm25Params(StageParams):
    eval_set: str
    recall_at: list[int] = Field(default_factory=lambda: [1, 10, 100])
    k1: float = 1.2
    b: float = Field(default=0.75, ge=0.0, le=1.0)


@register("embeddings.bm25")
class Bm25Baseline(Stage):
    """BM25 on an eval set: the lexical baseline every dense model is compared with."""

    Params = Bm25Params

    def run(self, ctx: StageContext) -> StageResult:
        p: Bm25Params = self.params
        queries, corpus, qrels = read_eval_set(p.eval_set)
        index = BM25(corpus, p.k1, p.b)
        depth = max([10, *p.recall_at])
        rankings = {query_id: index.search(text, depth) for query_id, text in queries.items()}
        summary, per_query = summarize_rankings(rankings, qrels, p.recall_at)
        metrics: dict[str, Any] = {f"eval/{name}": value for name, value in summary.items()}
        metrics.update(score_statistics(per_query, seed=self.seed(ctx)))
        metrics["bm25/corpus"] = float(len(corpus))
        artifacts = {
            "metrics": write_json(ctx.path("metrics.json"), metrics),
            "scores": write_json(ctx.path("scores.json"), per_query),
        }
        return StageResult(artifacts=artifacts, metrics=metrics)
