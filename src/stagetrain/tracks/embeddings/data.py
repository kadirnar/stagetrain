"""Pair data for a bi-encoder: cleaning, a leak-free split, and a retrieval eval set."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import Field

from ...core.data import DataSource, load_source, read_jsonl, write_json, write_jsonl
from ...core.stage import Stage, StageContext, StageParams, StageResult, register

_SPACE = re.compile(r"\s+")


def normalize_text(text: Any, collapse_whitespace: bool = True) -> str:
    if text is None:
        return ""
    text = str(text).strip()
    return _SPACE.sub(" ", text) if collapse_whitespace else text


def text_key(text: str) -> str:
    """What makes two texts "the same" for deduplication and for the split."""
    return _SPACE.sub(" ", text.strip().lower())


def clean_pairs(
    rows: Iterable[dict[str, Any]],
    min_chars: int = 1,
    max_chars: int | None = None,
    collapse_whitespace: bool = True,
    drop_identical: bool = True,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    """`{query, positive[, negative]}` rows, normalised and deduplicated on (query, positive).

    Dropped: rows with an empty or too short / too long query or positive, rows whose query and
    positive are the same text, and repeated pairs. A negative that is empty or equals the
    positive is removed from its row; the row is kept.
    """
    stats = {"read": 0, "dropped_empty": 0, "dropped_length": 0, "dropped_identical": 0, "dropped_duplicates": 0}
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, str]] = []
    for row in rows:
        stats["read"] += 1
        query = normalize_text(row.get("query"), collapse_whitespace)
        positive = normalize_text(row.get("positive"), collapse_whitespace)
        if not query or not positive:
            stats["dropped_empty"] += 1
            continue
        too_short = len(query) < min_chars or len(positive) < min_chars
        too_long = max_chars is not None and (len(query) > max_chars or len(positive) > max_chars)
        if too_short or too_long:
            stats["dropped_length"] += 1
            continue
        key = (text_key(query), text_key(positive))
        if drop_identical and key[0] == key[1]:
            stats["dropped_identical"] += 1
            continue
        if key in seen:
            stats["dropped_duplicates"] += 1
            continue
        seen.add(key)
        clean = {"query": query, "positive": positive}
        negative = normalize_text(row.get("negative"), collapse_whitespace)
        if negative and text_key(negative) != key[1]:
            clean["negative"] = negative
        out.append(clean)
    return out, stats


def query_bucket(query: str, seed: int) -> float:
    """A number in [0, 1) that depends only on the query text and the seed."""
    h = hashlib.sha1(f"{seed}:{text_key(query)}".encode()).hexdigest()
    return int(h[:12], 16) / 16**12


def split_by_query(
    rows: list[dict[str, str]], eval_fraction: float, seed: int, max_eval_queries: int | None = None
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """`(train, eval)`. Every pair of one query lands on the same side, so no eval query is seen in
    training. The side depends on a hash of the query, not on row order, so the split is the same
    when rows are added or reordered."""
    if eval_fraction <= 0:
        return list(rows), []
    buckets = {text_key(r["query"]): query_bucket(r["query"], seed) for r in rows}
    chosen = sorted((b, q) for q, b in buckets.items() if b < eval_fraction)
    if max_eval_queries is not None:
        chosen = chosen[:max_eval_queries]
    eval_queries = {q for _, q in chosen}
    train = [r for r in rows if text_key(r["query"]) not in eval_queries]
    held = [r for r in rows if text_key(r["query"]) in eval_queries]
    return train, held


def unique_texts(texts: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for t in texts:
        key = text_key(t)
        if key and key not in seen:
            seen.add(key)
            out.append(t)
    return out


def documents(rows: Iterable[dict[str, str]]) -> list[str]:
    """Every document (positive or negative) in `rows`, first occurrence first."""
    out = []
    for r in rows:
        out.append(r["positive"])
        if r.get("negative"):
            out.append(r["negative"])
    return unique_texts(out)


def build_eval_set(
    eval_rows: list[dict[str, str]], distractors: Iterable[str] = (), max_distractors: int | None = None
) -> tuple[dict[str, str], dict[str, str], dict[str, list[str]]]:
    """`(queries, corpus, qrels)` for retrieval evaluation.

    A query that occurs with several positives gets all of them as relevant. The corpus holds the
    eval rows' documents, then up to `max_distractors` other documents (typically the training
    corpus) that make retrieval harder; a distractor that equals an eval document is not repeated.
    """
    corpus_ids: dict[str, str] = {}  # text key -> doc id
    corpus: dict[str, str] = {}

    def doc_id(text: str) -> str:
        key = text_key(text)
        if key not in corpus_ids:
            corpus_ids[key] = f"d{len(corpus_ids)}"
            corpus[corpus_ids[key]] = text
        return corpus_ids[key]

    query_ids: dict[str, str] = {}
    queries: dict[str, str] = {}
    qrels: dict[str, list[str]] = {}
    for r in eval_rows:
        key = text_key(r["query"])
        if key not in query_ids:
            query_ids[key] = f"q{len(query_ids)}"
            queries[query_ids[key]] = r["query"]
            qrels[query_ids[key]] = []
        did = doc_id(r["positive"])
        if did not in qrels[query_ids[key]]:
            qrels[query_ids[key]].append(did)
    for r in eval_rows:
        if r.get("negative"):
            doc_id(r["negative"])
    added = 0
    for text in distractors:
        if max_distractors is not None and added >= max_distractors:
            break
        if text_key(text) not in corpus_ids:
            doc_id(text)
            added += 1
    return queries, corpus, qrels


def write_eval_set(eval_dir: Path, queries: dict[str, str], corpus: dict[str, str], qrels: dict[str, list[str]]) -> Path:
    write_jsonl(eval_dir / "queries.jsonl", ({"id": k, "text": v} for k, v in queries.items()))
    write_jsonl(eval_dir / "corpus.jsonl", ({"id": k, "text": v} for k, v in corpus.items()))
    write_jsonl(eval_dir / "qrels.jsonl", ({"query_id": k, "doc_ids": v} for k, v in qrels.items()))
    return eval_dir


def read_eval_set(eval_dir: str | Path) -> tuple[dict[str, str], dict[str, str], dict[str, set[str]]]:
    eval_dir = Path(eval_dir)
    for name in ("queries.jsonl", "corpus.jsonl", "qrels.jsonl"):
        if not (eval_dir / name).is_file():
            raise FileNotFoundError(f"{eval_dir}: not an eval set, {name} is missing")
    queries = {r["id"]: r["text"] for r in read_jsonl(eval_dir / "queries.jsonl")}
    corpus = {r["id"]: r["text"] for r in read_jsonl(eval_dir / "corpus.jsonl")}
    qrels = {r["query_id"]: set(r["doc_ids"]) for r in read_jsonl(eval_dir / "qrels.jsonl")}
    return queries, corpus, qrels


def source_rows(src: DataSource) -> Iterable[dict[str, Any]]:
    ds = load_source(src)
    for col in ("query", "positive"):
        if col not in ds.column_names:
            raise KeyError(
                f"{src.path}: no {col!r} column; map it with `columns: {{{col}: <name>}}`. Columns: {ds.column_names}"
            )
    keep = [c for c in ("query", "positive", "negative") if c in ds.column_names]
    return ds.select_columns(keep)


class PrepareDataParams(StageParams):
    sources: list[DataSource]  # columns: query, positive, optionally negative
    eval_sources: list[DataSource] = Field(default_factory=list)  # a ready dev split; replaces eval_fraction
    corpus_sources: list[DataSource] = Field(default_factory=list)  # extra documents, column `text`
    eval_fraction: float = Field(default=0.0, ge=0.0, lt=1.0)
    max_eval_queries: int | None = 2000
    max_eval_distractors: int | None = 100_000  # training documents added to the eval corpus
    min_chars: int = 1
    max_chars: int | None = None
    collapse_whitespace: bool = True
    drop_identical: bool = True


@register("embeddings.prepare_data")
class PrepareData(Stage):
    """Clean and deduplicate (query, positive[, negative]) pairs, split off eval queries, build a retrieval eval set."""

    Params = PrepareDataParams

    def run(self, ctx: StageContext) -> StageResult:
        p: PrepareDataParams = self.params
        clean = {
            "min_chars": p.min_chars, "max_chars": p.max_chars, "collapse_whitespace": p.collapse_whitespace,
            "drop_identical": p.drop_identical,
        }
        rows, stats = clean_pairs((r for s in p.sources for r in source_rows(s)), **clean)
        if not rows:
            raise ValueError(f"no usable pairs in the sources; {stats}")

        if p.eval_sources:
            held, _ = clean_pairs((r for s in p.eval_sources for r in source_rows(s)), **clean)
            eval_queries = {text_key(r["query"]) for r in held}
            train = [r for r in rows if text_key(r["query"]) not in eval_queries]
            stats["dropped_leaked"] = len(rows) - len(train)  # training pairs whose query is an eval query
            if p.max_eval_queries is not None:
                _, held = split_by_query(held, 1.0, self.seed(ctx), p.max_eval_queries)
        else:
            train, held = split_by_query(rows, p.eval_fraction, self.seed(ctx), p.max_eval_queries)
        if not train:
            raise ValueError("the split left no training pairs")

        # Candidates for hard-negative mining: training documents only, so no eval document leaks in.
        mining_docs = documents(train)

        # A negatives column must be complete for a loss to read it; a partial one is dropped.
        with_negative = sum(1 for r in train if "negative" in r)
        keep_negatives = with_negative == len(train)
        if not keep_negatives:
            train = [{"query": r["query"], "positive": r["positive"]} for r in train]
        stats["negatives_dropped"] = 0 if keep_negatives else with_negative

        extra = []
        for s in p.corpus_sources:
            ds = load_source(s)
            if "text" not in ds.column_names:
                raise KeyError(f"{s.path}: a corpus source needs a `text` column; columns: {ds.column_names}")
            extra.extend(normalize_text(t, p.collapse_whitespace) for t in ds["text"])
        corpus = unique_texts([*mining_docs, *extra])

        artifacts: dict[str, Any] = {
            "train": ctx.path("train.jsonl"),
            "corpus": ctx.path("corpus.jsonl"),
        }
        write_jsonl(artifacts["train"], train)
        write_jsonl(artifacts["corpus"], ({"text": t} for t in corpus))
        metrics = {
            "data/train_pairs": len(train),
            "data/train_queries": len({text_key(r["query"]) for r in train}),
            "data/corpus": len(corpus),
            "data/has_negatives": float(keep_negatives),
            "data/eval_pairs": len(held),
            **{f"data/{k}": v for k, v in stats.items()},
        }
        if held:
            queries, eval_corpus, qrels = build_eval_set(held, corpus, p.max_eval_distractors)
            artifacts["eval_pairs"] = ctx.path("eval_pairs.jsonl")
            write_jsonl(artifacts["eval_pairs"], ({"query": r["query"], "positive": r["positive"]} for r in held))
            artifacts["eval_set"] = write_eval_set(ctx.path("eval_set", "queries.jsonl").parent, queries, eval_corpus, qrels)
            metrics["data/eval_queries"] = len(queries)
            metrics["data/eval_corpus"] = len(eval_corpus)
        artifacts["stats"] = write_json(ctx.path("stats.json"), metrics)
        ctx.log.info("prepare_data: %d train pairs, %d eval pairs, %d corpus documents", len(train), len(held), len(corpus))
        return StageResult(artifacts=artifacts, metrics=metrics)
