"""`classifiers.prepare_data`: sources in, clean deterministic splits and a label map out."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, field_validator

from stagetrain.core import DataSource, Stage, StageContext, StageParams, StageResult, register, write_jsonl

from .labels import (
    LABELS_FILE,
    MULTI,
    LabelSpace,
    class_weights,
    dedupe,
    infer_names,
    label_counts,
    normalize_text,
    remove_leaks,
    split_labels,
    stratified_split,
    stratum,
    summarize,
)


def _as_list(v: Any) -> Any:
    if v is None:
        return []
    return [v] if isinstance(v, (dict, DataSource)) else v


class PrepareParams(StageParams):
    """Sources expose `text` and `label` (single-label) or `labels` (multi-label: a list, or a
    string split on `label_separator`); map other names with each source's `columns`. An optional
    `text_pair` column makes sentence-pair tasks."""

    train: list[DataSource]
    eval: list[DataSource] = Field(default_factory=list)  # model selection and calibration
    test: list[DataSource] = Field(default_factory=list)  # the final, untouched measurement
    unlabeled: list[DataSource] = Field(default_factory=list)  # `text` only: MLM adaptation, pseudo-labelling
    problem_type: Literal["single_label", "multi_label"] = "single_label"
    labels: list[str] | None = None  # fixes the classes and their order; inferred from train when None
    label_separator: str = ","
    eval_fraction: float = 0.1  # carved out of train, stratified, when no eval source is given
    test_fraction: float = 0.0  # likewise for test
    drop_conflicts: bool = True  # drop texts that carry two different labels
    min_chars: int = 1
    max_class_weight: float = 10.0

    @field_validator("train", "eval", "test", "unlabeled", mode="before")
    @classmethod
    def _sources(cls, v: Any) -> Any:
        return _as_list(v)


def _names_of(feature: Any) -> list[str] | None:
    """Class names of a `ClassLabel` column, or of a list of them."""
    inner = getattr(feature, "feature", feature)
    names = getattr(inner, "names", None)
    return list(names) if names else None


def read_labeled(src: DataSource, multi: bool, separator: str) -> list[dict[str, Any]]:
    """A source's rows as `{text, [text_pair], raw}`, `raw` being label names (`ClassLabel` ids are
    turned into their names)."""
    from stagetrain.core import load_source

    ds = load_source(src)
    column = "labels" if multi else "label"
    for need in ("text", column):
        if need not in ds.column_names:
            raise KeyError(
                f"{src.path}: no column {need!r}; columns: {ds.column_names}. "
                f"Map it with `columns: {{{need}: <name in the source>}}`"
            )
    names = _names_of(ds.features[column])
    has_pair = "text_pair" in ds.column_names
    rows = []
    for ex in ds:
        raw = split_labels(ex[column], separator) if multi else [ex[column]]
        if not multi and raw[0] is None:
            continue
        if names:
            raw = [names[r] if isinstance(r, int) else r for r in raw]
        row = {"text": normalize_text(ex["text"]), "raw": raw}
        if has_pair and ex["text_pair"]:
            row["text_pair"] = normalize_text(ex["text_pair"])
        rows.append(row)
    return rows


def read_unlabeled(src: DataSource) -> list[dict[str, Any]]:
    from stagetrain.core import load_source

    ds = load_source(src)
    if "text" not in ds.column_names:
        raise KeyError(f"{src.path}: no column 'text'; columns: {ds.column_names}")
    return [{"text": normalize_text(t)} for t in ds["text"]]


def encode_labels(rows: list[dict[str, Any]], space: LabelSpace, where: str) -> list[dict[str, Any]]:
    """Raw names as class indices: `label` for single-label rows, sorted `labels` for multi-label."""
    out = []
    for row in rows:
        try:
            ids = [space.index(r) for r in row["raw"]]
        except KeyError as e:
            raise KeyError(f"{where}: {e.args[0]}") from e
        new = {k: v for k, v in row.items() if k != "raw"}
        if space.multi:
            new["labels"] = sorted(set(ids))
        else:
            new["label"] = ids[0]
        out.append(new)
    return out


@register("classifiers.prepare_data")
class PrepareData(Stage):
    """Build the label map and clean, leak-free, deterministic train/eval/test splits."""

    Params = PrepareParams

    def run(self, ctx: StageContext) -> StageResult:
        p: PrepareParams = self.params
        multi = p.problem_type == MULTI
        raw = {
            split: [r for src in getattr(p, split) for r in read_labeled(src, multi, p.label_separator)]
            for split in ("train", "eval", "test")
        }
        if not raw["train"]:
            raise ValueError("the train sources hold no labelled row")
        names = p.labels or infer_names(x for row in raw["train"] for x in row["raw"])
        space = LabelSpace(list(names), p.problem_type)
        rows = {split: encode_labels(r, space, split) for split, r in raw.items()}

        stats = {"duplicates": 0, "conflicts": 0, "empty": 0, "leaked": 0, "short": 0}
        for split in rows:
            before = len(rows[split])
            rows[split] = [r for r in rows[split] if len(r["text"]) >= p.min_chars]
            stats["short"] += before - len(rows[split])
            rows[split], s = dedupe(rows[split], p.drop_conflicts)
            for k, v in s.items():
                stats[k] += v

        # Held-out splits win: a text in test leaves eval and train, a text in eval leaves train.
        rows["eval"], n = remove_leaks(rows["eval"], rows["test"])
        stats["leaked"] += n
        rows["train"], n = remove_leaks(rows["train"], [*rows["eval"], *rows["test"]])
        stats["leaked"] += n

        seed = self.seed(ctx)
        counts = label_counts(rows["train"], len(space))
        for split, fraction in (("test", p.test_fraction), ("eval", p.eval_fraction)):
            if not rows[split] and fraction > 0:
                # The fraction is of the original train set, whichever split is carved first.
                share = fraction / (1 - p.test_fraction) if split == "eval" and not p.test else fraction
                rows["train"], rows[split] = stratified_split(
                    rows["train"], min(share, 0.9), seed, lambda r: stratum(r, counts)
                )
        if not rows["train"]:
            raise ValueError("no train row is left after cleaning and splitting")

        space.counts = label_counts(rows["train"], len(space))
        space.class_weights = class_weights(space.counts, len(rows["train"]), p.problem_type, p.max_class_weight)

        artifacts: dict[str, Any] = {"labels": space.save(ctx.path(LABELS_FILE))}
        metrics: dict[str, float] = {"n_labels": float(len(space))}
        for split, split_rows in rows.items():
            metrics[f"n_{split}"] = float(len(split_rows))
            if split_rows:
                write_jsonl(ctx.path(f"{split}.jsonl"), split_rows)
                artifacts[split] = ctx.out_dir / f"{split}.jsonl"

        if p.unlabeled:
            pool = [r for src in p.unlabeled for r in read_unlabeled(src)]
            pool = [r for r in pool if len(r["text"]) >= p.min_chars]
            pool, _ = dedupe(pool)
            pool, n = remove_leaks(pool, [*rows["train"], *rows["eval"], *rows["test"]])
            stats["unlabeled_overlap"] = n
            write_jsonl(ctx.path("unlabeled.jsonl"), pool)
            artifacts["unlabeled"] = ctx.out_dir / "unlabeled.jsonl"
            metrics["n_unlabeled"] = float(len(pool))

        metrics.update({f"removed/{k}": float(v) for k, v in stats.items()})
        metrics.update(summarize(space.counts))
        ctx.log.info("prepare_data: %s", {k: int(v) for k, v in metrics.items() if k.startswith("n_")})
        return StageResult(
            artifacts=artifacts, metrics=metrics,
            notes={"labels": space.names, "train_counts": dict(zip(space.names, space.counts, strict=True))},
        )
