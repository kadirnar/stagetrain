"""Stages that use a trained classifier without updating it: pseudo-labelling, calibration,
evaluation, the label audit and export; and the TF-IDF baseline every classifier is measured against."""

from __future__ import annotations

import hashlib
import importlib.util
import shutil
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from stagetrain.core import (
    Stage,
    StageContext,
    StageParams,
    StageResult,
    group_runs,
    read_jsonl,
    register,
    write_json,
    write_jsonl,
)

from .labels import LABELS_FILE, LabelSpace, text_key
from .metrics import CALIBRATION_FILE

CONFIG_FILE = "classifier_config.json"


class InferenceParams(StageParams):
    model: str  # the `model` artifact of classifiers.finetune
    batch_size: int = 64
    max_length: int | None = None  # None: the length the model was trained with
    device: str | None = None  # default: cuda, then mps, then cpu
    trust_remote_code: bool = False


def _logits(p: InferenceParams, rows: list[dict[str, Any]], model: str | None = None):
    from .modeling import Predictor

    return Predictor(model or p.model, p.device, p.trust_remote_code).logits(rows, p.batch_size, p.max_length)


def row_id(row: dict[str, Any]) -> str:
    """Identifies a row across stages that read the same file."""
    return hashlib.sha1(text_key(row).encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# pseudo-labelling
# --------------------------------------------------------------------------- #


class PseudoLabelParams(InferenceParams):
    """One teacher (`model`) or an ensemble whose probabilities are averaged: `models`, or
    `group: <matrix name>` for runs of a matrix. With `group`, `ensemble: all` takes every run and
    `ensemble: winner` only the runs of the configuration a `classifiers.select` stage chose; give
    that stage's model as `model` so the two are tied together."""

    model: str | None = None
    models: list[str] = Field(default_factory=list)
    group: str | None = None
    ensemble: Literal["all", "winner"] = "all"
    data: str  # JSONL with `text`: the unlabeled pool, or labelled rows to soften (their labels are ignored)
    calibration: str | None = None  # calibrate a single teacher before thresholding its confidence
    min_confidence: float = 0.9  # single-label: probability of the predicted class
    multi_label_margin: float = 0.4  # multi-label: every label's probability is this far from 0.5
    max_per_class: int | None = None  # keep the most confident rows of each predicted class
    max_samples: int | None = None  # how many pool rows to read
    hard_labels: bool = False  # write the teacher's decision instead of its distribution
    keep_all: bool = False  # distillation: keep every row with the teacher's distribution, whatever its confidence

    @model_validator(mode="after")
    def _teachers(self) -> PseudoLabelParams:
        if self.group is None and not self.models and self.model is None:
            raise ValueError("pseudo_label needs a teacher: model, models, or group")
        if self.models and (self.group or self.model):
            raise ValueError("give models, or group, or a single model")
        if self.group is None and self.ensemble == "winner":
            raise ValueError("ensemble: winner needs group (the matrix) and model (the selecting stage's model)")
        if self.group and self.ensemble == "winner" and self.model is None:
            raise ValueError("ensemble: winner needs model: the `model` of the stage that selected among the group")
        if self.calibration and (self.models or self.group):
            raise ValueError("calibration applies to a single teacher; an ensemble's average is used as it is")
        return self


def teacher_models(p: PseudoLabelParams, ctx: StageContext) -> list[str]:
    """The model directories whose probabilities are averaged."""
    if p.models:
        return list(p.models)
    if p.group is None:
        return [p.model]
    runs = group_runs(ctx.upstream, p.group)
    if not runs:
        raise ValueError(f"pseudo_label: {p.group!r} is not a matrix this stage depends on")
    names = list(runs)
    if p.ensemble == "winner":
        chosen = [m.notes.get("winner_runs") for m in ctx.upstream.values()
                  if m.notes.get("winner_runs") and m.artifacts.get("model") and m.artifact("model", ctx.run_dir) == p.model]
        if not chosen:
            raise ValueError(
                "pseudo_label: ensemble: winner found no selecting stage whose model is the one given; "
                "set model to ${<select stage>.model}"
            )
        names = [n for n in chosen[0] if n in runs]
    return [runs[n].artifact("model", ctx.run_dir) for n in names]


def select_confident(
    probs, problem_type: str, min_confidence: float, margin: float, max_per_class: int | None = None
) -> list[tuple[int, float]]:
    """`(row index, confidence)` of the rows a teacher is sure about, most confident first within
    each predicted class, at most `max_per_class` per class (multi-label rows are not capped)."""
    import numpy as np

    from .labels import MULTI

    p = np.asarray(probs)
    if problem_type == MULTI:
        conf = np.abs(p - 0.5).min(axis=-1)
        keep = np.flatnonzero(conf >= margin)
        order = keep[np.argsort(-conf[keep], kind="stable")]
        return [(int(i), float(conf[i] + 0.5)) for i in order]
    conf, cls = p.max(axis=-1), p.argmax(axis=-1)
    out: list[tuple[int, float]] = []
    for c in range(p.shape[1]):
        idx = np.flatnonzero((cls == c) & (conf >= min_confidence))
        idx = idx[np.argsort(-conf[idx], kind="stable")]
        out += [(int(i), float(conf[i])) for i in (idx[:max_per_class] if max_per_class else idx)]
    return out


@register("classifiers.pseudo_label")
class PseudoLabel(Stage):
    """Label unlabeled text with a trained teacher and keep what it is confident about."""

    Params = PseudoLabelParams

    def run(self, ctx: StageContext) -> StageResult:
        from .metrics import load_calibration, probabilities

        p: PseudoLabelParams = self.params
        teachers = teacher_models(p, ctx)
        labels = LabelSpace.load(teachers[0])
        rows = []
        for row in read_jsonl(p.data):
            rows.append(row)
            if p.max_samples and len(rows) >= p.max_samples:
                break
        if not rows:
            raise ValueError(f"{p.data}: the unlabeled pool is empty")
        calibration = load_calibration(p.calibration)
        # Several teachers: the mean of their distributions. Seeds of one setting agree on the
        # aggregate and differ on individual rows, which is what the average smooths out.
        probs = sum(
            probabilities(_logits(p, rows, t), labels.problem_type, calibration["temperature"]) for t in teachers
        ) / len(teachers)
        if p.keep_all:
            confidence = probs.max(axis=-1) if not labels.multi else abs(probs - 0.5).min(axis=-1) + 0.5
            kept = [(i, float(confidence[i])) for i in range(len(rows))]
        else:
            kept = select_confident(probs, labels.problem_type, p.min_confidence, p.multi_label_margin, p.max_per_class)

        out, per_class = [], [0] * len(labels)
        for i, confidence in kept:
            row: dict[str, Any] = {"text": rows[i]["text"], "source": "pseudo", "confidence": round(confidence, 6)}
            if rows[i].get("text_pair"):
                row["text_pair"] = rows[i]["text_pair"]
            if labels.multi:
                row["labels"] = [c for c in range(len(labels)) if probs[i][c] >= 0.5]
                for c in row["labels"]:
                    per_class[c] += 1
            else:
                row["label"] = int(probs[i].argmax())
                per_class[row["label"]] += 1
            if not p.hard_labels:
                row["probs"] = [round(float(x), 6) for x in probs[i]]
            out.append(row)
        path = ctx.path("pseudo.jsonl")
        write_jsonl(path, out)
        metrics = {
            "pseudo/candidates": float(len(rows)),
            "pseudo/kept": float(len(out)),
            "pseudo/keep_rate": len(out) / len(rows),
            "pseudo/mean_confidence": sum(c for _, c in kept) / len(kept) if kept else 0.0,
            "pseudo/classes_covered": float(sum(1 for c in per_class if c > 0)),
            "pseudo/teachers": float(len(teachers)),
        }
        return StageResult(
            artifacts={"data": path}, metrics=metrics,
            notes={"kept_per_class": dict(zip(labels.names, per_class, strict=True)), "teachers": teachers},
        )


# --------------------------------------------------------------------------- #
# calibration
# --------------------------------------------------------------------------- #


class CalibrateParams(InferenceParams):
    # A held-out labelled split the model was not trained on. Prefer prepare_data's `calibration`
    # split over `eval`: the split that picked the best checkpoint flatters that checkpoint.
    data: str
    n_bins: int = 15
    fit_thresholds: bool = True  # multi-label only: a decision threshold per label
    # A label with fewer positives in `data` than this shares one threshold with the other rare
    # labels instead of getting its own: a cut fitted on a handful of positives is noise.
    min_positives: int = Field(default=10, ge=1)
    min_temperature: float = 0.05
    max_temperature: float = 20.0


@register("classifiers.calibrate")
class Calibrate(Stage):
    """Fit a temperature (and multi-label thresholds) on held-out data; the weights do not change."""

    Params = CalibrateParams

    def run(self, ctx: StageContext) -> StageResult:
        from .metrics import ece, fit_temperature, fit_thresholds, nll, probabilities
        from .modeling import gold_targets
        from .train import read_rows

        p: CalibrateParams = self.params
        labels = LabelSpace.load(p.model)
        rows = read_rows(p.data, labels)
        if not rows:
            raise ValueError(f"{p.data}: no row to calibrate on")
        logits, y = _logits(p, rows), gold_targets(rows, labels)
        kind = labels.problem_type
        temperature = fit_temperature(logits, y, kind, p.min_temperature, p.max_temperature)
        # A split the model gets entirely right (or wrong) has no finite optimum: the fit runs
        # into a bound and the temperature says nothing about new data.
        at_bound = min(temperature / p.min_temperature, p.max_temperature / temperature) < 1.01
        if at_bound:
            ctx.log.warning("calibrate: temperature %.3g is at its bound; the split is too easy or too small", temperature)
        before, after = probabilities(logits, kind), probabilities(logits, kind, temperature)
        thresholds = fit_thresholds(after, y, min_positives=p.min_positives) if labels.multi and p.fit_thresholds else None
        own = int((y.sum(axis=0) >= p.min_positives).sum()) if thresholds is not None else 0
        result = {
            "problem_type": kind,
            "temperature": temperature,
            "thresholds": thresholds,
            "at_bound": at_bound,
            "n": len(rows),
            "nll_before": nll(logits, y, kind),
            "nll_after": nll(logits, y, kind, temperature),
            "ece_before": ece(before, y, kind, p.n_bins),
            "ece_after": ece(after, y, kind, p.n_bins),
        }
        path = write_json(ctx.path(CALIBRATION_FILE), result)
        metrics = {f"calibration/{k}": float(result[k]) for k in
                   ("temperature", "nll_before", "nll_after", "ece_before", "ece_after", "n")}
        metrics["calibration/nll_drop"] = result["nll_before"] - result["nll_after"]
        metrics["calibration/at_bound"] = float(at_bound)
        if thresholds is not None:
            metrics["calibration/labels_with_own_threshold"] = float(own)
        return StageResult(artifacts={"calibration": path}, metrics=metrics)


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #


class ReportParams(StageParams):
    """What every stage that measures predictions on labelled rows shares."""

    data: str  # labelled JSONL; ideally prepare_data's test split
    n_bins: int = 15
    save_predictions: bool = True  # needed by a later stage's compare_to
    bootstrap_samples: int = Field(default=1000, ge=0)  # resamples behind the 95 % intervals; 0 turns them off
    # The `predictions` artifact of another evaluation of the same rows. Adds the paired difference
    # (this minus that) in macro-F1 and accuracy with its interval: eval/delta_macro_f1_ci_low > 0
    # means better beyond chance, which a fixed margin of a few thousandths cannot say.
    compare_to: str | None = None


def measure(
    ctx: StageContext, p: ReportParams, labels: LabelSpace, rows: list[dict[str, Any]], probs, pred, y,
    nll_value: float, source: str, extra: dict[str, Any] | None = None,
) -> StageResult:
    """The report, metrics and predictions of one set of predictions on labelled rows."""
    import numpy as np

    from .metrics import bootstrap_intervals, classification_report, confusion_matrix, ece, paired_bootstrap

    kind, seed = labels.problem_type, ctx.seed
    report = classification_report(pred, y, labels.names, kind)
    metrics = {f"eval/{k}": float(v) for k, v in report["metrics"].items()}
    metrics.update({
        "eval/n": float(len(rows)),
        "eval/nll": nll_value,
        "eval/ece": ece(probs, y, kind, p.n_bins),
        "eval/min_class_f1": min(v["f1"] for v in report["per_class"].values() if v["support"] > 0),
    })
    intervals: dict[str, Any] = {}
    if p.bootstrap_samples:
        intervals = bootstrap_intervals(pred, y, kind, len(labels), p.bootstrap_samples, seed)
        for name, (low, high) in intervals.items():
            metrics[f"eval/{name}_ci_low"], metrics[f"eval/{name}_ci_high"] = low, high

    comparison: dict[str, Any] = {}
    if p.compare_to:
        theirs = {r["id"]: r["pred"] for r in read_jsonl(p.compare_to)}
        shared = [i for i, row in enumerate(rows) if row_id(row) in theirs]
        if not shared:
            raise ValueError(f"{p.compare_to} shares no row with {p.data}; compare evaluations of the same split")
        other = [theirs[row_id(rows[i])] for i in shared]
        if labels.multi:
            matrix = np.zeros((len(other), len(labels)), dtype=bool)
            for i, on in enumerate(other):
                matrix[i, on] = True
            other = matrix
        comparison = paired_bootstrap(
            np.asarray(pred)[shared], np.asarray(other), np.asarray(y)[shared], kind, len(labels),
            max(p.bootstrap_samples, 200), seed,
        )
        for name, d in comparison.items():
            metrics[f"eval/delta_{name}"] = d["delta"]
            metrics[f"eval/delta_{name}_ci_low"], metrics[f"eval/delta_{name}_ci_high"] = d["ci_low"], d["ci_high"]
        metrics["eval/delta_n"] = float(len(shared))

    artifacts = {"report": write_json(ctx.path("report.json"), {
        "model": source, "data": p.data, **(extra or {}), **report,
        "intervals": {k: list(v) for k, v in intervals.items()}, "comparison": comparison,
    })}
    if not labels.multi:
        artifacts["confusion_matrix"] = write_json(ctx.path("confusion_matrix.json"), {
            "labels": labels.names, "rows": "true", "columns": "predicted",
            "matrix": confusion_matrix(pred, y, len(labels)),
        })
    if p.save_predictions:
        def decision(i: int) -> Any:
            return [c for c in range(len(labels)) if pred[i][c]] if labels.multi else int(pred[i])

        def name(i: int) -> Any:
            return [labels.names[c] for c in decision(i)] if labels.multi else labels.names[decision(i)]

        artifacts["predictions"] = ctx.path("predictions.jsonl")
        write_jsonl(artifacts["predictions"], (
            {"id": row_id(row), "text": row["text"], "gold": row.get("labels", row.get("label")),
             "pred": decision(i), "predicted": name(i), "probs": [round(float(x), 6) for x in probs[i]]}
            for i, row in enumerate(rows)
        ))
    return StageResult(artifacts=artifacts, metrics=metrics)


class EvalParams(InferenceParams, ReportParams):
    calibration: str | None = None


@register("classifiers.eval")
class Evaluate(Stage):
    """Measure a classifier on a labelled split: accuracy and F1 with intervals, calibration, a paired comparison."""

    Params = EvalParams

    def run(self, ctx: StageContext) -> StageResult:
        from .metrics import decide, ece, load_calibration, nll, probabilities
        from .modeling import gold_targets
        from .train import read_rows

        p: EvalParams = self.params
        labels = LabelSpace.load(p.model)
        rows = read_rows(p.data, labels)
        if not rows:
            raise ValueError(f"{p.data}: no row to evaluate")
        kind = labels.problem_type
        calibration = load_calibration(p.calibration)
        logits, y = _logits(p, rows), gold_targets(rows, labels)
        probs = probabilities(logits, kind, calibration["temperature"])
        pred = decide(probs, kind, calibration["thresholds"])
        result = measure(
            ctx, p, labels, rows, probs, pred, y, nll(logits, y, kind, calibration["temperature"]), p.model,
            {"calibration": calibration},
        )
        if p.calibration:
            result.metrics["eval/ece_uncalibrated"] = ece(probabilities(logits, kind), y, kind, p.n_bins)
        return result


class BaselineParams(ReportParams):
    """TF-IDF features and a linear model: minutes on a CPU, no GPU."""

    labels: str  # labels.json from prepare_data
    train_data: str
    ngram_range: tuple[int, int] = (1, 2)
    analyzer: Literal["word", "char_wb"] = "word"  # char_wb for languages without clear word boundaries
    min_df: int = 1
    max_features: int | None = 200_000
    c: float = Field(default=1.0, gt=0.0)  # inverse regularisation strength of the logistic regression
    max_iter: int = 1000


def baseline_text(row: dict[str, Any]) -> str:
    return f"{row['text']} [SEP] {row['text_pair']}" if row.get("text_pair") else str(row["text"])


@register("classifiers.baseline")
class Baseline(Stage):
    """TF-IDF + logistic regression on the same splits: the bar a fine-tuned encoder has to clear."""

    Params = BaselineParams

    def run(self, ctx: StageContext) -> StageResult:
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.multiclass import OneVsRestClassifier

        from .metrics import decide, nll_of_probabilities, targets_array
        from .train import read_rows

        p: BaselineParams = self.params
        labels = LabelSpace.load(p.labels)
        kind = labels.problem_type
        train = [r for r in read_rows(p.train_data, labels) if r.get("probs") is None]
        rows = read_rows(p.data, labels)
        if not train or not rows:
            raise ValueError("baseline needs labelled training rows and labelled rows to measure on")
        vectorizer = TfidfVectorizer(
            ngram_range=tuple(p.ngram_range), analyzer=p.analyzer, min_df=p.min_df, max_features=p.max_features,
            sublinear_tf=True,
        )
        x_train = vectorizer.fit_transform([baseline_text(r) for r in train])
        x = vectorizer.transform([baseline_text(r) for r in rows])
        y_train, y = targets_array(train, len(labels), kind), targets_array(rows, len(labels), kind)
        linear = LogisticRegression(C=p.c, max_iter=p.max_iter, random_state=self.seed(ctx))
        if labels.multi:
            # A label without a positive (or without a negative) in train cannot be fitted: it is never predicted.
            seen = [c for c in range(len(labels)) if 0 < y_train[:, c].sum() < len(train)]
            probs = np.zeros((len(rows), len(labels)))
            if seen:
                probs[:, seen] = OneVsRestClassifier(linear).fit(x_train, y_train[:, seen]).predict_proba(x)
        else:
            model = linear.fit(x_train, y_train)
            probs = np.zeros((len(rows), len(labels)))
            probs[:, model.classes_] = model.predict_proba(x)
        pred = decide(probs, kind)
        result = measure(
            ctx, p, labels, rows, probs, pred, y, nll_of_probabilities(probs, y, kind), "tfidf+logistic-regression",
            {"features": int(x_train.shape[1])},
        )
        result.metrics["baseline/features"] = float(x_train.shape[1])
        result.metrics["baseline/n_train"] = float(len(train))
        return result


# --------------------------------------------------------------------------- #
# label audit
# --------------------------------------------------------------------------- #


class AuditParams(StageParams):
    """Reads the `oof` artifact of every run of a `classifiers.finetune` matrix over `fold`."""

    group: str  # the matrix whose runs each held one fold out
    labels: str  # labels.json from prepare_data
    train_data: str | None = None  # the audited file; needed to write a cleaned copy and to report coverage
    review_fraction: float = Field(default=0.02, gt=0.0, le=1.0)  # the highest-loss share written as suspects
    # The share of training rows removed from `train_clean`, highest loss first, among rows the
    # model also disagrees with. 0 only reports: relabelling the suspects by hand beats deleting them.
    drop_fraction: float = Field(default=0.0, ge=0.0, lt=1.0)

    @model_validator(mode="after")
    def _needs_file(self) -> AuditParams:
        if self.drop_fraction > 0 and not self.train_data:
            raise ValueError("drop_fraction needs train_data: the file to write without the dropped rows")
        return self


def rank_suspects(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Out-of-fold rows, one per text (a text predicted by several runs gets their mean loss and
    mean distribution), highest loss first."""
    merged: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = row_id(row)
        if key not in merged:
            merged[key] = {**row, "id": key, "runs": 0, "loss": 0.0, "probs": [0.0] * len(row["probs"])}
        m = merged[key]
        m["runs"] += 1
        m["loss"] += row["loss"]
        m["probs"] = [a + b for a, b in zip(m["probs"], row["probs"], strict=True)]
    out = []
    for m in merged.values():
        n = m["runs"]
        probs = [x / n for x in m["probs"]]
        if "labels" in m:
            predicted = [c for c, x in enumerate(probs) if x >= 0.5]
            disagrees = sorted(predicted) != sorted(m["labels"])
        else:
            predicted = max(range(len(probs)), key=probs.__getitem__)
            disagrees = predicted != m["label"]
        out.append({**m, "loss": m["loss"] / n, "probs": probs, "predicted": predicted, "disagrees": disagrees})
    return sorted(out, key=lambda r: -r["loss"])


@register("classifiers.audit_labels")
class AuditLabels(Stage):
    """Rank training rows by their out-of-fold loss: the likeliest label errors come first."""

    Params = AuditParams

    def run(self, ctx: StageContext) -> StageResult:
        import numpy as np

        from .metrics import classification_report, targets_array

        p: AuditParams = self.params
        labels = LabelSpace.load(p.labels)
        runs = group_runs(ctx.upstream, p.group)
        missing = [name for name, m in runs.items() if "oof" not in m.artifacts]
        if not runs or missing:
            raise ValueError(
                f"audit_labels: {missing or p.group!r} wrote no out-of-fold predictions; "
                "run classifiers.finetune as a matrix over `fold`"
            )
        rows = [r for m in runs.values() for r in read_jsonl(m.artifact("oof", ctx.run_dir))]
        ranked = rank_suspects(rows)
        kind = labels.problem_type
        y = targets_array(ranked, len(labels), kind)
        if labels.multi:
            pred = np.zeros((len(ranked), len(labels)), dtype=bool)
            for i, r in enumerate(ranked):
                pred[i, r["predicted"]] = True
        else:
            pred = np.asarray([r["predicted"] for r in ranked])
        report = classification_report(pred, y, labels.names, kind)["metrics"]

        def named(v: Any) -> Any:
            return [labels.names[c] for c in v] if isinstance(v, list) else labels.names[v]

        n_review = max(1, round(len(ranked) * p.review_fraction))
        suspects = [
            {**{k: r[k] for k in ("id", "text", "text_pair") if k in r}, "given": named(r.get("labels", r.get("label"))),
             "predicted": named(r["predicted"]), "disagrees": r["disagrees"], "loss": round(r["loss"], 6),
             "probs": [round(x, 6) for x in r["probs"]]}
            for r in ranked[:n_review]
        ]
        artifacts: dict[str, Any] = {"suspects": ctx.path("suspects.jsonl")}
        write_jsonl(artifacts["suspects"], suspects)
        metrics = {
            "audit/rows": float(len(ranked)), "audit/folds": float(len({r["fold"] for r in rows})),
            "audit/suspects": float(len(suspects)),
            "audit/disagreements": float(sum(r["disagrees"] for r in ranked)),
            # Every row was predicted by a model that never trained on it: an honest estimate of the
            # recipe's quality on all of the training data, without spending the eval split.
            "audit/oof_accuracy": report["accuracy"], "audit/oof_macro_f1": report["macro_f1"],
        }
        if p.train_data:
            train = list(read_jsonl(p.train_data))
            audited = {r["id"] for r in ranked}
            metrics["audit/coverage"] = sum(row_id(r) in audited for r in train) / max(len(train), 1)
            if p.drop_fraction > 0:
                limit = round(len(train) * p.drop_fraction)
                drop = {r["id"] for r in [x for x in ranked if x["disagrees"]][:limit]}
                artifacts["train_clean"] = ctx.path("train_clean.jsonl")
                write_jsonl(artifacts["train_clean"], (r for r in train if row_id(r) not in drop))
                metrics["audit/dropped"] = float(len(drop))
        return StageResult(artifacts=artifacts, metrics=metrics)


# --------------------------------------------------------------------------- #
# export
# --------------------------------------------------------------------------- #


class ExportParams(StageParams):
    model: str
    calibration: str | None = None
    max_length: int | None = None  # None: the length the model was trained with
    onnx: bool = False  # also write model.onnx; skipped with a note unless onnx and onnxscript are installed
    onnx_opset: int | None = None
    trust_remote_code: bool = False


def export_onnx(model_dir: Path, path: Path, max_length: int, opset: int | None, trust_remote_code: bool) -> None:
    """`model.onnx` with dynamic batch and sequence axes: `(input_ids, attention_mask) -> logits`."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=trust_remote_code)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir), trust_remote_code=trust_remote_code)
    model.eval()

    class Logits(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, input_ids, attention_mask):
            return self.inner(input_ids=input_ids, attention_mask=attention_mask).logits

    sample = tokenizer(["export sample", "a second, longer export sample"], padding=True, truncation=True,
                       max_length=max_length, return_tensors="pt")
    batch, sequence = torch.export.Dim("batch"), torch.export.Dim("sequence")
    kwargs: dict[str, Any] = {"opset_version": opset} if opset else {}
    torch.onnx.export(
        Logits(model).eval(), (sample["input_ids"], sample["attention_mask"]), str(path),
        input_names=["input_ids", "attention_mask"], output_names=["logits"],
        dynamic_shapes={"input_ids": {0: batch, 1: sequence}, "attention_mask": {0: batch, 1: sequence}},
        dynamo=True, **kwargs,
    )


@register("classifiers.export")
class Export(Stage):
    """Assemble the deployable directory: weights, tokenizer, label map and calibration."""

    Params = ExportParams

    def run(self, ctx: StageContext) -> StageResult:
        from .metrics import load_calibration
        from .modeling import read_preprocessing

        p: ExportParams = self.params
        labels = LabelSpace.load(p.model)
        calibration = load_calibration(p.calibration)
        preprocessing = read_preprocessing(p.model)
        max_length = p.max_length or preprocessing["max_length"]
        out = ctx.out_dir / "model"
        if out.exists():
            shutil.rmtree(out)
        shutil.copytree(p.model, out, ignore=shutil.ignore_patterns("checkpoint-*", "training_args.bin"))
        labels.save(out / LABELS_FILE)
        if p.calibration:
            write_json(out / CALIBRATION_FILE, calibration)
        write_json(out / CONFIG_FILE, {
            "problem_type": labels.problem_type,
            "labels": labels.names,
            "temperature": calibration["temperature"],
            "thresholds": calibration["thresholds"],
            "max_length": max_length,
            "truncation": preprocessing["truncation"],
        })
        artifacts: dict[str, Any] = {"model": out}
        notes: dict[str, Any] = {}
        onnx_ok = 0.0
        if p.onnx:
            missing = [m for m in ("onnx", "onnxscript") if importlib.util.find_spec(m) is None]
            if missing:
                notes["onnx"] = f"skipped: {' and '.join(missing)} not installed (pip install onnx onnxscript)"
                ctx.log.warning("export: ONNX %s", notes["onnx"])
            else:
                try:
                    export_onnx(out, out / "model.onnx", max_length, p.onnx_opset, p.trust_remote_code)
                    artifacts["onnx"] = out / "model.onnx"
                    notes["onnx"], onnx_ok = "written", 1.0
                except Exception as e:  # an export failure must not lose the finished model
                    notes["onnx"] = f"failed: {type(e).__name__}: {e}"
                    ctx.log.warning("export: ONNX %s", notes["onnx"])
        size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
        return StageResult(
            artifacts=artifacts, notes=notes,
            metrics={"export/size_mb": round(size, 3), "export/onnx": onnx_ok, "export/n_labels": float(len(labels))},
        )
