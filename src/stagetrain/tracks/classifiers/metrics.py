"""Classification metrics and post-hoc calibration on logits. numpy only, imported when used."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .labels import MULTI

CALIBRATION_FILE = "calibration.json"


def _np():
    import numpy as np

    return np


def probabilities(logits, problem_type: str, temperature: float = 1.0):
    """Softmax (single-label) or sigmoid (multi-label) of `logits / temperature`, `(N, C)`."""
    np = _np()
    z = np.asarray(logits, dtype=np.float64) / temperature
    if problem_type == MULTI:
        return np.exp(-np.logaddexp(0.0, -z))  # 1 / (1 + exp(-z)) without overflow
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def nll(logits, targets, problem_type: str, temperature: float = 1.0) -> float:
    """Mean negative log-likelihood. `targets` are class indices `(N,)` for single-label and a
    0/1 matrix `(N, C)` for multi-label (averaged over every label of every row)."""
    np = _np()
    z = np.asarray(logits, dtype=np.float64) / temperature
    if problem_type == MULTI:
        y = np.asarray(targets, dtype=np.float64)
        # log(1 + exp(z)) - y*z, written to stay finite for large |z|
        return float((np.maximum(z, 0) - y * z + np.log1p(np.exp(-np.abs(z)))).mean())
    z = z - z.max(axis=-1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=-1, keepdims=True))
    return float(-logp[np.arange(len(z)), np.asarray(targets, dtype=np.int64)].mean())


def ece(probs, targets, problem_type: str, n_bins: int = 15) -> float:
    """Expected calibration error with equal-width confidence bins. Single-label: the confidence
    of the predicted class against whether it is right. Multi-label: every (row, label) decision
    at 0.5, with confidence `max(p, 1 - p)`."""
    np = _np()
    p = np.asarray(probs, dtype=np.float64)
    if problem_type == MULTI:
        y = np.asarray(targets) >= 0.5
        conf = np.maximum(p, 1 - p).ravel()
        correct = ((p >= 0.5) == y).ravel()
    else:
        conf = p.max(axis=-1)
        correct = p.argmax(axis=-1) == np.asarray(targets, dtype=np.int64)
    if len(conf) == 0:
        return 0.0
    bins = np.minimum((conf * n_bins).astype(np.int64), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            total += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(total)


def fit_temperature(logits, targets, problem_type: str, low: float = 0.05, high: float = 20.0) -> float:
    """The temperature that minimises the NLL of `logits / T`: golden-section search over log T.
    The NLL is unimodal in log T, so the search finds the minimum on `[low, high]`."""
    a, b = math.log(low), math.log(high)
    ratio = (math.sqrt(5) - 1) / 2

    def f(x: float) -> float:
        return nll(logits, targets, problem_type, math.exp(x))

    c, d = b - ratio * (b - a), a + ratio * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(60):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - ratio * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + ratio * (b - a)
            fd = f(d)
    return float(math.exp((a + b) / 2))


def _f1(tp: float, fp: float, fn: float) -> float:
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def _best_cut(pred_at, positives, grid: list[float], default: float) -> float:
    """The cut on `grid` with the best F1; ties go to the cut nearest `default`."""
    best, best_f1 = default, -1.0
    for t in sorted(grid, key=lambda t: abs(t - default)):
        pred = pred_at(t)
        score = _f1((pred & positives).sum(), (pred & ~positives).sum(), (~pred & positives).sum())
        if score > best_f1 + 1e-12:
            best, best_f1 = t, score
    return float(best)


def fit_thresholds(
    probs, targets, grid: list[float] | None = None, default: float = 0.5, min_positives: int = 1
) -> list[float]:
    """For every label of a multi-label task, the probability cut that maximises its F1 on this
    split.

    A cut fitted on a handful of positives is noise, so a label with fewer than `min_positives`
    positives here does not get its own: it takes the one cut that is best for all labels together
    (micro-F1). With `min_positives=1` such a label is one without any positive, and keeps `default`.
    """
    np = _np()
    p, y = np.asarray(probs, dtype=np.float64), np.asarray(targets) >= 0.5
    grid = grid or [round(0.05 * i, 2) for i in range(1, 20)]
    enough = y.sum(axis=0) >= max(min_positives, 1)
    shared = default
    if min_positives > 1 and not enough.all() and y.any():
        shared = _best_cut(lambda t: p >= t, y, grid, default)
    return [
        _best_cut(lambda t, c=c: p[:, c] >= t, y[:, c], grid, default) if enough[c] else float(shared)
        for c in range(p.shape[1])
    ]


def decide(probs, problem_type: str, thresholds: list[float] | None = None):
    """Predictions from probabilities: class indices `(N,)`, or a boolean matrix `(N, C)`."""
    np = _np()
    p = np.asarray(probs)
    if problem_type == MULTI:
        return p >= (np.asarray(thresholds) if thresholds is not None else 0.5)
    return p.argmax(axis=-1)


def _per_class(tp, fp, fn, names: list[str]) -> dict[str, dict[str, float]]:
    out = {}
    for i, name in enumerate(names):
        t, p, n = float(tp[i]), float(fp[i]), float(fn[i])
        out[name] = {
            "precision": t / (t + p) if t + p else 0.0,
            "recall": t / (t + n) if t + n else 0.0,
            "f1": _f1(t, p, n),
            "support": int(t + n),
        }
    return out


def _averages(tp, fp, fn, per_class: dict[str, dict[str, float]]) -> dict[str, float]:
    # Macro averages over the classes that occur in the targets or the predictions, as scikit-learn does.
    seen = [i for i in range(len(tp)) if tp[i] + fp[i] + fn[i] > 0]
    f1 = [v["f1"] for v in per_class.values()]
    support = [v["support"] for v in per_class.values()]
    return {
        "macro_f1": sum(f1[i] for i in seen) / len(seen) if seen else 0.0,
        "micro_f1": _f1(float(sum(tp)), float(sum(fp)), float(sum(fn))),
        "weighted_f1": sum(f * s for f, s in zip(f1, support, strict=True)) / sum(support) if sum(support) else 0.0,
    }


def confusion_matrix(pred, target, n_labels: int) -> list[list[int]]:
    """`matrix[true][predicted]` counts."""
    np = _np()
    m = np.zeros((n_labels, n_labels), dtype=np.int64)
    np.add.at(m, (np.asarray(target, dtype=np.int64), np.asarray(pred, dtype=np.int64)), 1)
    return m.tolist()


def classification_report(pred, target, names: list[str], problem_type: str) -> dict[str, Any]:
    """`{"metrics": {...}, "per_class": {name: precision/recall/f1/support}}`. `accuracy` is exact
    match: the class for single-label, the whole label set for multi-label."""
    np = _np()
    if problem_type == MULTI:
        p, y = np.asarray(pred).astype(bool), np.asarray(target) >= 0.5
        tp, fp, fn = (p & y).sum(0), (p & ~y).sum(0), (~p & y).sum(0)
        metrics = {"accuracy": float((p == y).all(axis=1).mean()) if len(p) else 0.0,
                   "hamming_loss": float((p != y).mean()) if p.size else 0.0}
    else:
        p, y = np.asarray(pred, dtype=np.int64), np.asarray(target, dtype=np.int64)
        m = np.asarray(confusion_matrix(p, y, len(names)))
        tp = np.diag(m)
        fp, fn = m.sum(0) - tp, m.sum(1) - tp
        metrics = {"accuracy": float((p == y).mean()) if len(p) else 0.0}
    per_class = _per_class(tp, fp, fn, names)
    metrics.update(_averages(tp, fp, fn, per_class))
    return {"metrics": metrics, "per_class": per_class}


# --------------------------------------------------------------------------- #
# targets, per-row losses, and intervals
# --------------------------------------------------------------------------- #


def targets_array(rows, n_labels: int, problem_type: str):
    """Evaluation targets: class indices `(N,)`, or a 0/1 matrix `(N, C)` for multi-label."""
    np = _np()
    if problem_type == MULTI:
        y = np.zeros((len(rows), n_labels), dtype=np.int64)
        for i, row in enumerate(rows):
            y[i, row["labels"]] = 1
        return y
    return np.asarray([row["label"] for row in rows], dtype=np.int64)


def row_losses(logits, targets, problem_type: str):
    """Each row's own negative log-likelihood `(N,)`: what ranks rows by how much the model
    disagrees with their label."""
    np = _np()
    z = np.asarray(logits, dtype=np.float64)
    if problem_type == MULTI:
        y = np.asarray(targets, dtype=np.float64)
        return (np.maximum(z, 0) - y * z + np.log1p(np.exp(-np.abs(z)))).mean(axis=-1)
    z = z - z.max(axis=-1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=-1, keepdims=True))
    return -logp[np.arange(len(z)), np.asarray(targets, dtype=np.int64)]


def nll_of_probabilities(probs, targets, problem_type: str) -> float:
    """Mean negative log-likelihood from probabilities, for a model that has no logits."""
    np = _np()
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1 - 1e-12)
    if problem_type == MULTI:
        y = np.asarray(targets, dtype=np.float64)
        return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
    return float(-np.log(p[np.arange(len(p)), np.asarray(targets, dtype=np.int64)]).mean())


def headline(pred, target, problem_type: str, n_labels: int) -> tuple[float, float]:
    """`(accuracy, macro_f1)`, the same numbers as `classification_report` but cheap enough to
    compute a thousand times."""
    np = _np()
    if problem_type == MULTI:
        p, y = np.asarray(pred).astype(bool), np.asarray(target) >= 0.5
        tp, fp, fn = (p & y).sum(0), (p & ~y).sum(0), (~p & y).sum(0)
        accuracy = float((p == y).all(axis=1).mean()) if len(p) else 0.0
    else:
        p, y = np.asarray(pred, dtype=np.int64), np.asarray(target, dtype=np.int64)
        m = np.bincount(y * n_labels + p, minlength=n_labels * n_labels).reshape(n_labels, n_labels)
        tp = np.diag(m)
        fp, fn = m.sum(0) - tp, m.sum(1) - tp
        accuracy = float(tp.sum() / len(p)) if len(p) else 0.0
    seen = (tp + fp + fn) > 0
    f1 = 2 * tp[seen] / (2 * tp[seen] + fp[seen] + fn[seen])
    return accuracy, float(f1.mean()) if seen.any() else 0.0


def bootstrap_intervals(
    pred, target, problem_type: str, n_labels: int, samples: int = 1000, seed: int = 0
) -> dict[str, tuple[float, float]]:
    """95 % percentile intervals of accuracy and macro-F1, from resampling the rows with
    replacement. They say how far the number would move on another draw of this split."""
    np = _np()
    pred, target = np.asarray(pred), np.asarray(target)
    rng = np.random.default_rng(seed)
    draws = np.empty((samples, 2))
    for b in range(samples):
        idx = rng.integers(0, len(pred), len(pred))
        draws[b] = headline(pred[idx], target[idx], problem_type, n_labels)
    low, high = np.percentile(draws, [2.5, 97.5], axis=0)
    return {"accuracy": (float(low[0]), float(high[0])), "macro_f1": (float(low[1]), float(high[1]))}


def paired_bootstrap(
    pred, other, target, problem_type: str, n_labels: int, samples: int = 1000, seed: int = 0
) -> dict[str, dict[str, float]]:
    """This model minus another on the same rows: the difference in accuracy and macro-F1 with a
    95 % interval.

    Both models are scored on the same resampled rows, so what they get wrong together cancels
    out. That makes the interval of the difference much tighter than the two models' own
    intervals: `ci_low > 0` means better beyond chance, `ci_high < 0` worse.
    """
    np = _np()
    pred, other, target = np.asarray(pred), np.asarray(other), np.asarray(target)
    rng = np.random.default_rng(seed)
    point = np.subtract(headline(pred, target, problem_type, n_labels), headline(other, target, problem_type, n_labels))
    draws = np.empty((samples, 2))
    for b in range(samples):
        idx = rng.integers(0, len(pred), len(pred))
        draws[b] = np.subtract(
            headline(pred[idx], target[idx], problem_type, n_labels),
            headline(other[idx], target[idx], problem_type, n_labels),
        )
    low, high = np.percentile(draws, [2.5, 97.5], axis=0)
    return {
        name: {"delta": float(point[i]), "ci_low": float(low[i]), "ci_high": float(high[i])}
        for i, name in enumerate(("accuracy", "macro_f1"))
    }


def load_calibration(path: str | Path | None) -> dict[str, Any]:
    """A `calibration.json`, from the file or a directory holding it; the identity when `path` is None."""
    if path is None:
        return {"temperature": 1.0, "thresholds": None}
    path = Path(path)
    if path.is_dir():
        path = path / CALIBRATION_FILE
    data = json.loads(path.read_text(encoding="utf-8"))
    data.setdefault("temperature", 1.0)
    data.setdefault("thresholds", None)
    return data
