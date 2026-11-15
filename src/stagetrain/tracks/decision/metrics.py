"""Metrics over scored decisions, and the temperature fit. Plain Python: no torch needed.

A prediction is `{"type", "source", "scores": [logit per option], "target": [probability per option]}`.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from .schema import CHOICE, NOUL, SCORE, TYPES

EPS = 1e-12


def softmax(scores: Sequence[float], temperature: float = 1.0) -> list[float]:
    scaled = [s / temperature for s in scores]
    m = max(scaled)
    exps = [math.exp(s - m) for s in scaled]
    total = sum(exps)
    return [e / total for e in exps]


def probabilities(pred: Mapping[str, Any], temperatures: Mapping[str, float] | None = None) -> list[float]:
    return softmax(pred["scores"], (temperatures or {}).get(pred["type"], 1.0))


def _argmax(xs: Sequence[float]) -> int:
    return max(range(len(xs)), key=xs.__getitem__)


def ece(confidences: Sequence[float], correct: Sequence[bool], bins: int = 15) -> float:
    """Expected calibration error of the top answer, equal-width bins."""
    if not confidences:
        return 0.0
    sums = [[0.0, 0.0, 0] for _ in range(bins)]
    for c, ok in zip(confidences, correct, strict=True):
        b = min(int(c * bins), bins - 1)
        sums[b][0] += c
        sums[b][1] += float(ok)
        sums[b][2] += 1
    n = len(confidences)
    return sum(abs(conf / k - acc / k) * k / n for conf, acc, k in sums if k)


def auroc(scores: Sequence[float], positives: Sequence[bool]) -> float | None:
    """Area under the ROC curve by ranks; ties share their rank. None when one class is missing."""
    n_pos = sum(positives)
    n_neg = len(positives) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    order = sorted(range(len(scores)), key=scores.__getitem__)
    rank_sum, i = 0.0, 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        mean_rank = (i + j) / 2 + 1
        rank_sum += mean_rank * sum(positives[order[k]] for k in range(i, j + 1))
        i = j + 1
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def nll(preds: Sequence[Mapping[str, Any]], temperatures: Mapping[str, float] | None = None) -> float:
    """Mean cross-entropy between the targets and the predicted distributions."""
    total = 0.0
    for p in preds:
        probs = probabilities(p, temperatures)
        total -= sum(t * math.log(max(q, EPS)) for t, q in zip(p["target"], probs, strict=True) if t > 0)
    return total / max(len(preds), 1)


def summarize(
    preds: Sequence[Mapping[str, Any]], temperatures: Mapping[str, float] | None = None, bins: int = 15
) -> dict[str, float]:
    """Accuracy, NLL, ECE and Brier over all predictions, accuracy per question type, the mean
    absolute level error of score questions, and the AUROC of yes/no questions."""
    if not preds:
        return {"n": 0.0}
    conf, correct, brier = [], [], 0.0
    by_type: dict[str, list[bool]] = defaultdict(list)
    level_error, yes_prob, yes_true = [], [], []
    for p in preds:
        probs = probabilities(p, temperatures)
        gold, top = _argmax(p["target"]), _argmax(probs)
        conf.append(probs[top])
        correct.append(top == gold)
        by_type[p["type"]].append(top == gold)
        brier += sum((q - t) ** 2 for q, t in zip(probs, p["target"], strict=True))
        if p["type"] == SCORE:
            level_error.append(abs(top - gold))
        elif p["type"] == NOUL:
            yes_prob.append(probs[0])
            yes_true.append(gold == 0)
    n = len(preds)
    out = {
        "n": float(n),
        "accuracy": sum(correct) / n,
        "nll": nll(preds, temperatures),
        "ece": ece(conf, correct, bins),
        "brier": brier / n,
    }
    for kind in TYPES:
        if by_type[kind]:
            out[f"accuracy_{kind}"] = sum(by_type[kind]) / len(by_type[kind])
    if level_error:
        out["score_mae"] = sum(level_error) / len(level_error)
    area = auroc(yes_prob, yes_true)
    if area is not None:
        out["noul_auroc"] = area
    return out


def by_source(
    preds: Sequence[Mapping[str, Any]], temperatures: Mapping[str, float] | None = None
) -> dict[str, dict[str, float]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for p in preds:
        groups[p.get("source", "default")].append(p)
    return {name: summarize(group, temperatures) for name, group in sorted(groups.items())}


def fit_temperature(preds: Sequence[Mapping[str, Any]], lo: float = 0.05, hi: float = 20.0) -> float:
    """The temperature that minimises NLL on `preds`, by golden-section search over log T.
    NLL is convex in 1/T, so one minimum."""
    if not preds:
        return 1.0

    def loss(log_t: float) -> float:
        t = math.exp(log_t)
        total = 0.0
        for p in preds:
            probs = softmax(p["scores"], t)
            total -= sum(y * math.log(max(q, EPS)) for y, q in zip(p["target"], probs, strict=True) if y > 0)
        return total

    a, b = math.log(lo), math.log(hi)
    phi = (math.sqrt(5) - 1) / 2
    c, d = b - phi * (b - a), a + phi * (b - a)
    fc, fd = loss(c), loss(d)
    for _ in range(60):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = loss(c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = loss(d)
    return math.exp((a + b) / 2)


def fit_temperatures(
    preds: Sequence[Mapping[str, Any]], min_examples: int = 20, lo: float = 0.2, hi: float = 10.0
) -> dict[str, float]:
    """One temperature per question type, within `[lo, hi]`; a type with too few examples keeps 1.0.
    The bounds matter when the held-out set is answered without a single error: NLL then keeps
    falling as T goes to zero, and the fit would only say "be certain"."""
    out = {}
    for kind in (NOUL, CHOICE, SCORE):
        group = [p for p in preds if p["type"] == kind]
        out[kind] = fit_temperature(group, lo, hi) if len(group) >= min_examples else 1.0
    return out
