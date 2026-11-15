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


def band(n_options: int) -> str:
    """The option-count band a question falls in: a model's confidence behaves differently with
    2, a handful, or dozens of options, so each band can carry its own temperature."""
    if n_options <= 2:
        return "2"
    if n_options <= 5:
        return "3-5"
    return "6-10" if n_options <= 10 else "11+"


def temperature_of(kind: str, n_options: int, calibration: Mapping[str, Any] | None = None) -> float:
    """The temperature for a question of `kind` with `n_options` options. `calibration` is
    `{"temperature": {type: T}, "bands": {type: {band: T}}}`; a flat `{type: T}` is read too."""
    if not calibration:
        return 1.0
    by_type = calibration.get("temperature", calibration)
    banded = (calibration.get("bands") or {}).get(kind, {}).get(band(n_options))
    if banded is not None:
        return float(banded)
    value = by_type.get(kind, 1.0)
    return float(value) if isinstance(value, (int, float)) else 1.0


def probabilities(pred: Mapping[str, Any], calibration: Mapping[str, Any] | None = None) -> list[float]:
    return softmax(pred["scores"], temperature_of(pred["type"], len(pred["scores"]), calibration))


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


def nll(preds: Sequence[Mapping[str, Any]], temperatures: Mapping[str, Any] | None = None) -> float:
    """Mean cross-entropy between the targets and the predicted distributions."""
    total = 0.0
    for p in preds:
        probs = probabilities(p, temperatures)
        total -= sum(t * math.log(max(q, EPS)) for t, q in zip(p["target"], probs, strict=True) if t > 0)
    return total / max(len(preds), 1)


def summarize(
    preds: Sequence[Mapping[str, Any]], temperatures: Mapping[str, Any] | None = None, bins: int = 15
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
    preds: Sequence[Mapping[str, Any]], temperatures: Mapping[str, Any] | None = None
) -> dict[str, dict[str, float]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for p in preds:
        groups[p.get("source", "default")].append(p)
    return {name: summarize(group, temperatures) for name, group in sorted(groups.items())}


def hard(pred: Mapping[str, Any]) -> list[float]:
    """The target as a hard label: all mass on the most likely option."""
    gold = _argmax(pred["target"])
    return [1.0 if i == gold else 0.0 for i in range(len(pred["target"]))]


def fit_temperature(preds: Sequence[Mapping[str, Any]], lo: float = 0.5, hi: float = 5.0) -> float:
    """The temperature in `[lo, hi]` that minimises NLL on `preds`, against hard labels, by
    golden-section search over log T. NLL is convex in 1/T, so one minimum.

    The bounds are part of the method. A temperature chosen freely on a set the model answers
    without error runs to zero ("be certain"), and one far below 1 turns probabilities into 0 / 1.
    """
    if not preds:
        return 1.0
    targets = [hard(p) for p in preds]

    def loss(log_t: float) -> float:
        t = math.exp(log_t)
        total = 0.0
        for p, target in zip(preds, targets, strict=True):
            probs = softmax(p["scores"], t)
            total -= sum(y * math.log(max(q, EPS)) for y, q in zip(target, probs, strict=True) if y > 0)
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


def fit_calibration(
    preds: Sequence[Mapping[str, Any]], min_examples: int = 50, lo: float = 0.5, hi: float = 5.0
) -> dict[str, Any]:
    """One temperature per question type, and one per option-count band of a type that has at
    least `min_examples` rows; a band without enough rows falls back to its type's temperature."""
    temperature: dict[str, float] = {}
    bands: dict[str, dict[str, float]] = {}
    for kind in (NOUL, CHOICE, SCORE):
        group = [p for p in preds if p["type"] == kind]
        temperature[kind] = fit_temperature(group, lo, hi) if len(group) >= min_examples else 1.0
        by_band: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for p in group:
            by_band[band(len(p["scores"]))].append(p)
        if len(by_band) > 1:  # a type with one band is already covered by its own temperature
            fitted = {b: fit_temperature(rows, lo, hi) for b, rows in sorted(by_band.items()) if len(rows) >= min_examples}
            if fitted:
                bands[kind] = fitted
    return {"temperature": temperature, "bands": bands}


def fit_temperatures(
    preds: Sequence[Mapping[str, Any]], min_examples: int = 20, lo: float = 0.5, hi: float = 5.0
) -> dict[str, float]:
    """One temperature per question type (no bands)."""
    return fit_calibration(preds, min_examples, lo, hi)["temperature"]


def out_of_fold(preds: Sequence[Mapping[str, Any]], min_examples: int = 50, lo: float = 0.5, hi: float = 5.0) -> dict[str, float]:
    """NLL and ECE when each half of `preds` is scored with the calibration fitted on the other
    half: what the calibration is worth on rows it was not fitted on."""
    halves = [list(preds[0::2]), list(preds[1::2])]
    if not halves[1]:
        return {}
    scored: list[tuple[Mapping[str, Any], dict[str, Any]]] = []
    for mine, other in ((halves[0], halves[1]), (halves[1], halves[0])):
        calibration = fit_calibration(other, min_examples, lo, hi)
        scored += [(p, calibration) for p in mine]
    conf, correct, total = [], [], 0.0
    for p, calibration in scored:
        probs = probabilities(p, calibration)
        gold, top = _argmax(p["target"]), _argmax(probs)
        conf.append(probs[top])
        correct.append(top == gold)
        total -= math.log(max(probs[gold], EPS))
    return {"nll": total / len(scored), "ece": ece(conf, correct)}


# --------------------------------------------------------------------------- #
# what a number is worth: intervals, paired differences, chance
# --------------------------------------------------------------------------- #


def bootstrap_interval(values: Sequence[float], resamples: int = 1000, seed: int = 0) -> tuple[float, float]:
    """A 95 % percentile-bootstrap interval of the mean of `values`."""
    import random

    n = len(values)
    if n < 2:
        mean = sum(values) / n if n else 0.0
        return mean, mean
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(resamples))
    return means[int(0.025 * resamples)], means[min(int(0.975 * resamples), resamples - 1)]


def item_scores(preds: Sequence[Mapping[str, Any]], calibration: Mapping[str, Any] | None = None) -> dict[str, dict[str, float]]:
    """Per record id: whether the top answer is right, and the log loss of the right answer."""
    out = {}
    for p in preds:
        probs = probabilities(p, calibration)
        gold = _argmax(p["target"])
        out[str(p["id"])] = {"correct": float(_argmax(probs) == gold), "nll": -math.log(max(probs[gold], EPS))}
    return out


def paired_difference(
    mine: Mapping[str, Mapping[str, float]], baseline: Mapping[str, Mapping[str, float]], resamples: int = 1000
) -> dict[str, float]:
    """This model minus a baseline over the records both answered: accuracy (higher is better)
    and log loss (lower is better), each with a paired-bootstrap interval. Two models make
    correlated errors on the same records, so the paired interval is far tighter than the two
    separate ones."""
    shared = sorted(set(mine) & set(baseline))
    if not shared:
        return {"n": 0.0}
    accuracy = [mine[k]["correct"] - baseline[k]["correct"] for k in shared]
    loss = [mine[k]["nll"] - baseline[k]["nll"] for k in shared]
    acc_low, acc_high = bootstrap_interval(accuracy, resamples)
    nll_low, nll_high = bootstrap_interval(loss, resamples)
    return {
        "n": float(len(shared)),
        "delta_accuracy": sum(accuracy) / len(shared), "delta_accuracy_ci_low": acc_low, "delta_accuracy_ci_high": acc_high,
        "delta_nll": sum(loss) / len(shared), "delta_nll_ci_low": nll_low, "delta_nll_ci_high": nll_high,
    }


def skill(preds: Sequence[Mapping[str, Any]], calibration: Mapping[str, Any] | None = None) -> float:
    """Accuracy corrected for chance, `(accuracy - chance) / (1 - chance)` clipped to [0, 1], where
    chance is what guessing uniformly among each question's options would score. 0 is guessing
    and 1 is perfect, whatever the number of options, so sources can be averaged."""
    if not preds:
        return 0.0
    accuracy = sum(_argmax(probabilities(p, calibration)) == _argmax(p["target"]) for p in preds) / len(preds)
    chance = sum(1.0 / len(p["scores"]) for p in preds) / len(preds)
    return min(max((accuracy - chance) / (1 - chance), 0.0), 1.0) if chance < 1 else 0.0


def macro_skill(preds: Sequence[Mapping[str, Any]], calibration: Mapping[str, Any] | None = None) -> float:
    """The mean chance-corrected accuracy over sources: every source counts once, however many
    rows it has."""
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for p in preds:
        groups[p.get("source", "default")].append(p)
    return sum(skill(g, calibration) for g in groups.values()) / len(groups) if groups else 0.0


def confident_errors(preds: Sequence[Mapping[str, Any]], calibration: Mapping[str, Any] | None = None, level: float = 0.9) -> float:
    """The share of answers that are wrong and given at least `level` confidence: the errors a
    threshold on confidence would let through."""
    if not preds:
        return 0.0
    wrong = 0
    for p in preds:
        probs = probabilities(p, calibration)
        top = _argmax(probs)
        wrong += probs[top] >= level and top != _argmax(p["target"])
    return wrong / len(preds)
