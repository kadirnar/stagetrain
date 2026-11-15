"""Stages `decision.calibrate` and `decision.eval`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from stagetrain.core import (
    Stage,
    StageContext,
    StageParams,
    StageResult,
    read_jsonl,
    register,
    write_json,
    write_jsonl,
)

from .metrics import (
    bootstrap_interval,
    by_source,
    confident_errors,
    fit_calibration,
    item_scores,
    macro_skill,
    out_of_fold,
    paired_difference,
    probabilities,
    skill,
    summarize,
)
from .modeling import load_model, load_records, load_tokenizer, read_meta, score_records
from .prompt import PromptStyle
from .schema import CHOICE


class ScoreParams(StageParams):
    model: str
    data: str
    max_samples: int | None = None
    batch_size: int = 16
    max_length: int | None = None  # None: the length the model was trained with
    # None: the style the model was trained in. Set it to score a model that was never trained as
    # a decision model, such as the base model, in the style the trained one will use.
    prompt: PromptStyle | None = None
    precision: str = "auto"
    trust_remote_code: bool = False


def load_for_scoring(p: ScoreParams) -> tuple[Any, Any, PromptStyle, int | None, dict[str, Any]]:
    meta = read_meta(p.model)
    style = p.prompt or PromptStyle.model_validate(meta.get("prompt") or {})
    tokenizer = load_tokenizer(p.model, p.trust_remote_code)
    model = load_model(p.model, p.precision, p.trust_remote_code)
    return model, tokenizer, style, p.max_length or meta.get("max_length"), meta


def read_calibration(path: str | None, meta: dict[str, Any]) -> dict[str, Any] | None:
    """The calibration to apply: a `calibration.json`, else what the model directory carries."""
    if path:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    return meta.get("calibration")


class CalibrateParams(ScoreParams):
    """`data` should hold sources the model was not trained on (`${data.heldout_calibration}`). A
    temperature fitted on a held-out split of the training sources describes those sources; on
    new tasks the same model is more over-confident, and that is where it will be used."""

    min_examples: int = 50  # a question type, or an option-count band, with fewer rows gets no temperature of its own
    min_temperature: float = 0.5
    max_temperature: float = 5.0
    bands: bool = True  # also fit one temperature per option-count band (2, 3-5, 6-10, 11+) of a type


@register("decision.calibrate")
class Calibrate(Stage):
    """Fit temperatures per question type and option-count band on records from unseen sources."""

    Params = CalibrateParams

    def run(self, ctx: StageContext) -> StageResult:
        p = self.params
        model, tokenizer, style, max_length, _ = load_for_scoring(p)
        records = load_records(p.data, p.max_samples)
        if not records:
            raise ValueError(f"{p.data}: no calibration records")
        preds = score_records(model, tokenizer, records, style, max_length, p.batch_size)
        calibration = fit_calibration(preds, p.min_examples, p.min_temperature, p.max_temperature)
        if not p.bands:
            calibration["bands"] = {}
        fitted = [*calibration["temperature"].values(), *(t for b in calibration["bands"].values() for t in b.values())]
        at_bound = sum(min(abs(t - p.min_temperature), abs(t - p.max_temperature)) < 1e-3 * t for t in fitted)
        if at_bound:
            ctx.log.warning("calibrate: %d temperature(s) sit on a bound; look at the held-out set", at_bound)
        before, after = summarize(preds), summarize(preds, calibration)
        folded = out_of_fold(preds, p.min_examples, p.min_temperature, p.max_temperature)
        path = write_json(ctx.path("calibration.json"), {**calibration, "n": len(preds)})
        metrics = {
            "calibration/n": float(len(preds)),
            "calibration/ece_before": before["ece"], "calibration/ece_after": after["ece"],
            "calibration/nll_before": before["nll"], "calibration/nll_after": after["nll"],
            "calibration/at_bound": float(at_bound),
            "calibration/bands": float(sum(len(b) for b in calibration["bands"].values())),
            **{f"calibration/temperature_{k}": v for k, v in calibration["temperature"].items()},
        }
        if folded:
            # Fitted on one half, measured on the other: the honest version of the "after" numbers.
            metrics.update({"calibration/nll_out_of_fold": folded["nll"], "calibration/ece_out_of_fold": folded["ece"]})
        return StageResult(artifacts={"calibration": path}, metrics=metrics, notes={"calibration": calibration})


class EvalParams(ScoreParams):
    calibration: str | None = None  # calibration.json; None: what the model directory carries
    blind_probe: bool = True  # also answer with the state hidden
    permutations: int = 0  # also answer under this many reshuffles of every choice's options
    compare_to: str | None = None  # the `predictions` of another decision.eval over the same records


def permutation_report(runs: list[list[dict[str, Any]]]) -> dict[str, float]:
    """How stable the chosen option is when the options are listed in another order. `runs[0]` is
    the given order; answers are mapped back to the record's own option order before comparing."""
    picks = []
    for preds in runs:
        picks.append([
            p["order"][max(range(len(p["scores"])), key=p["scores"].__getitem__)] if p["type"] == CHOICE else None
            for p in preds
        ])
    gold = [
        p["order"][max(range(len(p["target"])), key=p["target"].__getitem__)] if p["type"] == CHOICE else None
        for p in runs[0]
    ]
    rows = [i for i, g in enumerate(gold) if g is not None]
    if not rows:
        return {}
    same = sum(len({run[i] for run in picks}) == 1 for i in rows) / len(rows)
    accuracy = sum(run[i] == gold[i] for run in picks for i in rows) / (len(rows) * len(picks))
    return {"permutation_consistency": same, "permutation_accuracy": accuracy}


@register("decision.eval")
class Eval(Stage):
    """Accuracy with its interval, proper scores per source, a paired comparison, and two shortcut probes."""

    Params = EvalParams

    def run(self, ctx: StageContext) -> StageResult:
        p = self.params
        model, tokenizer, style, max_length, meta = load_for_scoring(p)
        records = load_records(p.data, p.max_samples)
        if not records:
            raise ValueError(f"{p.data}: no eval records")
        calibration = read_calibration(p.calibration, meta)
        preds = score_records(model, tokenizer, records, style, max_length, p.batch_size)
        overall = summarize(preds, calibration)
        items = item_scores(preds, calibration)
        low, high = bootstrap_interval([v["correct"] for v in items.values()])
        overall.update({
            "accuracy_ci_low": low, "accuracy_ci_high": high,
            # Every source counts once, corrected for the chance level of its questions.
            "macro_skill": macro_skill(preds, calibration),
            "confident_errors": confident_errors(preds, calibration),
        })
        metrics = {f"eval/{k}": v for k, v in overall.items()}
        sources = by_source(preds, calibration)
        for name, group in sources.items():
            group["skill"] = skill([q for q in preds if q.get("source", "default") == name], calibration)
        report: dict[str, Any] = {"overall": overall, "by_source": sources}
        if calibration:
            raw = summarize(preds)
            metrics.update({"eval/nll_uncalibrated": raw["nll"], "eval/ece_uncalibrated": raw["ece"]})
            report["calibration"] = calibration
        if p.compare_to:
            others = {str(o["id"]): {"correct": o["correct"], "nll": o["nll"]} for o in read_jsonl(p.compare_to)}
            diff = paired_difference(items, others)
            if not diff["n"]:
                raise ValueError(f"{p.compare_to} shares no record with this evaluation; compare runs over the same data")
            metrics.update({f"eval/{k}": v for k, v in diff.items() if k != "n"})
            metrics["eval/delta_n"] = diff["n"]
            report["compared_to"] = {"path": p.compare_to, **diff}
        if p.blind_probe:
            # A model that reads the state should fall to the answer prior when the state is hidden.
            # A small gap means the questions, not the states, carry the answers.
            blind = summarize(score_records(model, tokenizer, records, style, max_length, p.batch_size, blind=True))
            metrics["eval/blind_accuracy"] = blind["accuracy"]
            metrics["eval/blind_gap"] = overall["accuracy"] - blind["accuracy"]
            report["blind"] = blind
        if p.permutations > 0:
            runs = [preds] + [
                score_records(model, tokenizer, records, style, max_length, p.batch_size, permute_seed=k)
                for k in range(p.permutations)
            ]
            stability = permutation_report(runs)
            metrics.update({f"eval/{k}": v for k, v in stability.items()})
            report["permutations"] = stability
        rows = [{**pred, "probabilities": probabilities(pred, calibration), **items[str(pred["id"])]} for pred in preds]
        artifacts = {
            "report": write_json(ctx.path("report.json"), report),
            "predictions": ctx.path("predictions.jsonl"),
        }
        write_jsonl(artifacts["predictions"], rows)
        return StageResult(artifacts=artifacts, metrics=metrics, notes={"by_source": report["by_source"]})
