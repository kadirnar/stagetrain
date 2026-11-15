"""Stages `decision.calibrate` and `decision.eval`."""

from __future__ import annotations

from typing import Any

from stagetrain.core import Stage, StageContext, StageParams, StageResult, register, write_json, write_jsonl

from .metrics import by_source, fit_temperatures, probabilities, summarize
from .modeling import load_model, load_records, load_tokenizer, read_meta, score_records
from .prompt import PromptStyle
from .schema import CHOICE


class ScoreParams(StageParams):
    model: str
    data: str
    max_samples: int | None = None
    batch_size: int = 16
    max_length: int | None = None  # None: the length the model was trained with
    precision: str = "auto"
    trust_remote_code: bool = False


def load_for_scoring(p: ScoreParams) -> tuple[Any, Any, PromptStyle, int | None, dict[str, Any]]:
    meta = read_meta(p.model)
    style = PromptStyle.model_validate(meta.get("prompt") or {})
    tokenizer = load_tokenizer(p.model, p.trust_remote_code)
    model = load_model(p.model, p.precision, p.trust_remote_code)
    return model, tokenizer, style, p.max_length or meta.get("max_length"), meta


def read_temperatures(path: str | None, meta: dict[str, Any]) -> dict[str, float] | None:
    if path:
        import json
        from pathlib import Path

        return json.loads(Path(path).read_text(encoding="utf-8"))["temperature"]
    return meta.get("calibration")


class CalibrateParams(ScoreParams):
    min_examples: int = 20  # a question type with fewer held-out examples keeps temperature 1
    min_temperature: float = 0.2
    max_temperature: float = 10.0


@register("decision.calibrate")
class Calibrate(Stage):
    """Fit one temperature per question type on held-out records the model was not trained on."""

    Params = CalibrateParams

    def run(self, ctx: StageContext) -> StageResult:
        p = self.params
        model, tokenizer, style, max_length, _ = load_for_scoring(p)
        records = load_records(p.data, p.max_samples)
        if not records:
            raise ValueError(f"{p.data}: no calibration records")
        preds = score_records(model, tokenizer, records, style, max_length, p.batch_size)
        temperatures = fit_temperatures(preds, p.min_examples, p.min_temperature, p.max_temperature)
        for kind, t in temperatures.items():
            if min(abs(t - p.min_temperature), abs(t - p.max_temperature)) < 1e-3 * t:
                ctx.log.warning("calibrate: the %s temperature sits on its bound (%.3g); check the held-out set", kind, t)
        before, after = summarize(preds), summarize(preds, temperatures)
        path = write_json(ctx.path("calibration.json"), {"temperature": temperatures, "n": len(preds)})
        metrics = {
            "calibration/n": float(len(preds)),
            "calibration/ece_before": before["ece"], "calibration/ece_after": after["ece"],
            "calibration/nll_before": before["nll"], "calibration/nll_after": after["nll"],
            **{f"calibration/temperature_{k}": v for k, v in temperatures.items()},
        }
        return StageResult(artifacts={"calibration": path}, metrics=metrics)


class EvalParams(ScoreParams):
    calibration: str | None = None  # calibration.json; None: what the model directory carries
    blind_probe: bool = True  # also answer with the state hidden
    permutations: int = 0  # also answer under this many reshuffles of every choice's options


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
    """Accuracy, NLL and calibration per source and question type, plus two shortcut probes."""

    Params = EvalParams

    def run(self, ctx: StageContext) -> StageResult:
        p = self.params
        model, tokenizer, style, max_length, meta = load_for_scoring(p)
        records = load_records(p.data, p.max_samples)
        if not records:
            raise ValueError(f"{p.data}: no eval records")
        temperatures = read_temperatures(p.calibration, meta)
        preds = score_records(model, tokenizer, records, style, max_length, p.batch_size)
        overall = summarize(preds, temperatures)
        metrics = {f"eval/{k}": v for k, v in overall.items()}
        report: dict[str, Any] = {"overall": overall, "by_source": by_source(preds, temperatures)}
        if temperatures:
            raw = summarize(preds)
            metrics.update({"eval/nll_uncalibrated": raw["nll"], "eval/ece_uncalibrated": raw["ece"]})
            report["temperature"] = temperatures
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
        rows = [{**pred, "probabilities": probabilities(pred, temperatures)} for pred in preds]
        artifacts = {
            "report": write_json(ctx.path("report.json"), report),
            "predictions": ctx.path("predictions.jsonl"),
        }
        write_jsonl(artifacts["predictions"], rows)
        return StageResult(artifacts=artifacts, metrics=metrics, notes={"by_source": report["by_source"]})
