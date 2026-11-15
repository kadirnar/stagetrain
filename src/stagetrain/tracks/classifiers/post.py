"""Stages that use a trained classifier without updating it: pseudo-labelling, calibration,
evaluation and export."""

from __future__ import annotations

import importlib.util
import shutil
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

from .labels import LABELS_FILE, LabelSpace
from .metrics import CALIBRATION_FILE

CONFIG_FILE = "classifier_config.json"


class InferenceParams(StageParams):
    model: str  # the `model` artifact of classifiers.finetune
    batch_size: int = 64
    max_length: int = 256
    device: str | None = None  # default: cuda, then mps, then cpu
    trust_remote_code: bool = False


def _logits(p: InferenceParams, rows: list[dict[str, Any]]):
    from .modeling import Predictor

    return Predictor(p.model, p.device, p.trust_remote_code).logits(rows, p.batch_size, p.max_length)


# --------------------------------------------------------------------------- #
# pseudo-labelling
# --------------------------------------------------------------------------- #


class PseudoLabelParams(InferenceParams):
    data: str  # JSONL with `text`: the unlabeled pool
    calibration: str | None = None  # calibrate the teacher before thresholding its confidence
    min_confidence: float = 0.9  # single-label: probability of the predicted class
    multi_label_margin: float = 0.4  # multi-label: every label's probability is this far from 0.5
    max_per_class: int | None = None  # keep the most confident rows of each predicted class
    max_samples: int | None = None  # how many pool rows to read
    hard_labels: bool = False  # write the teacher's decision instead of its distribution


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
        labels = LabelSpace.load(p.model)
        rows = []
        for row in read_jsonl(p.data):
            rows.append(row)
            if p.max_samples and len(rows) >= p.max_samples:
                break
        if not rows:
            raise ValueError(f"{p.data}: the unlabeled pool is empty")
        calibration = load_calibration(p.calibration)
        probs = probabilities(_logits(p, rows), labels.problem_type, calibration["temperature"])
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
        }
        return StageResult(
            artifacts={"data": path}, metrics=metrics,
            notes={"kept_per_class": dict(zip(labels.names, per_class, strict=True))},
        )


# --------------------------------------------------------------------------- #
# calibration
# --------------------------------------------------------------------------- #


class CalibrateParams(InferenceParams):
    data: str  # a held-out labelled split the model was not trained on (prepare_data's eval)
    n_bins: int = 15
    fit_thresholds: bool = True  # multi-label only: a decision threshold per label
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
        thresholds = fit_thresholds(after, y) if labels.multi and p.fit_thresholds else None
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
        return StageResult(artifacts={"calibration": path}, metrics=metrics)


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #


class EvalParams(InferenceParams):
    data: str  # labelled JSONL; ideally prepare_data's test split
    calibration: str | None = None
    n_bins: int = 15
    save_predictions: bool = False


@register("classifiers.eval")
class Evaluate(Stage):
    """Measure a classifier on a labelled split: accuracy, F1, calibration, per-class report."""

    Params = EvalParams

    def run(self, ctx: StageContext) -> StageResult:
        from .metrics import (
            classification_report,
            confusion_matrix,
            decide,
            ece,
            load_calibration,
            nll,
            probabilities,
        )
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
        report = classification_report(pred, y, labels.names, kind)

        metrics = {f"eval/{k}": float(v) for k, v in report["metrics"].items()}
        metrics.update({
            "eval/n": float(len(rows)),
            "eval/nll": nll(logits, y, kind, calibration["temperature"]),
            "eval/ece": ece(probs, y, kind, p.n_bins),
            "eval/min_class_f1": min(v["f1"] for v in report["per_class"].values() if v["support"] > 0),
        })
        if p.calibration:
            metrics["eval/ece_uncalibrated"] = ece(probabilities(logits, kind), y, kind, p.n_bins)
        artifacts = {"report": write_json(ctx.path("report.json"), {
            "model": p.model, "data": p.data, "calibration": calibration, **report,
        })}
        if not labels.multi:
            artifacts["confusion_matrix"] = write_json(ctx.path("confusion_matrix.json"), {
                "labels": labels.names, "rows": "true", "columns": "predicted",
                "matrix": confusion_matrix(pred, y, len(labels)),
            })
        if p.save_predictions:
            def name(i: int) -> Any:
                if labels.multi:
                    return [labels.names[c] for c in range(len(labels)) if pred[i][c]]
                return labels.names[int(pred[i])]

            artifacts["predictions"] = ctx.path("predictions.jsonl")
            write_jsonl(artifacts["predictions"], (
                {"text": row["text"], "gold": row.get("labels", row.get("label")), "predicted": name(i),
                 "probs": [round(float(x), 6) for x in probs[i]]}
                for i, row in enumerate(rows)
            ))
        return StageResult(artifacts=artifacts, metrics=metrics)


# --------------------------------------------------------------------------- #
# export
# --------------------------------------------------------------------------- #


class ExportParams(StageParams):
    model: str
    calibration: str | None = None
    max_length: int = 256
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

        p: ExportParams = self.params
        labels = LabelSpace.load(p.model)
        calibration = load_calibration(p.calibration)
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
            "max_length": p.max_length,
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
                    export_onnx(out, out / "model.onnx", p.max_length, p.onnx_opset, p.trust_remote_code)
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
