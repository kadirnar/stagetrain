"""Use an exported classifier: label names and calibrated probabilities for raw texts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .labels import MULTI, LabelSpace
from .metrics import load_calibration
from .post import CONFIG_FILE


class Classifier:
    """The directory written by `classifiers.export` (or a `classifiers.finetune` model).

        clf = Classifier("runs/classifiers/my-run/export/model")
        clf.predict(["refund my order"])
        # single-label: [{"label": "billing", "confidence": 0.93, "probabilities": {...}}]
        # multi-label:  [{"labels": ["joy", "love"], "probabilities": {...}}]
    """

    def __init__(self, model_dir: str | Path, device: str | None = None, trust_remote_code: bool = False):
        from .modeling import Predictor

        model_dir = Path(model_dir)
        config_path = model_dir / CONFIG_FILE
        if config_path.is_file():
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            self.labels = LabelSpace(cfg["labels"], cfg["problem_type"])
            self.temperature, self.thresholds = cfg.get("temperature", 1.0), cfg.get("thresholds")
            self.max_length = cfg.get("max_length", 256)
        else:
            self.labels = LabelSpace.load(model_dir)
            calibration = load_calibration(model_dir if (model_dir / "calibration.json").is_file() else None)
            self.temperature, self.thresholds, self.max_length = calibration["temperature"], calibration["thresholds"], 256
        self.predictor = Predictor(model_dir, device, trust_remote_code)

    def predict(self, texts: list[str], batch_size: int = 64) -> list[dict[str, Any]]:
        from .metrics import decide, probabilities

        logits = self.predictor.logits([{"text": t} for t in texts], batch_size, self.max_length)
        probs = probabilities(logits, self.labels.problem_type, self.temperature)
        pred = decide(probs, self.labels.problem_type, self.thresholds)
        out = []
        for i in range(len(texts)):
            dist = {name: float(probs[i][c]) for c, name in enumerate(self.labels.names)}
            if self.labels.problem_type == MULTI:
                out.append({"labels": [n for c, n in enumerate(self.labels.names) if pred[i][c]], "probabilities": dist})
            else:
                name = self.labels.names[int(pred[i])]
                out.append({"label": name, "confidence": dist[name], "probabilities": dist})
        return out
