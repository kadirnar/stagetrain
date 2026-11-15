"""`history.jsonl`: what a training stage logged, step by step, under the same metric names for
every track. `stagetrain report` draws its curves from it.

A row is `{"step": 120, "epoch": 0.4, "train/loss": 1.31, "train/learning_rate": 9e-5}`; an
evaluation adds a row such as `{"step": 200, "eval/loss": 1.2, "eval/accuracy": 0.81}`.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

HISTORY = "history.jsonl"
_SKIP_SUFFIXES = ("_runtime", "_per_second", "_time", "_flos")


def normalize_log(entry: Mapping[str, Any]) -> dict[str, Any] | None:
    """One entry of a `Trainer`'s `log_history` as a history row, or `None` for the summary entry
    that closes a run and for entries without a step."""
    if "step" not in entry or "train_runtime" in entry:
        return None
    row: dict[str, Any] = {"step": int(entry["step"])}
    if isinstance(entry.get("epoch"), (int, float)):
        row["epoch"] = float(entry["epoch"])
    for key, value in entry.items():
        if key in ("step", "epoch") or isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if not math.isfinite(value) or key.endswith(_SKIP_SUFFIXES):
            continue
        if key.startswith("eval_"):
            row[f"eval/{key[len('eval_'):]}"] = float(value)
        elif key == "loss":
            row["train/loss"] = float(value)
        else:
            row[f"train/{key}"] = float(value)
    return row if len(row) > 1 + ("epoch" in row) else None


def save_history(stage_dir: str | Path, log_history: Iterable[Mapping[str, Any]]) -> Path:
    """Write a finished run's whole log; replaces what `history_callback` wrote while it ran."""
    path = Path(stage_dir) / HISTORY
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [row for row in map(normalize_log, log_history) if row is not None]
    tmp = path.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    tmp.replace(path)
    return path


def append_history(stage_dir: str | Path, row: Mapping[str, Any]) -> None:
    path = Path(stage_dir) / HISTORY
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(row)) + "\n")


def load_history(stage_dir: str | Path) -> list[dict[str, Any]]:
    path = Path(stage_dir) / HISTORY
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue  # a line cut short by an interruption
        if isinstance(row, dict) and isinstance(row.get("step"), int):
            rows.append(row)
    return rows


def series(rows: Iterable[Mapping[str, Any]], max_points: int = 1500) -> dict[str, list[list[float]]]:
    """`{metric: [[step, value], ...]}` in step order. A step logged twice (an interrupted run
    that was resumed) keeps its last value; long series are thinned evenly, keeping both ends."""
    by_metric: dict[str, dict[int, float]] = {}
    for row in rows:
        for key, value in row.items():
            if key in ("step", "epoch") or not isinstance(value, (int, float)):
                continue
            by_metric.setdefault(key, {})[row["step"]] = float(value)
    out = {}
    for key, points in by_metric.items():
        ordered = sorted(points.items())
        if len(ordered) > max_points:
            stride = math.ceil(len(ordered) / max_points)
            ordered = ordered[:-1:stride] + [ordered[-1]]
        out[key] = [[step, value] for step, value in ordered]
    return out


def history_callback(stage_dir: str | Path):
    """A `transformers.TrainerCallback` that appends every log to `history.jsonl` as it happens,
    so the report can be drawn while the stage is still training."""
    from transformers import TrainerCallback

    class HistoryCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if not state.is_world_process_zero or not logs:
                return
            row = normalize_log({"step": state.global_step, "epoch": state.epoch, **logs})
            if row is not None:
                append_history(stage_dir, row)

    return HistoryCallback()
