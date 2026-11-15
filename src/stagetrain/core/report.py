"""`report.html`: one self-contained page per pipeline run: every stage's state, gates and
metrics, the training curves of each training stage, and how each metric moves across the stages.

The page carries its data as JSON and draws it with the small script in `report_template.html`;
it needs no network and no library.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .history import load_history, series
from .pipeline import CACHED, GATE_FAILED, Pipeline
from .select import lower_is_better

REPORT = "report.html"
TEMPLATE = Path(__file__).with_name("report_template.html")
MAX_CURVES = 8  # per stage, after the loss and the learning rate

def is_count(metric: str) -> bool:
    """Sizes (`eval/n`, `eval/rows`) describe the data, and error bars (`eval/score_se`,
    `eval/delta_ci_low`) describe another metric: neither is a quantity to chart on its own."""
    name = metric.rsplit("/", 1)[-1].lower()
    return name in ("n", "rows", "samples", "examples", "count", "generated") or name.endswith(("_se", "_ci_low", "_ci_high", "_n"))


def stage_curves(history: dict[str, list[list[float]]]) -> list[dict[str, Any]]:
    """The charts of one training stage. Train and eval loss share a chart, because they share a
    unit; every other metric gets its own, so no chart has two scales."""
    charts: list[dict[str, Any]] = []
    loss = [(name, label) for name, label in (("train/loss", "train"), ("eval/loss", "eval")) if name in history]
    if loss:
        charts.append({
            "title": "Loss", "x": "step",
            "series": [{"name": label, "metric": name, "points": history[name]} for name, label in loss],
        })
    rest = [m for m in history if m not in ("train/loss", "eval/loss", "train/learning_rate") and not is_count(m)]
    # What a run is judged by first: rewards and eval metrics, then training diagnostics.
    rest.sort(key=lambda m: (not m.startswith(("eval/", "train/reward")), m))
    for metric in rest[:MAX_CURVES]:
        charts.append({
            "title": metric, "x": "step",
            "series": [{"name": metric, "metric": metric, "points": history[metric]}],
        })
    if "train/learning_rate" in history:
        charts.append({
            "title": "Learning rate", "x": "step",
            "series": [{"name": "learning rate", "metric": "train/learning_rate",
                        "points": history["train/learning_rate"]}],
        })
    return charts


def report_data(pipeline: Pipeline) -> dict[str, Any]:
    stages = []
    across: dict[str, list[dict[str, Any]]] = {}
    total_duration = 0.0
    gates_passed = gates_failed = 0
    for status in pipeline.status():
        manifest = pipeline.manifest(status.name)
        spec = pipeline.config.stage(status.name)
        gates = pipeline._gate_results(spec, manifest) if manifest is not None and manifest.completed else []
        history = series(load_history(pipeline.stage_dir(status.name)))
        finished = status.state in (CACHED, GATE_FAILED)
        if manifest is not None and manifest.duration_s:
            total_duration += manifest.duration_s
        gates_passed += sum(g["passed"] for g in gates)
        gates_failed += sum(not g["passed"] for g in gates)
        stages.append({
            "name": status.name, "kind": status.kind, "needs": status.needs, "state": status.state,
            "variant": dict(spec.variant),
            "reason": "" if status.state == CACHED else status.reason,
            "duration_s": manifest.duration_s if manifest else None,
            "finished_at": manifest.finished_at if manifest else None,
            "metrics": dict(sorted(status.metrics.items())) if finished else {},
            "gates": gates,
            "charts": stage_curves(history),
        })
        if finished:
            for metric, value in status.metrics.items():
                if math.isfinite(value):
                    across.setdefault(metric, []).append({"stage": status.name, "value": value})
    comparisons = [
        {"metric": metric, "lower_is_better": lower_is_better(metric), "rows": rows}
        for metric, rows in sorted(across.items())
        if len(rows) >= 2 and metric.startswith("eval/") and not is_count(metric)
    ]
    return {
        "track": pipeline.config.track,
        "name": pipeline.config.name,
        "run_dir": str(pipeline.run_dir),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "summary": {
            "stages": len(stages),
            "finished": sum(s["state"] in (CACHED, GATE_FAILED) for s in stages),
            "duration_s": total_duration,
            "gates_passed": gates_passed,
            "gates_failed": gates_failed,
        },
        "stages": stages,
        "comparisons": comparisons,
    }


def _finite(obj: Any) -> Any:
    """JSON has no NaN or infinity; a diverged loss becomes a gap."""
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite(v) for v in obj]
    return obj


def render_report(data: dict[str, Any]) -> str:
    # "</" cannot appear inside a script element; JSON allows the escaped form.
    payload = json.dumps(_finite(data), ensure_ascii=False, allow_nan=False, default=str).replace("</", "<\\/")
    template = TEMPLATE.read_text(encoding="utf-8")
    title = f"{data['track']} / {data['name']}".replace("&", "&amp;").replace("<", "&lt;")
    return template.replace("__TITLE__", title).replace("__DATA__", payload)


def write_report(pipeline: Pipeline, out: str | Path | None = None) -> Path:
    """Write the report of a run (by default to `<run_dir>/report.html`) and return its path."""
    path = Path(out) if out else pipeline.run_dir / REPORT
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".html.tmp")
    tmp.write_text(render_report(report_data(pipeline)), encoding="utf-8")
    tmp.replace(path)
    return path
