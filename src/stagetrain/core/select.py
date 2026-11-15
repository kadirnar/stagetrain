"""`<track>.select`: the best of a matrix's runs, by one metric.

The selecting stage hands on the winner's artifacts under the same names, so a later stage
writes `${best.model}` without knowing which run won.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Literal

from pydantic import Field

from .manifest import Manifest
from .stage import _REGISTRY, Stage, StageContext, StageParams, StageResult, register

# Lower is better for these; every other metric is read as higher is better.
_LOWER = ("loss", "nll", "ece", "brier", "perplexity", "mae", "error", "kl", "wer", "cer", "no_answer")


def lower_is_better(metric: str) -> bool:
    name = metric.rsplit("/", 1)[-1].lower()
    return any(name == word or name.startswith(f"{word}_") or name.endswith(f"_{word}") for word in _LOWER)


class SelectParams(StageParams):
    group: str  # the name of a stage with a `matrix`
    metric: str  # a metric every run reports, e.g. eval/accuracy
    mode: Literal["max", "min"] | None = None  # None: `min` for losses and errors, `max` otherwise
    # Matrix keys that are repeats, not settings: usually ["seed"]. Runs that differ only in these
    # are one configuration, judged by its mean. Without this the winner is the best single run,
    # which rewards a lucky seed as much as a good setting.
    average_over: list[str] = Field(default_factory=list)


def group_runs(upstream: Mapping[str, Manifest], group: str) -> dict[str, Manifest]:
    """The runs of a matrix among a stage's upstream manifests, in run order."""
    runs = {name: m for name, m in upstream.items() if name.startswith(f"{group}-") and m.variant}
    return dict(sorted(runs.items(), key=lambda item: int(item[0].rsplit("-", 1)[-1])))


class SelectBest(Stage):
    """Pick the best configuration of a matrix by a metric and hand on one of its runs' artifacts."""

    Params = SelectParams

    def run(self, ctx: StageContext) -> StageResult:
        p: SelectParams = self.params
        runs = group_runs(ctx.upstream, p.group)
        if not runs:
            raise ValueError(f"select: {p.group!r} is not a matrix this stage depends on; it sees {sorted(ctx.upstream)}")
        missing = sorted(name for name, m in runs.items() if p.metric not in m.metrics)
        if missing:
            reported = sorted(next(iter(runs.values())).metrics)
            raise ValueError(f"select: {missing} do not report {p.metric!r}; reported: {reported}")
        unknown = [k for k in p.average_over if not any(k in m.variant for m in runs.values())]
        if unknown:
            raise ValueError(f"select: average_over names {unknown}, which the matrix does not vary")
        minimise = p.mode == "min" if p.mode else lower_is_better(p.metric)

        # One configuration per combination of the settings that are not averaged over.
        configs: dict[str, list[str]] = {}
        for name, m in runs.items():
            setting = {k: v for k, v in m.variant.items() if k not in p.average_over}
            configs.setdefault(json.dumps(setting, sort_keys=True, default=str), []).append(name)
        table = []
        for key, names in configs.items():
            values = [runs[n].metrics[p.metric] for n in names]
            mean = sum(values) / len(values)
            std = (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5 if len(values) > 1 else 0.0
            table.append({"setting": json.loads(key), "runs": names, "values": values, "mean": mean, "std": std})
        table.sort(key=lambda row: row["mean"], reverse=not minimise)
        best = table[0]
        # The run handed on is the configuration's most typical one, not its luckiest.
        winner = min(best["runs"], key=lambda n: (abs(runs[n].metrics[p.metric] - best["mean"]), n))
        chosen = runs[winner]
        ctx.log.info(
            "select: %s of %s: %s mean %.6g over %d run(s) %s", winner, p.group, p.metric, best["mean"],
            len(best["runs"]), best["setting"],
        )
        means = [row["mean"] for row in table]
        metrics = dict(chosen.metrics)
        metrics.update({
            "select/runs": float(len(runs)),
            "select/configurations": float(len(table)),
            "select/best_run": float(winner.rsplit("-", 1)[-1]),
            "select/mean": best["mean"],
            "select/std": best["std"],  # across the averaged runs: differences smaller than this are noise
            "select/spread": max(means) - min(means),  # how much the settings matter
        })
        artifacts = {key: chosen.artifact(key, ctx.run_dir) for key in chosen.artifacts}
        return StageResult(artifacts=artifacts, metrics=metrics, notes={
            "winner": winner, "variant": chosen.variant, "setting": best["setting"], "winner_runs": best["runs"],
            "metric": p.metric, "mode": "min" if minimise else "max", "configurations": table,
            "runs": [{"stage": n, "variant": runs[n].variant, p.metric: runs[n].metrics[p.metric]} for n in runs],
        })


def register_select(track: str) -> type[Stage]:
    """Make `<track>.select` available; every track registers its own. Calling it again is a no-op."""
    existing = _REGISTRY.get(f"{track}.select")
    if existing is not None and issubclass(existing, SelectBest):
        return existing
    return register(f"{track}.select")(type("Select", (SelectBest,), {"__doc__": SelectBest.__doc__}))
