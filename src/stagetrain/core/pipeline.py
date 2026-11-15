"""Runs a track's stages in dependency order, resumes finished ones, and stops at a failed gate."""

from __future__ import annotations

import logging
import shutil
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import Gate, PipelineConfig, StageSpec, resolve_refs, topological_order
from .manifest import COMPLETED, FAILED, RUNNING, Manifest, digest, environment, store_artifacts
from .stage import Stage, StageContext, get_stage_class

log = logging.getLogger("stagetrain")

# What `Pipeline.state` returns.
CACHED, PENDING, STALE, INTERRUPTED, GATE_FAILED = "cached", "pending", "stale", "interrupted", "gate_failed"


class PipelineError(RuntimeError):
    pass


class GateFailed(PipelineError):
    def __init__(self, stage: str, results: list[dict[str, Any]]):
        self.stage, self.results = stage, results
        failed = "; ".join(r["detail"] for r in results if not r["passed"])
        super().__init__(f"stage {stage!r} did not pass its gate: {failed}")


@dataclass
class StageStatus:
    name: str
    kind: str
    needs: list[str]
    state: str
    reason: str = ""
    metrics: dict[str, float] = field(default_factory=dict)


def evaluate_gate(gate: Gate, metrics: dict[str, float], upstream: dict[str, Manifest]) -> dict[str, Any]:
    """One gate against a stage's metrics: `{metric, value, passed, detail}`."""
    out: dict[str, Any] = {"metric": gate.metric, "value": metrics.get(gate.metric), "passed": False}
    value = out["value"]
    if value is None:
        out["detail"] = f"{gate.metric} was not reported (reported: {sorted(metrics)})"
        return out
    problems = []
    if gate.min is not None and value < gate.min:
        problems.append(f"{gate.metric}={value:.6g} is below min {gate.min:.6g}")
    if gate.max is not None and value > gate.max:
        problems.append(f"{gate.metric}={value:.6g} is above max {gate.max:.6g}")
    if gate.baseline is not None:
        base = upstream.get(gate.baseline_stage)
        ref = base.metrics.get(gate.baseline_metric) if base else None
        out["baseline"] = ref
        if ref is None:
            problems.append(f"baseline {gate.baseline} was not reported")
        else:
            if gate.max_drop is not None and value < ref - gate.max_drop:
                problems.append(
                    f"{gate.metric}={value:.6g} dropped more than {gate.max_drop:.6g} below "
                    f"{gate.baseline}={ref:.6g}"
                )
            if gate.max_rise is not None and value > ref + gate.max_rise:
                problems.append(
                    f"{gate.metric}={value:.6g} rose more than {gate.max_rise:.6g} above "
                    f"{gate.baseline}={ref:.6g}"
                )
    out["passed"] = not problems
    out["detail"] = "; ".join(problems) if problems else f"{gate.metric}={value:.6g} ok"
    return out


class Pipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.order: list[StageSpec] = topological_order(config.stages)
        self.run_dir: Path = config.run_dir
        self._fingerprints: dict[str, str] = {}

    # ------------------------------------------------------------------ state

    def stage_dir(self, name: str) -> Path:
        return self.run_dir / name

    def manifest(self, name: str) -> Manifest | None:
        return Manifest.load(self.stage_dir(name))

    def fingerprint(self, name: str) -> str:
        """Hash of everything that decides a stage's output: its kind, parameters and seed, and
        the fingerprints of the stages it needs. Gates are not part of it."""
        if name not in self._fingerprints:
            spec = self.config.stage(name)
            self._fingerprints[name] = digest({
                "kind": spec.kind,
                "params": spec.params,
                "seed": self.config.seed,
                "needs": {dep: self.fingerprint(dep) for dep in sorted(spec.needs)},
            })
        return self._fingerprints[name]

    def state(self, name: str) -> tuple[str, str]:
        """`(state, reason)`; only a `cached` stage is skipped by `run`."""
        spec = self.config.stage(name)
        m = self.manifest(name)
        if m is None:
            return PENDING, "not run yet"
        if m.status == RUNNING:
            return INTERRUPTED, "the last run did not finish"
        if m.status != COMPLETED:
            return m.status, (m.error or "").strip().splitlines()[-1] if m.error else "the last run failed"
        if m.fingerprint != self.fingerprint(name):
            return STALE, "its configuration, or an earlier stage's, changed"
        for dep in spec.needs:
            dep_state, _ = self.state(dep)
            dm = self.manifest(dep)
            if dm is None or dep_state not in (CACHED, GATE_FAILED) or m.upstream.get(dep) != dm.run_id:
                return STALE, f"{dep} was run again or is not finished"
        failed = [r for r in self._gate_results(spec, m) if not r["passed"]]
        if failed:
            return GATE_FAILED, "; ".join(r["detail"] for r in failed)
        return CACHED, "finished with this configuration"

    def status(self) -> list[StageStatus]:
        out = []
        for spec in self.order:
            state, reason = self.state(spec.name)
            m = self.manifest(spec.name)
            out.append(StageStatus(spec.name, spec.kind, list(spec.needs), state, reason, dict(m.metrics) if m else {}))
        return out

    def _gate_results(self, spec: StageSpec, m: Manifest) -> list[dict[str, Any]]:
        upstream = {g.baseline_stage: self.manifest(g.baseline_stage) for g in spec.gate if g.baseline_stage}
        return [evaluate_gate(g, m.metrics, {k: v for k, v in upstream.items() if v}) for g in spec.gate]

    # -------------------------------------------------------------- selection

    def select(self, only: list[str] | None = None, start: str | None = None, until: str | None = None) -> list[str]:
        names = [s.name for s in self.order]
        for n in [*(only or []), *([start] if start else []), *([until] if until else [])]:
            self.config.stage(n)  # raises on an unknown name
        if only:
            return [n for n in names if n in set(only)]
        lo = names.index(start) if start else 0
        hi = names.index(until) + 1 if until else len(names)
        return names[lo:hi]

    # ----------------------------------------------------------------- params

    def resolve_params(self, spec: StageSpec, placeholders: bool = False) -> dict[str, Any]:
        """A stage's parameters with `${run_dir}` and `${stage.artifact}` filled in. With
        `placeholders`, artifacts that do not exist yet become `<stage.artifact>` strings, which
        is enough to validate the parameters before anything has run."""

        def lookup(ref: str) -> Any:
            if ref == "run_dir":
                return str(self.run_dir)
            stage, _, key = ref.partition(".")
            if not key:
                raise PipelineError(
                    f"stage {spec.name!r}: ${{{ref}}} names a stage but no artifact; write ${{{stage}.<artifact>}}"
                )
            m = self.manifest(stage)
            if m is not None and m.completed and key in m.artifacts:
                return m.artifact(key, self.run_dir)
            if placeholders:
                return f"<{ref}>"
            if m is None or not m.completed:
                raise PipelineError(f"stage {spec.name!r} needs {stage!r}, which has not finished")
            raise PipelineError(
                f"stage {spec.name!r}: {stage!r} has no artifact {key!r}; it has: {sorted(m.artifacts)}"
            )

        return resolve_refs(spec.params, lookup)

    def build(self, spec: StageSpec, placeholders: bool = False) -> Stage:
        cls = get_stage_class(spec.kind)
        try:
            params = cls.Params.model_validate(self.resolve_params(spec, placeholders))
        except PipelineError:
            raise
        except ValueError as e:
            raise PipelineError(f"stage {spec.name!r} ({spec.kind}): invalid parameters\n{e}") from e
        return cls(params)

    def validate(self, names: list[str] | None = None) -> None:
        """Check every stage's kind and parameters without running anything."""
        for spec in self.order:
            if names is None or spec.name in names:
                self.build(spec, placeholders=True)

    # -------------------------------------------------------------------- run

    def run(
        self,
        only: list[str] | None = None,
        start: str | None = None,
        until: str | None = None,
        force: bool = False,
        dry_run: bool = False,
        allow_stale: bool = False,
    ) -> list[Manifest]:
        """Run the selected stages in order. A stage that already finished with the same
        configuration and the same upstream runs is skipped unless `force`. Returns the manifests
        of the selected stages; raises `GateFailed` at the first stage whose gate does not hold."""
        selected = self.select(only, start, until)
        self.validate()
        if dry_run:
            for s in self.status():
                mark = "run " if s.name in selected and (force or s.state != CACHED) else "skip"
                log.info("[dry-run] %s %-18s %-28s %s", mark, s.name, s.kind, s.state)
            return []
        out = []
        for spec in self.order:
            if spec.name not in selected:
                continue
            self._check_upstream(spec, allow_stale)
            state, reason = self.state(spec.name)
            if state in (CACHED, GATE_FAILED) and not force:
                m = self.manifest(spec.name)
                log.info("stage %s: skipped (%s)", spec.name, "cached")
            else:
                if state not in (PENDING, CACHED, GATE_FAILED):
                    log.info("stage %s: %s (%s)", spec.name, state, reason)
                m = self._execute(spec)
            results = self._gate_results(spec, m)
            if results != m.gates:
                m.gates = results
                m.save(self.stage_dir(spec.name))
            if any(not r["passed"] for r in results):
                raise GateFailed(spec.name, results)
            out.append(m)
        return out

    def _check_upstream(self, spec: StageSpec, allow_stale: bool) -> None:
        for dep in spec.needs:
            state, reason = self.state(dep)
            if state == CACHED:
                continue
            m = self.manifest(dep)
            if state == STALE and m is not None and m.completed and allow_stale:
                log.warning("stage %s: using stale %s (%s)", spec.name, dep, reason)
                continue
            hint = " (pass --allow-stale to use its old output)" if state == STALE else ""
            raise PipelineError(f"stage {spec.name!r} needs {dep!r}, which is {state}: {reason}{hint}")

    def _execute(self, spec: StageSpec) -> Manifest:
        stage = self.build(spec)
        out_dir = self.stage_dir(spec.name)
        upstream = {dep: self.manifest(dep) for dep in spec.needs}
        # A stage directory may hold what an earlier attempt left behind (checkpoints, partial
        # files). That is only safe to continue from when the attempt was this very run: same
        # configuration, same upstream outputs, and not finished. Anything else is removed, so a
        # changed or forced stage never resumes from the old run's state.
        prev = self.manifest(spec.name)
        resumed = (
            prev is not None and prev.status in (RUNNING, FAILED)
            and prev.fingerprint == self.fingerprint(spec.name)
            and prev.upstream == {dep: dm.run_id for dep, dm in upstream.items()}
        )
        if out_dir.exists() and not resumed:
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ctx = StageContext(
            track=self.config.track, pipeline=self.config.name, stage=spec.name, out_dir=out_dir,
            run_dir=self.run_dir, seed=self.config.seed, upstream=upstream, log=log, resumed=resumed,
        )
        m = Manifest(
            name=spec.name, kind=spec.kind, status=RUNNING, fingerprint=self.fingerprint(spec.name),
            run_id=uuid.uuid4().hex[:12], params=stage.params.model_dump(mode="json"),
            upstream={dep: dm.run_id for dep, dm in upstream.items()},
            started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"), env=environment(),
        )
        m.save(out_dir)
        handler = logging.FileHandler(out_dir / "stage.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        log.addHandler(handler)
        log.info("stage %s: running %s", spec.name, spec.kind)
        t0 = time.monotonic()
        try:
            result = stage.run(ctx)
        except BaseException as e:
            m.status = FAILED
            m.error = "".join(traceback.format_exception_only(type(e), e)).strip()
            raise
        else:
            m.status = COMPLETED
            m.artifacts = store_artifacts(result.artifacts, self.run_dir)
            m.metrics = {k: float(v) for k, v in result.metrics.items()}
            m.notes = result.notes
            log.info("stage %s: finished; metrics %s", spec.name, m.metrics)
        finally:
            m.duration_s = round(time.monotonic() - t0, 3)
            m.finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            m.save(out_dir)
            log.removeHandler(handler)
            handler.close()
        return m
