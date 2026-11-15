"""Pipeline configuration: one YAML file describes one track's multi-stage run.

    track: decision
    name: my-run
    vars: {base_model: org/model}
    defaults:                      # per stage kind, merged under each stage's params
      decision.train: {lora: {r: 16}}
    stages:
      - name: align
        kind: decision.train
        params: {init_from: "${vars.base_model}"}
      - name: full
        kind: decision.train
        params: {init_from: "${align.model}"}     # an artifact of an earlier stage
        gate: {metric: eval/accuracy, min: 0.8}

References:
    ${vars.KEY}          resolved when the file is loaded
    ${run_dir}           the pipeline's run directory, resolved when a stage starts
    ${STAGE.ARTIFACT}    an artifact path of an earlier stage, resolved when a stage starts;
                         it also makes STAGE a dependency

Several runs of one stage:

      - name: sft
        kind: llm.sft
        matrix:                                  # one stage per combination: sft-1 ... sft-6
          train.learning_rate: [1e-4, 3e-4, 1e-3]
          seed: [1, 2]
        params: {...}
      - name: best
        kind: llm.select                         # the run with the best metric; ${best.model}
        params: {group: sft, metric: eval/accuracy}
"""

from __future__ import annotations

import copy
import re
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

TRACKS = ("llm", "decision", "classifiers", "embeddings")
RESERVED = ("vars", "run_dir")
REF = re.compile(r"\$\{([^}]+)\}")
STAGE_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


class ConfigError(ValueError):
    """The configuration file is wrong; the message says where."""


class Gate(BaseModel):
    """A condition a finished stage must meet for the pipeline to go on.

    Absolute: `min` / `max` on one of the stage's metrics. Relative: `baseline` names an earlier
    stage's metric as `<stage>.<metric>`, and the metric may be at most `max_drop` below it (for
    metrics where higher is better) or at most `max_rise` above it (where lower is better).
    """

    model_config = ConfigDict(extra="forbid")

    metric: str
    min: float | None = None
    max: float | None = None
    baseline: str | None = None
    max_drop: float | None = None
    max_rise: float | None = None

    @model_validator(mode="after")
    def _check(self) -> Gate:
        relative = self.max_drop is not None or self.max_rise is not None
        if self.min is None and self.max is None and not relative:
            raise ValueError(f"gate on {self.metric!r} has no bound (min, max, max_drop or max_rise)")
        if relative and self.baseline is None:
            raise ValueError(f"gate on {self.metric!r}: max_drop / max_rise need a baseline")
        if self.baseline is not None:
            if not relative:
                raise ValueError(f"gate on {self.metric!r}: a baseline needs max_drop or max_rise")
            if "." not in self.baseline:
                raise ValueError(f"gate baseline {self.baseline!r} must be '<stage>.<metric>'")
        return self

    @property
    def baseline_stage(self) -> str | None:
        return self.baseline.split(".", 1)[0] if self.baseline else None

    @property
    def baseline_metric(self) -> str | None:
        return self.baseline.split(".", 1)[1] if self.baseline else None


class StageSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    kind: str
    needs: list[str] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)
    gate: list[Gate] = Field(default_factory=list)
    enabled: bool = True
    variant: dict[str, Any] = Field(default_factory=dict)  # set by a matrix: what this run varies

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        if not STAGE_NAME.match(v) or v in RESERVED:
            raise ValueError(f"stage name {v!r} must match {STAGE_NAME.pattern} and not be one of {RESERVED}")
        return v

    @field_validator("gate", mode="before")
    @classmethod
    def _gate_list(cls, v: Any) -> Any:
        return [v] if isinstance(v, dict) else (v or [])

    @field_validator("params", mode="before")
    @classmethod
    def _params(cls, v: Any) -> Any:
        return v or {}


class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    track: Literal["llm", "decision", "classifiers", "embeddings"]
    name: str
    output_dir: str = "runs"
    seed: int = 42
    vars: dict[str, Any] = Field(default_factory=dict)
    defaults: dict[str, dict[str, Any]] = Field(default_factory=dict)
    stages: list[StageSpec]
    groups: dict[str, list[str]] = Field(default_factory=dict)  # matrix name -> its runs; filled on load

    @property
    def run_dir(self) -> Path:
        return Path(self.output_dir) / self.track / self.name

    def stage(self, name: str) -> StageSpec:
        for s in self.stages:
            if s.name == name:
                return s
        raise ConfigError(f"no stage named {name!r}; stages: {[s.name for s in self.stages]}")

    @model_validator(mode="after")
    def _structure(self) -> PipelineConfig:
        self.stages = [s for s in self.stages if s.enabled]
        if not self.stages:
            raise ValueError("the pipeline has no enabled stage")
        names = [s.name for s in self.stages]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(f"duplicate stage names: {dupes}")
        known = set(names)
        for s in self.stages:
            if s.kind.split(".", 1)[0] != self.track or "." not in s.kind:
                raise ValueError(
                    f"stage {s.name!r}: kind {s.kind!r} is not a {self.track!r} stage; "
                    f"tracks are independent, so a pipeline only holds '{self.track}.*' kinds"
                )
            inferred = stage_refs(s.params) | {g.baseline_stage for g in s.gate if g.baseline_stage}
            for dep in sorted(inferred):
                if dep not in s.needs:
                    s.needs.append(dep)
            for dep in s.needs:
                if dep == s.name:
                    raise ValueError(f"stage {s.name!r} depends on itself")
                if dep not in known:
                    raise ValueError(
                        f"stage {s.name!r} refers to {dep!r}, which is not an enabled stage; stages: {names}"
                    )
        topological_order(self.stages)  # raises on a cycle
        return self


# --------------------------------------------------------------------------- #
# references
# --------------------------------------------------------------------------- #


def walk_strings(obj: Any) -> Iterable[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from walk_strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from walk_strings(v)


def stage_refs(params: Any) -> set[str]:
    """Names of the stages whose artifacts `params` refers to."""
    out: set[str] = set()
    for text in walk_strings(params):
        for ref in REF.findall(text):
            head = ref.strip().split(".", 1)[0]
            if head not in RESERVED:
                out.add(head)
    return out


def resolve_refs(obj: Any, lookup: Callable[[str], Any]) -> Any:
    """`obj` with every `${...}` replaced by `lookup(ref)`. `lookup` returns `None` to leave a
    reference as it is. A string that is one whole reference takes the value's own type."""
    if isinstance(obj, dict):
        return {k: resolve_refs(v, lookup) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [resolve_refs(v, lookup) for v in obj]
    if not isinstance(obj, str):
        return obj
    whole = REF.fullmatch(obj)
    if whole:
        value = lookup(whole.group(1).strip())
        return obj if value is None else value

    def sub(m: re.Match) -> str:
        value = lookup(m.group(1).strip())
        return m.group(0) if value is None else str(value)

    return REF.sub(sub, obj)


def deep_merge(base: dict, over: dict) -> dict:
    """`over` on top of `base`; nested mappings merge, everything else is replaced."""
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def topological_order(stages: list[StageSpec]) -> list[StageSpec]:
    """Stages with every dependency before its dependants, otherwise in file order."""
    by_name = {s.name: s for s in stages}
    done: dict[str, StageSpec] = {}
    visiting: list[str] = []

    def visit(s: StageSpec) -> None:
        if s.name in done:
            return
        if s.name in visiting:
            cycle = " -> ".join(visiting[visiting.index(s.name):] + [s.name])
            raise ValueError(f"stages depend on each other in a cycle: {cycle}")
        visiting.append(s.name)
        for dep in s.needs:
            visit(by_name[dep])
        visiting.pop()
        done[s.name] = s

    for s in stages:
        visit(s)
    return list(done.values())


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #


def _set_path(root: dict, dotted: str, value: Any) -> None:
    """`a.b.c = value` in `root`; a first segment naming a stage addresses that stage's params."""
    parts = dotted.split(".")
    stages = {s.get("name"): s for s in root.get("stages") or [] if isinstance(s, dict)}
    if parts[0] in stages:
        node = stages[parts[0]].setdefault("params", {})
        parts = parts[1:]
        if not parts:
            raise ConfigError(f"override {dotted!r} names a stage but no parameter")
    else:
        node = root
    for p in parts[:-1]:
        nxt = node.get(p)
        if not isinstance(nxt, dict):
            nxt = node[p] = {}
        node = nxt
    node[parts[-1]] = value


def parse_overrides(pairs: Iterable[str]) -> dict[str, Any]:
    """`key=value` strings from the command line; values are read as YAML."""
    out: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ConfigError(f"override {pair!r} is not key=value")
        key, raw = pair.split("=", 1)
        out[key.strip()] = yaml.safe_load(raw)
    return out


def set_dotted(params: dict[str, Any], dotted: str, value: Any) -> None:
    node = params
    parts = dotted.split(".")
    for part in parts[:-1]:
        nxt = node.get(part)
        if not isinstance(nxt, dict):
            nxt = node[part] = {}
        node = nxt
    node[parts[-1]] = value


def expand_matrix(stages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    """Every stage with a `matrix` becomes one stage per combination of its values, named
    `<name>-1`, `<name>-2`, ...; a `needs` entry naming the matrix means all of its runs."""
    import itertools

    out: list[dict[str, Any]] = []
    groups: dict[str, list[str]] = {}
    for stage in stages:
        matrix = stage.pop("matrix", None)
        if matrix is None:
            out.append(stage)
            continue
        name = stage.get("name")
        if not isinstance(matrix, dict) or not matrix or not all(isinstance(v, list) and v for v in matrix.values()):
            raise ConfigError(f"stage {name!r}: matrix must map parameter paths to non-empty lists")
        if stage.get("gate"):
            raise ConfigError(f"stage {name!r}: a matrix stage takes no gate; put it on the stage that selects a run")
        if stage.get("enabled", True) is False:
            continue
        keys = list(matrix)
        groups[name] = []
        for i, values in enumerate(itertools.product(*(matrix[k] for k in keys)), start=1):
            variant = dict(zip(keys, values, strict=True))
            params = copy.deepcopy(stage.get("params") or {})
            for key, value in variant.items():
                set_dotted(params, key, value)
            run = {k: v for k, v in stage.items() if k not in ("params", "name")}
            out.append({**run, "name": f"{name}-{i}", "params": params, "variant": variant})
            groups[name].append(f"{name}-{i}")
    for stage in out:
        # A stage that works on a matrix's runs (`group: <matrix name>`) depends on all of them.
        group = (stage.get("params") or {}).get("group")
        needs = list(stage.get("needs") or [])
        selects = str(stage.get("kind", "")).endswith(".select")
        if isinstance(group, str) and (group in groups or selects) and group not in needs:
            needs.append(group)
        expanded: list[str] = []
        for dep in needs:
            expanded.extend(groups.get(dep, [dep]))
        if expanded:
            stage["needs"] = expanded
        for ref in stage_refs(stage.get("params") or {}):
            if ref in groups:
                raise ConfigError(
                    f"stage {stage.get('name')!r} refers to ${{{ref}...}}, but {ref!r} is a matrix of "
                    f"{len(groups[ref])} runs; refer to one run ({groups[ref][0]}) or to a stage that selects one"
                )
    return out, groups


def build_config(raw: dict, overrides: dict[str, Any] | None = None) -> PipelineConfig:
    if not isinstance(raw, dict):
        raise ConfigError("the configuration must be a mapping")
    raw = copy.deepcopy(raw)
    for key, value in (overrides or {}).items():
        _set_path(raw, key, value)

    variables = raw.get("vars") or {}

    def var_lookup(ref: str) -> Any:
        if not ref.startswith("vars."):
            return None
        key = ref[len("vars."):]
        if key not in variables:
            raise ConfigError(f"${{{ref}}} is not defined under vars; defined: {sorted(variables)}")
        return variables[key]

    defaults = raw.get("defaults") or {}
    stages = []
    for s in raw.get("stages") or []:
        if not isinstance(s, dict):
            raise ConfigError(f"a stage must be a mapping, got {s!r}")
        s = dict(s)
        s["params"] = deep_merge(defaults.get(s.get("kind"), {}), s.get("params") or {})
        stages.append(resolve_refs(s, var_lookup))
    raw["stages"], raw["groups"] = expand_matrix(stages)
    try:
        return PipelineConfig.model_validate(raw)
    except ConfigError:
        raise
    except ValueError as e:  # pydantic's ValidationError is a ValueError
        raise ConfigError(str(e)) from e


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> PipelineConfig:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"no configuration file at {path}")
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    try:
        return build_config(raw, overrides)
    except ConfigError as e:
        raise ConfigError(f"{path}: {e}") from e
