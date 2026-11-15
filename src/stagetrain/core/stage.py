"""A stage: one step of a track's pipeline, with typed parameters, artifacts and metrics."""

from __future__ import annotations

import importlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from .manifest import Manifest


class StageParams(BaseModel):
    """Base of every stage's parameters. Unknown keys are an error, so a typo in a config fails
    before anything trains."""

    model_config = ConfigDict(extra="forbid")

    seed: int | None = None


@dataclass
class StageContext:
    track: str
    pipeline: str
    stage: str
    out_dir: Path  # this stage's directory; everything it writes goes here
    run_dir: Path  # the pipeline's directory
    seed: int
    upstream: dict[str, Manifest] = field(default_factory=dict)  # the stages this one needs
    # True when `out_dir` holds the leftovers of an unfinished attempt of this same run (same
    # configuration, same upstream outputs); otherwise the directory starts empty.
    resumed: bool = False
    log: logging.Logger = field(default_factory=lambda: logging.getLogger("stagetrain"))

    def path(self, *parts: str) -> Path:
        """A path under the stage's directory; its parent is created."""
        p = self.out_dir.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p


@dataclass
class StageResult:
    """What a stage hands to the next: named paths and numbers."""

    artifacts: dict[str, str | Path] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    notes: dict[str, Any] = field(default_factory=dict)


class Stage(ABC):
    """Subclass, set `Params`, implement `run`, and register:

        @register("decision.train")
        class Train(Stage):
            Params = TrainParams
            def run(self, ctx): ...

    Import torch, transformers and other heavy libraries inside `run`, so that planning and
    validating a pipeline needs none of them.
    """

    kind: ClassVar[str] = ""
    Params: ClassVar[type[StageParams]] = StageParams

    def __init__(self, params: StageParams):
        self.params = params

    def seed(self, ctx: StageContext) -> int:
        return self.params.seed if self.params.seed is not None else ctx.seed

    @abstractmethod
    def run(self, ctx: StageContext) -> StageResult: ...


_REGISTRY: dict[str, type[Stage]] = {}


def register(kind: str):
    def wrap(cls: type[Stage]) -> type[Stage]:
        if kind in _REGISTRY and _REGISTRY[kind] is not cls:
            raise ValueError(f"stage kind {kind!r} is registered twice")
        cls.kind = kind
        _REGISTRY[kind] = cls
        return cls

    return wrap


def load_track(track: str) -> None:
    """Import a track's package, which registers its stages."""
    importlib.import_module(f"stagetrain.tracks.{track}")


def get_stage_class(kind: str) -> type[Stage]:
    if kind not in _REGISTRY:
        try:
            load_track(kind.split(".", 1)[0])
        except ModuleNotFoundError as e:
            if e.name and e.name.startswith("stagetrain.tracks"):
                raise KeyError(f"unknown stage kind {kind!r}: no such track") from e
            raise
    if kind not in _REGISTRY:
        track = kind.split(".", 1)[0]
        raise KeyError(f"unknown stage kind {kind!r}; {track} has: {sorted(kinds(track))}")
    return _REGISTRY[kind]


def kinds(track: str | None = None) -> dict[str, type[Stage]]:
    if track is not None:
        try:
            load_track(track)
        except ModuleNotFoundError:
            pass
    return {k: v for k, v in sorted(_REGISTRY.items()) if track is None or k.split(".", 1)[0] == track}
