"""`stage.json`: what a stage ran with and what it produced. The pipeline reads it to resume."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any

MANIFEST = "stage.json"
PACKAGES = (
    "stagetrain", "torch", "transformers", "peft", "datasets", "accelerate", "trl", "unsloth",
    "unsloth_zoo", "sentence-transformers", "scikit-learn", "safetensors",
)

COMPLETED, RUNNING, FAILED, GATE_FAILED = "completed", "running", "failed", "gate_failed"


def digest(obj: Any) -> str:
    """A stable hash of JSON-like data."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def environment() -> dict[str, Any]:
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    env: dict[str, Any] = {"python": sys.version.split()[0], "platform": platform.platform(), "packages": versions}
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=False
        )
        if sha.returncode == 0:
            env["git_sha"] = sha.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return env


@dataclass
class Manifest:
    name: str
    kind: str
    status: str
    fingerprint: str
    run_id: str
    params: dict[str, Any] = field(default_factory=dict)
    variant: dict[str, Any] = field(default_factory=dict)  # what a matrix run varies
    upstream: dict[str, str] = field(default_factory=dict)  # needed stage -> the run_id it was read at
    artifacts: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    notes: dict[str, Any] = field(default_factory=dict)
    gates: list[dict[str, Any]] = field(default_factory=list)
    started_at: str | None = None
    finished_at: str | None = None
    duration_s: float | None = None
    env: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def completed(self) -> bool:
        return self.status == COMPLETED

    def save(self, stage_dir: Path) -> Path:
        stage_dir.mkdir(parents=True, exist_ok=True)
        path = stage_dir / MANIFEST
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2, default=str) + "\n", encoding="utf-8")
        tmp.replace(path)
        return path

    @classmethod
    def load(cls, stage_dir: Path) -> Manifest | None:
        path = stage_dir / MANIFEST
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def artifact(self, key: str, run_dir: Path) -> str:
        """An artifact's path. Paths inside the run directory are stored relative to it, so a
        run directory can be moved; anything else (a Hub id, an absolute path) is kept as it is."""
        if key not in self.artifacts:
            raise KeyError(f"stage {self.name!r} has no artifact {key!r}; it has: {sorted(self.artifacts)}")
        value = self.artifacts[key]
        if value.startswith("run://"):
            return str(run_dir / value[len("run://"):])
        return value


def store_artifacts(artifacts: dict[str, str | Path], run_dir: Path) -> dict[str, str]:
    out = {}
    root = run_dir.resolve()
    for key, value in artifacts.items():
        if isinstance(value, Path) or Path(str(value)).exists():
            try:
                out[key] = "run://" + Path(value).resolve().relative_to(root).as_posix()
                continue
            except ValueError:
                pass
        out[key] = str(value)
    return out
