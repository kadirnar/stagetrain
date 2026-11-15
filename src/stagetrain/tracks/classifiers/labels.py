"""Label spaces and data hygiene for classifier training. Pure Python: no torch, no numpy."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SINGLE, MULTI = "single_label", "multi_label"
LABELS_FILE = "labels.json"


@dataclass
class LabelSpace:
    """The classes of one task, in the order of the model's output units."""

    names: list[str]
    problem_type: str = SINGLE
    class_weights: list[float] | None = None  # CE weights (single-label) or BCE pos_weight (multi-label)
    counts: list[int] | None = None

    def __post_init__(self) -> None:
        if self.problem_type not in (SINGLE, MULTI):
            raise ValueError(f"problem_type must be {SINGLE!r} or {MULTI!r}, got {self.problem_type!r}")
        if len(set(self.names)) != len(self.names) or not self.names:
            raise ValueError(f"label names must be unique and not empty: {self.names}")
        self._index = {name: i for i, name in enumerate(self.names)}

    def __len__(self) -> int:
        return len(self.names)

    @property
    def multi(self) -> bool:
        return self.problem_type == MULTI

    def index(self, raw: Any) -> int:
        """A raw label as a class index: by name first, then as an integer position."""
        key = str(raw).strip()
        if key in self._index:
            return self._index[key]
        if isinstance(raw, int) and not isinstance(raw, bool) and 0 <= raw < len(self.names):
            return raw
        raise KeyError(f"unknown label {raw!r}; labels: {self.names}")

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "problem_type": self.problem_type, "labels": self.names,
            "class_weights": self.class_weights, "counts": self.counts,
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> LabelSpace:
        """From a `labels.json`, or from a model directory that holds one."""
        path = Path(path)
        if path.is_dir():
            path = path / LABELS_FILE
        if not path.is_file():
            raise FileNotFoundError(f"no label map at {path}; it is written by classifiers.prepare_data")
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(data["labels"], data.get("problem_type", SINGLE), data.get("class_weights"), data.get("counts"))


# --------------------------------------------------------------------------- #
# raw labels
# --------------------------------------------------------------------------- #


def split_labels(value: Any, separator: str = ",") -> list[Any]:
    """A multi-label cell as a list: a list stays, a string is split on `separator`, None is empty."""
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(separator) if part.strip()]
    if isinstance(value, Iterable):
        return list(value)
    return [value]


def _natural(name: str) -> tuple[int, Any]:
    try:
        return (0, int(name))
    except ValueError:
        return (1, name)


def infer_names(raw_labels: Iterable[Any]) -> list[str]:
    """Sorted distinct label names; names that are all integers sort as numbers."""
    return sorted({str(r).strip() for r in raw_labels}, key=_natural)


# --------------------------------------------------------------------------- #
# hygiene
# --------------------------------------------------------------------------- #


def normalize_text(text: Any) -> str:
    return " ".join(str(text or "").split())


def text_key(row: dict[str, Any]) -> str:
    """What makes two rows the same example: the case-folded, whitespace-normalised text (and pair)."""
    key = normalize_text(row.get("text")).casefold()
    if row.get("text_pair"):
        key += "\x1f" + normalize_text(row["text_pair"]).casefold()
    return key


def _target(row: dict[str, Any]) -> Any:
    return tuple(sorted(row["labels"])) if "labels" in row else row.get("label")


def dedupe(rows: Sequence[dict[str, Any]], drop_conflicts: bool = True) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Rows with one copy of every text. Copies that agree keep the first; a text labelled two
    different ways is dropped entirely when `drop_conflicts`, else its first copy is kept."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[text_key(row)].append(row)
    kept, stats = [], {"duplicates": 0, "conflicts": 0, "empty": 0}
    for key, group in groups.items():
        if not key:
            stats["empty"] += len(group)
            continue
        if len({_target(r) for r in group}) > 1:
            stats["conflicts"] += len(group)
            if drop_conflicts:
                continue
        stats["duplicates"] += len(group) - 1
        kept.append(group[0])
    return kept, stats


def remove_leaks(rows: Sequence[dict[str, Any]], held_out: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """`rows` without the texts that also appear in `held_out`."""
    seen = {text_key(r) for r in held_out}
    kept = [r for r in rows if text_key(r) not in seen]
    return kept, len(rows) - len(kept)


def label_counts(rows: Iterable[dict[str, Any]], n_labels: int) -> list[int]:
    counts = [0] * n_labels
    for row in rows:
        for i in row["labels"] if "labels" in row else [row["label"]]:
            counts[i] += 1
    return counts


def stratum(row: dict[str, Any], counts: Sequence[int]) -> Any:
    """The group a row is split within: its label, or for multi-label its rarest label."""
    if "labels" in row:
        return min(row["labels"], key=lambda i: (counts[i], i)) if row["labels"] else -1
    return row["label"]


def stratified_split(
    rows: Sequence[dict[str, Any]], fraction: float, seed: int, key: Callable[[dict[str, Any]], Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """`(rest, held)`: `fraction` of every stratum goes to `held`, the same rows for the same seed.
    A stratum with at least two rows gives at least one; a stratum of one stays in `rest`."""
    if not 0 <= fraction < 1:
        raise ValueError(f"split fraction must be in [0, 1), got {fraction}")
    if fraction == 0:
        return list(rows), []
    groups: dict[Any, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        groups[key(row)].append(i)
    held: set[int] = set()
    for name in sorted(groups, key=str):
        idx = groups[name]
        random.Random(f"{seed}:{name}").shuffle(idx)
        take = 0 if len(idx) < 2 else min(len(idx) - 1, max(1, round(len(idx) * fraction)))
        held.update(idx[:take])
    return [r for i, r in enumerate(rows) if i not in held], [r for i, r in enumerate(rows) if i in held]


def class_weights(counts: Sequence[int], n_rows: int, problem_type: str, max_weight: float = 10.0) -> list[float]:
    """Balanced weights. Single-label: `n / (classes * count)`, the weight of a class in the
    cross-entropy. Multi-label: `negatives / positives`, the `pos_weight` of a label's BCE. Both are
    clipped to `max_weight`; a class with no example gets `max_weight`."""
    out = []
    for c in counts:
        if c == 0:
            w = max_weight
        elif problem_type == MULTI:
            w = (n_rows - c) / c
        else:
            w = n_rows / (len(counts) * c)
        out.append(round(min(max_weight, w), 6))
    return out


# --------------------------------------------------------------------------- #
# targets
# --------------------------------------------------------------------------- #


def row_target(row: dict[str, Any], n_labels: int, problem_type: str, smoothing: float = 0.0) -> list[float]:
    """The distribution a row is trained towards.

    A row with `probs` (a pseudo-labelled row) uses them as they are. A gold single-label row is
    one-hot, smoothed towards uniform; a gold multi-label row is multi-hot, smoothed towards 0.5.
    """
    if row.get("probs") is not None:
        probs = [float(p) for p in row["probs"]]
        if len(probs) != n_labels:
            raise ValueError(f"a row has {len(probs)} probabilities for {n_labels} labels")
        return probs
    if problem_type == MULTI:
        on = set(row["labels"])
        return [(1.0 if i in on else 0.0) * (1 - smoothing) + 0.5 * smoothing for i in range(n_labels)]
    target = [smoothing / n_labels] * n_labels
    target[row["label"]] += 1.0 - smoothing
    return target


def summarize(counts: Sequence[int]) -> dict[str, float]:
    present = [c for c in counts if c > 0]
    return {
        "min_class_count": float(min(counts)) if counts else 0.0,
        "imbalance_ratio": float(max(present) / min(present)) if present else 0.0,
    }


def source_counts(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(r.get("source", "gold") for r in rows))
