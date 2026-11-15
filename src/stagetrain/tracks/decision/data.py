"""Stage `decision.build_data`: ordinary labelled datasets become decision records, split by
content into train / eval / calibration, with an audit of the shortcuts the data offers."""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from stagetrain.core import (
    DataSource,
    Stage,
    StageContext,
    StageParams,
    StageResult,
    register,
    write_json,
    write_jsonl,
)

from .schema import CHOICE, Record, content_key, unit_interval

SPLITS = ("train", "eval", "calibration", "heldout_eval", "heldout_calibration")


class SourceSpec(BaseModel):
    """One dataset and how its rows become questions.

    Columns are read under these names (map them with `data.columns`):
      classification / binary / rating:  `text` (or `state_columns`) and `label`
      multiple_choice:                   `question`, `choices` (a list, or {"text": [...]}), `answer` (index,
                                         letter or text), optional `context`
      records:                           `state`, `question`, `answer` (already decision records)
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    task: Literal["records", "classification", "binary", "rating", "multiple_choice"]
    data: DataSource
    eval_data: DataSource | None = None  # a separate eval split; otherwise `eval_fraction` of `data`
    instructions: str | list[str] | None = None  # several: one is picked per row, so wording is not a cue
    state_columns: dict[str, str] | None = None  # state field -> column, for states that are objects
    # classification: raw label -> option name, as a list (index = integer label) or a mapping
    label_names: list[str] | dict[str, str] | None = None
    descriptions: dict[str, str] = Field(default_factory=dict)  # option name -> description; binary: true / false
    positive: list[str] = Field(default_factory=lambda: ["1", "true", "yes"])  # binary: labels that mean yes
    threshold: float | None = None  # binary, numeric labels: yes when label >= threshold
    levels: list[str] | None = None  # rating: a description per level, lowest first
    offset: int = 0  # rating: level = label - offset
    threshold_soft: bool = False  # binary, labels in [0, 1]: keep the fraction as a soft target [p, 1 - p]
    eval_fraction: float = 0.02
    calibration_fraction: float = 0.02
    max_train: int | None = None
    max_eval: int | None = 2000
    # `heldout`: no row of this source is trained on. Half of its rows go to `heldout_eval`, half
    # to `heldout_calibration`. A model is over-confident on tasks it has not seen, so that is
    # where its temperature has to be fitted and where its accuracy means something.
    role: Literal["train", "heldout"] = "train"
    # Long states with the evidence buried: this share of the training rows gets its state placed
    # among `buried_records` other states of the same source, and the question names which record
    # it is about. Short datasets otherwise never teach the model to find what matters in a long input.
    buried_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    buried_records: int = Field(default=12, ge=2)

    @model_validator(mode="after")
    def _check(self) -> SourceSpec:
        if self.task != "records" and not self.instructions and self.task != "multiple_choice":
            raise ValueError(f"source {self.name!r}: task {self.task!r} needs instructions")
        if self.task == "rating" and not self.levels:
            raise ValueError(f"source {self.name!r}: a rating needs levels")
        if not 0 <= self.eval_fraction + self.calibration_fraction < 1:
            raise ValueError(f"source {self.name!r}: eval_fraction + calibration_fraction must be in [0, 1)")
        return self


class BuildDataParams(StageParams):
    sources: list[SourceSpec]
    # A source whose most frequent answer covers more than this share of its rows is reported.
    majority_warning: float = 0.9


# --------------------------------------------------------------------------- #
# rows -> records
# --------------------------------------------------------------------------- #


def _state(row: Mapping[str, Any], spec: SourceSpec) -> Any:
    if spec.state_columns:
        return {field: row[column] for field, column in spec.state_columns.items()}
    return row["text"]


def _instructions(spec: SourceSpec, key: str) -> str:
    if isinstance(spec.instructions, list):
        return spec.instructions[int(key[:8], 16) % len(spec.instructions)]
    return spec.instructions or "Choose the correct answer."


def _option_name(spec: SourceSpec, raw: Any) -> str:
    names = spec.label_names
    if isinstance(names, list):
        if not 0 <= int(raw) < len(names):
            raise ValueError(f"label {raw!r} is outside the {len(names)} label names")  # e.g. -1 for "no label"
        return names[int(raw)]
    if isinstance(names, dict):
        return names[str(raw)]
    return str(raw)


def _options(spec: SourceSpec, rows: Iterable[Mapping[str, Any]]) -> list[str]:
    """The option list of a classification source, the same for every row."""
    if isinstance(spec.label_names, list):
        return list(spec.label_names)
    if isinstance(spec.label_names, dict):
        return list(dict.fromkeys(spec.label_names.values()))
    return sorted({str(r["label"]) for r in rows})


def _answer_index(answer: Any, choices: list[str]) -> int:
    if isinstance(answer, int) and not isinstance(answer, bool):
        return answer
    text = str(answer).strip()
    if text in choices:
        return choices.index(text)
    if len(text) == 1 and text.isalpha() and ord(text.upper()) - ord("A") < len(choices):
        return ord(text.upper()) - ord("A")
    if text.isdigit():
        return int(text)
    raise ValueError(f"cannot place answer {answer!r} among {len(choices)} choices")


def to_record(row: Mapping[str, Any], spec: SourceSpec, options: list[str] | None = None) -> dict[str, Any]:
    """One source row as a decision record (a plain dict, ready for JSONL)."""
    if spec.task == "records":
        record = Record.parse({**row, "source": spec.name, "id": None})
        return {**record.to_json(), "id": content_key(spec.name, record.state, record.question.to_json())}
    if spec.task == "multiple_choice":
        choices = row["choices"]
        if isinstance(choices, Mapping):  # {"label": [...], "text": [...]}
            choices = choices["text"]
        choices = [str(c) for c in choices]
        state: Any = row["question"]
        if row.get("context"):
            state = {"context": row["context"], "question": row["question"]}
        key = content_key(spec.name, state, choices)
        question = {
            "type": "choice", "instructions": _instructions(spec, key),
            "criteria": {f"opt{i + 1}": c for i, c in enumerate(choices)},
        }
        answer: Any = f"opt{_answer_index(row['answer'], choices) + 1}"
    else:
        state = _state(row, spec)
        key = content_key(spec.name, state)
        instructions = _instructions(spec, key)
        if spec.task == "classification":
            names = options or _options(spec, [row])
            question = {"type": "choice", "instructions": instructions,
                        "criteria": {n: spec.descriptions.get(n) for n in names}}
            answer = _option_name(spec, row["label"])
        elif spec.task == "binary":
            criteria = {k: spec.descriptions[k] for k in ("true", "false") if k in spec.descriptions}
            question = {"type": "noul", "instructions": instructions, "criteria": criteria or None}
            if spec.threshold_soft:
                p_yes = min(max(float(row["label"]), 0.0), 1.0)
                record = Record.parse({
                    "id": key, "source": spec.name, "state": state, "question": question, "target": [p_yes, 1.0 - p_yes],
                }) if 0.0 < p_yes < 1.0 else None
                if record is not None:
                    return record.to_json()
                answer = p_yes >= 0.5
            elif spec.threshold is not None:
                answer = float(row["label"]) >= spec.threshold
            else:
                answer = str(row["label"]).strip().lower() in {p.lower() for p in spec.positive}
        else:  # rating
            question = {"type": "score", "instructions": instructions, "criteria": list(spec.levels or [])}
            answer = int(row["label"]) - spec.offset
    record = Record.parse({"id": key, "source": spec.name, "state": state, "question": question, "answer": answer})
    return record.to_json()


def split_of(key: str, spec: SourceSpec) -> str:
    """`train`, `eval` or `calibration` (for a held-out source: `heldout_eval` or
    `heldout_calibration`), decided by the content hash: a duplicate row always lands in the same
    split, so nothing leaks between them."""
    u = unit_interval(key)
    if spec.role == "heldout":
        return "heldout_eval" if u < 0.5 else "heldout_calibration"
    if spec.eval_data is None and u < spec.eval_fraction:
        return "eval"
    if u < spec.eval_fraction * (spec.eval_data is None) + spec.calibration_fraction:
        return "calibration"
    return "train"


def bury(records: list[dict[str, Any]], fraction: float, n_records: int, seed: int) -> int:
    """Rewrite a share of `records` in place so that each one's state sits among other states of
    the same list, as `{"records": [{"id", "content"}, ...]}`, and its question names the record
    it is about. Returns how many were rewritten. Which rows, which neighbours and which position
    depend only on the row's id and the seed."""
    if fraction <= 0 or len(records) <= n_records:
        return 0
    states = [r["state"] for r in records]
    buried = 0
    for i, record in enumerate(records):
        rng = random.Random(f"{seed}:bury:{record['id']}")
        if rng.random() >= fraction:
            continue
        others = rng.sample([j for j in range(len(records)) if j != i], n_records - 1)
        position = rng.randrange(n_records)
        contents = [states[j] for j in others]
        contents.insert(position, states[i])
        record["state"] = {"records": [{"id": f"r{k + 1}", "content": c} for k, c in enumerate(contents)]}
        question = dict(record["question"])
        question["instructions"] = f"About the record with id r{position + 1}: {question['instructions']}"
        record["question"] = question
        record["id"] = content_key(record["id"], "buried")
        buried += 1
    return buried


# --------------------------------------------------------------------------- #
# audit
# --------------------------------------------------------------------------- #


def audit_source(records: list[dict[str, Any]], majority_warning: float = 0.9) -> dict[str, Any]:
    """What a model could learn from this source without reading the state."""
    if not records:
        return {"n": 0, "warnings": ["no rows"]}
    answers = Counter(str(r["answer"]) for r in records)
    majority = answers.most_common(1)[0][1] / len(records)
    lengths = sorted(len(str(r["state"])) for r in records)
    out: dict[str, Any] = {
        "n": len(records),
        "answers": dict(answers.most_common(20)),
        "majority_baseline": round(majority, 4),
        "state_chars": {"p50": lengths[len(lengths) // 2], "p95": lengths[int(len(lengths) * 0.95)], "max": lengths[-1]},
        "warnings": [],
    }
    if majority > majority_warning:
        out["warnings"].append(
            f"one answer covers {majority:.0%} of the rows: accuracy says little here, read NLL and the blind probe"
        )
    positions = Counter()
    n_choice = 0
    for r in records:
        if r["question"]["type"] == CHOICE:
            names = list(r["question"]["criteria"])
            positions[names.index(r["answer"])] += 1
            n_choice += 1
    if n_choice:
        top_position, top_count = positions.most_common(1)[0]
        n_options = max(len(r["question"]["criteria"]) for r in records if r["question"]["type"] == CHOICE)
        out["answer_position"] = {str(k): v for k, v in sorted(positions.items())}
        if top_count / n_choice > max(2.0 / n_options, majority_warning) and len(answers) > 1:
            out["warnings"].append(
                f"the answer is option {top_position + 1} in {top_count / n_choice:.0%} of the rows: "
                "keep shuffle_options on"
            )
    if sum(1 for r in records if not r["state"]) > 0:
        out["warnings"].append("some states are empty")
    return out


# --------------------------------------------------------------------------- #
# the stage
# --------------------------------------------------------------------------- #


def build_source(spec: SourceSpec, seed: int) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    """`({split: records}, counters)` for one source."""
    from stagetrain.core import load_source

    dataset = load_source(spec.data)
    # A dataset that names its classes (a ClassLabel column) supplies the option names itself.
    declared = getattr(getattr(dataset, "features", {}).get("label"), "names", None)
    if spec.task == "classification" and spec.label_names is None and declared:
        spec = spec.model_copy(update={"label_names": [str(n) for n in declared]})
    rows = list(dataset)
    options = _options(spec, rows) if spec.task == "classification" else None
    splits: dict[str, list[dict[str, Any]]] = {name: [] for name in SPLITS}
    seen: set[str] = set()
    counters = {"duplicates": 0, "leaks_removed": 0, "skipped": 0, "buried": 0}

    held_out: set[str] = set()
    if spec.eval_data is not None:
        for row in load_source(spec.eval_data):
            record = to_record(row, spec, options)
            if record["id"] not in held_out:
                held_out.add(record["id"])
                splits["eval"].append(record)
    first_error: Exception | None = None
    for row in rows:
        try:
            record = to_record(row, spec, options)
        except (KeyError, ValueError, IndexError, TypeError) as e:
            counters["skipped"] += 1
            first_error = first_error or e
            continue
        if record["id"] in held_out:
            counters["leaks_removed"] += 1
            continue
        if record["id"] in seen:
            counters["duplicates"] += 1
            continue
        seen.add(record["id"])
        splits[split_of(record["id"], spec)].append(record)

    if first_error is not None and counters["skipped"] > len(rows) / 2:
        raise ValueError(
            f"source {spec.name!r}: {counters['skipped']} of {len(rows)} rows could not be converted; "
            f"the first error was {type(first_error).__name__}: {first_error}"
        ) from first_error
    rng = random.Random(f"{seed}:{spec.name}")
    limits = {"train": spec.max_train, **{name: spec.max_eval for name in SPLITS if name != "train"}}
    for name, limit in limits.items():
        if limit is not None and len(splits[name]) > limit:
            splits[name] = rng.sample(splits[name], limit)
    counters["buried"] = bury(splits["train"], spec.buried_fraction, spec.buried_records, seed)
    return splits, counters


@register("decision.build_data")
class BuildData(Stage):
    """Convert labelled datasets into decision records and split them by content."""

    Params = BuildDataParams

    def run(self, ctx: StageContext) -> StageResult:
        seed = self.seed(ctx)
        merged: dict[str, list[dict[str, Any]]] = {name: [] for name in SPLITS}
        audit: dict[str, Any] = {}
        totals: Counter = Counter()
        names = [s.name for s in self.params.sources]
        if len(set(names)) != len(names):
            raise ValueError(f"source names must be distinct: {names}")
        for spec in self.params.sources:
            splits, counters = build_source(spec, seed)
            totals.update(counters)
            audited = splits["train"] if spec.role == "train" else splits["heldout_eval"] + splits["heldout_calibration"]
            audit[spec.name] = {**audit_source(audited, self.params.majority_warning), **counters, "role": spec.role}
            for warning in audit[spec.name]["warnings"]:
                ctx.log.warning("source %s: %s", spec.name, warning)
            for name, records in splits.items():
                merged[name].extend(records)
        rng = random.Random(seed)
        paths = {}
        for name, records in merged.items():
            rng.shuffle(records)
            paths[name] = ctx.path(f"{name}.jsonl")
            write_jsonl(paths[name], records)
        if not merged["train"]:
            raise ValueError("no training records were built; check the sources' columns")
        paths["audit"] = write_json(ctx.path("audit.json"), audit)
        baselines = [a["majority_baseline"] for a in audit.values() if a.get("n")]
        metrics = {
            "data/train": len(merged["train"]), "data/eval": len(merged["eval"]),
            "data/calibration": len(merged["calibration"]), "data/sources": len(names),
            "data/heldout_sources": sum(s.role == "heldout" for s in self.params.sources),
            "data/heldout_eval": len(merged["heldout_eval"]),
            "data/heldout_calibration": len(merged["heldout_calibration"]), "data/buried": totals["buried"],
            "data/duplicates": totals["duplicates"], "data/leaks_removed": totals["leaks_removed"],
            "data/skipped": totals["skipped"], "data/max_majority_baseline": max(baselines, default=0.0),
            "data/warnings": sum(len(a["warnings"]) for a in audit.values()),
        }
        return StageResult(artifacts=paths, metrics=metrics, notes={"audit": audit})
