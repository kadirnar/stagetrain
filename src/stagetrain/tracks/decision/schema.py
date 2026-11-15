"""Decision records: a state, one typed question over it, and the answer.

A question is written as `{"type": "noul" | "choice" | "score", "instructions": str, "criteria": ...}`:

    noul    yes / no.        criteria: optional {"true": description, "false": description}
    choice  one named option. criteria: {name: description or null}, at least two
    score   an ordered level. criteria: [description of level 0, level 1, ...], two to ten

A record is `{"id", "source", "state", "question", "answer"}` with `answer` a bool, an option
name, or a level index; an optional `target` holds a probability per option (soft labels).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

NOUL, CHOICE, SCORE = "noul", "choice", "score"
TYPES = (NOUL, CHOICE, SCORE)
MAX_LEVELS = 10

_TRUE = {True, "1", "yes", "true", "y"}
_FALSE = {False, "0", "no", "false", "n"}


@dataclass(frozen=True)
class Question:
    type: str
    instructions: str
    names: tuple[str, ...]  # option labels: the choice's names, ("yes", "no"), or ("0", ..., "k-1")
    descriptions: tuple[str | None, ...]

    @classmethod
    def parse(cls, q: Mapping[str, Any] | Question) -> Question:
        if isinstance(q, Question):
            return q
        kind = q.get("type", CHOICE)
        instructions = str(q.get("instructions") or "").strip()
        if not instructions:
            raise ValueError("a question needs instructions")
        criteria = q.get("criteria")
        if kind == NOUL:
            criteria = criteria or {}
            return cls(NOUL, instructions, ("yes", "no"), (criteria.get("true"), criteria.get("false")))
        if kind == SCORE:
            levels = list(criteria or [])
            if not 2 <= len(levels) <= MAX_LEVELS:
                raise ValueError(f"a score question has 2 to {MAX_LEVELS} levels, got {len(levels)}")
            return cls(SCORE, instructions, tuple(str(i) for i in range(len(levels))), tuple(levels))
        if kind == CHOICE:
            if isinstance(criteria, Mapping):
                names, descriptions = list(criteria.keys()), list(criteria.values())
            else:
                names, descriptions = list(criteria or []), [None] * len(criteria or [])
            names = [str(n) for n in names]
            if len(names) < 2 or len(set(names)) != len(names):
                raise ValueError(f"a choice question needs at least two distinct options, got {names}")
            return cls(CHOICE, instructions, tuple(names), tuple(descriptions))
        raise ValueError(f"unknown question type {kind!r}; types: {TYPES}")

    def to_json(self) -> dict[str, Any]:
        if self.type == NOUL:
            criteria: Any = None
            if any(self.descriptions):
                criteria = {"true": self.descriptions[0], "false": self.descriptions[1]}
        elif self.type == SCORE:
            criteria = list(self.descriptions)
        else:
            criteria = dict(zip(self.names, self.descriptions, strict=True))
        return {"type": self.type, "instructions": self.instructions, "criteria": criteria}

    def answer_index(self, answer: Any) -> int:
        """Where `answer` sits among the options."""
        if self.type == NOUL:
            key = answer.strip().lower() if isinstance(answer, str) else answer
            if key in _TRUE:
                return 0
            if key in _FALSE:
                return 1
            raise ValueError(f"a noul answer is a bool or yes/no, got {answer!r}")
        if self.type == SCORE:
            if isinstance(answer, bool) or not isinstance(answer, (int, str)) or not str(answer).isdigit():
                raise ValueError(f"a score answer is a level index, got {answer!r}")
            level = int(answer)
            if level >= len(self.names):
                raise ValueError(f"score level {level} is outside 0..{len(self.names) - 1}")
            return level
        if str(answer) not in self.names:
            raise ValueError(f"answer {answer!r} is not one of the options {list(self.names)}")
        return self.names.index(str(answer))

    def answer_value(self, index: int) -> Any:
        if self.type == NOUL:
            return index == 0
        return index if self.type == SCORE else self.names[index]

    def permuted(self, order: Sequence[int]) -> Question:
        """The options in a new order; only a choice has no inherent order."""
        if self.type != CHOICE:
            raise ValueError(f"only choice options can be reordered, not {self.type}")
        return Question(
            self.type, self.instructions, tuple(self.names[i] for i in order),
            tuple(self.descriptions[i] for i in order),
        )

    def without_descriptions(self) -> Question:
        if self.type != CHOICE:
            return self
        return Question(self.type, self.instructions, self.names, (None,) * len(self.names))


@dataclass(frozen=True)
class Record:
    id: str
    source: str
    state: Any
    question: Question
    target: tuple[float, ...]  # a probability per option, in the question's order

    @classmethod
    def parse(cls, row: Mapping[str, Any]) -> Record:
        question = Question.parse(row["question"])
        n = len(question.names)
        if row.get("target") is not None:
            target = [float(x) for x in row["target"]]
            if len(target) != n or min(target) < 0 or sum(target) <= 0:
                raise ValueError(f"target {target} is not a distribution over {n} options")
            target = [x / sum(target) for x in target]
        elif "answer" in row and row["answer"] is not None:
            index = question.answer_index(row["answer"])
            target = [1.0 if i == index else 0.0 for i in range(n)]
        else:
            raise ValueError("a record needs an answer or a target")
        state = row.get("state")
        return cls(
            id=str(row.get("id") or content_key(row.get("source", ""), state, question.instructions)),
            source=str(row.get("source") or "default"), state=state, question=question, target=tuple(target),
        )

    @property
    def answer_index(self) -> int:
        return max(range(len(self.target)), key=self.target.__getitem__)

    def to_json(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "id": self.id, "source": self.source, "state": self.state, "question": self.question.to_json(),
            "answer": self.question.answer_value(self.answer_index),
        }
        if max(self.target) < 1.0:
            row["target"] = list(self.target)
        return row


def content_key(*parts: Any) -> str:
    """A short stable hash of JSON-like content; equal content gives an equal key."""
    blob = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def unit_interval(key: str) -> float:
    """A hex key as a number in [0, 1); used to split deterministically by content."""
    return int(key[:8], 16) / 16**8
