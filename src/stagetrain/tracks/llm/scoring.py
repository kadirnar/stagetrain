"""Task metrics for generated answers, and the statistics that say whether a difference is real.

A scorer turns `(output, reference, row)` into a number in [0, 1]. The same scorers judge an
evaluation, pick a checkpoint, filter generated training data and pay out a GRPO reward, so every
stage of a pipeline means the same thing by "correct".
"""

from __future__ import annotations

import importlib
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator

THINK_BLOCK = re.compile(r"<think>(.*?)</think>", re.DOTALL)
NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?|-?\.\d+")
CHOICE_PATTERNS = (
    re.compile(r"(?:answer|option|choice)\s*(?:is|:)?\s*\(?([A-Za-z])\)?(?![A-Za-z])", re.IGNORECASE),
    re.compile(r"^\s*\(?([A-Za-z])\)?[\s.:)]*$"),
    re.compile(r"\(([A-Za-z])\)"),
    re.compile(r"(?<![A-Za-z])([A-Z])(?![A-Za-z])[\s.)]*$"),
)
FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)

Scorer = Callable[[str, str, Mapping[str, Any]], float]


class ScorerSpec(BaseModel):
    """How an answer is judged against its reference.

        exact      the answer equals the reference (case and whitespace aside)
        contains   the reference appears in the answer
        numeric    the last number in the answer equals the last number in the reference
        choice     the option letter the answer settles on equals the reference's
        regex      `pattern`'s first group in the answer equals the reference (or the same group in it)
        json       the answer parses as JSON and equals the reference JSON, on `fields` if given
        custom     `function`: "package.module:name", called as f(output, reference, row) -> 0..1
    """

    model_config = ConfigDict(extra="forbid")

    name: Literal["exact", "contains", "numeric", "choice", "regex", "json", "custom"] = "exact"
    pattern: str | None = None
    fields: list[str] | None = None
    function: str | None = None
    tolerance: float = 1e-6  # numeric: relative
    case_sensitive: bool = False
    strip_thinking: bool = True  # judge what follows the reasoning block, not the reasoning

    @model_validator(mode="after")
    def _check(self) -> ScorerSpec:
        if self.name == "regex":
            if not self.pattern:
                raise ValueError("a regex scorer needs a pattern")
            check_pattern(self.pattern)
        if self.name == "custom" and (not self.function or ":" not in self.function):
            raise ValueError('a custom scorer needs function: "package.module:name"')
        return self


def check_pattern(pattern: str) -> None:
    try:
        re.compile(pattern)
    except re.error as e:
        raise ValueError(f"invalid regular expression {pattern!r}: {e}") from e


# --------------------------------------------------------------------------- #
# reasoning
# --------------------------------------------------------------------------- #


def split_reasoning(text: str) -> tuple[str | None, str]:
    """`(reasoning, answer)`. `reasoning` is `None` when the text has no reasoning block. A block
    that was opened and never closed (the generation ran out of tokens) leaves an empty answer."""
    match = THINK_BLOCK.search(text)
    if match:
        return match.group(1).strip(), (text[: match.start()] + text[match.end():]).strip()
    if "<think>" in text:
        return text.split("<think>", 1)[1].strip(), ""
    if "</think>" in text:  # the template opened the block in the prompt
        reasoning, answer = text.split("</think>", 1)
        return reasoning.strip(), answer.strip()
    return None, text.strip()


def has_valid_reasoning(text: str, min_chars: int = 1) -> bool:
    """A closed, non-empty reasoning block followed by an answer."""
    reasoning, answer = split_reasoning(text)
    return reasoning is not None and len(reasoning) >= min_chars and bool(answer)


# --------------------------------------------------------------------------- #
# scorers
# --------------------------------------------------------------------------- #


def normalise(text: str, case_sensitive: bool = False) -> str:
    text = " ".join(str(text).split())
    return text if case_sensitive else text.lower()


def last_number(text: str) -> float | None:
    found = NUMBER.findall(text)
    if not found:
        return None
    try:
        return float(found[-1].replace(",", ""))
    except ValueError:
        return None


def choice_letter(text: str) -> str | None:
    """The option letter an answer settles on: "The answer is (B).", "B", "(b)", "... so C."."""
    text = text.strip()
    for pattern in CHOICE_PATTERNS:
        found = pattern.findall(text)
        if found:
            return found[-1].upper()
    return None


def parse_json(text: str) -> Any:
    """The JSON value in a text: the whole text, a fenced block, or the outermost object/array."""
    candidates = [text.strip(), *(m.strip() for m in FENCE.findall(text))]
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = text.find(opener), text.rfind(closer)
        if 0 <= start < end:
            candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
    raise ValueError("no JSON value found")


def load_function(path: str) -> Callable[..., Any]:
    module, _, name = path.partition(":")
    fn = getattr(importlib.import_module(module), name, None)
    if not callable(fn):
        raise ValueError(f"{path} is not a callable")
    return fn


def build_scorer(spec: ScorerSpec | None = None) -> Scorer:
    spec = spec or ScorerSpec()
    custom = load_function(spec.function) if spec.name == "custom" else None
    pattern = re.compile(spec.pattern, re.DOTALL) if spec.pattern else None

    def captured(text: str) -> str | None:
        found = pattern.findall(text)
        if not found:
            return None
        last = found[-1]
        return last if isinstance(last, str) else last[0]

    def score(output: str, reference: str, row: Mapping[str, Any]) -> float:
        answer = split_reasoning(output)[1] if spec.strip_thinking else output
        if spec.name == "custom":
            return float(min(max(custom(answer, reference, row), 0.0), 1.0))
        if spec.name == "exact":
            return float(normalise(answer, spec.case_sensitive) == normalise(reference, spec.case_sensitive))
        if spec.name == "contains":
            ref = normalise(reference, spec.case_sensitive)
            return float(bool(ref) and ref in normalise(answer, spec.case_sensitive))
        if spec.name == "numeric":
            got, want = last_number(answer), last_number(reference)
            if got is None or want is None:
                return 0.0
            return float(abs(got - want) <= spec.tolerance * max(1.0, abs(want)))
        if spec.name == "choice":
            got, want = choice_letter(answer), choice_letter(reference)
            return float(got is not None and got == want)
        if spec.name == "regex":
            got = captured(answer)
            want = captured(reference)
            if got is None:
                return 0.0
            return float(normalise(got, spec.case_sensitive) == normalise(reference if want is None else want, spec.case_sensitive))
        try:  # json
            got, want = parse_json(answer), parse_json(reference)
        except ValueError:
            return 0.0
        if spec.fields:
            if not isinstance(got, dict) or not isinstance(want, dict):
                return 0.0
            return float(all(got.get(f) == want.get(f) for f in spec.fields))
        return float(got == want)

    return score


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #


def mean_with_error(scores: Sequence[float]) -> dict[str, float]:
    """Mean, its standard error, and a 95 % interval (normal approximation)."""
    n = len(scores)
    if n == 0:
        return {"n": 0.0}
    mean = sum(scores) / n
    variance = sum((s - mean) ** 2 for s in scores) / (n - 1) if n > 1 else 0.0
    se = math.sqrt(variance / n)
    return {"n": float(n), "mean": mean, "se": se, "ci_low": mean - 1.96 * se, "ci_high": mean + 1.96 * se}


def paired_difference(scores: Mapping[str, float], baseline: Mapping[str, float]) -> dict[str, float]:
    """This model minus a baseline, item by item, over the items both were scored on.

    Two models answering the same questions make correlated errors, so the per-item difference
    has a much tighter interval than two independent accuracies would: a small eval set can
    still tell which model is better. `ci_low > 0` means better at the 95 % level.
    """
    shared = sorted(set(scores) & set(baseline))
    if not shared:
        return {"n": 0.0}
    diffs = [scores[k] - baseline[k] for k in shared]
    stats = mean_with_error(diffs)
    return {
        "n": stats["n"], "delta": stats["mean"], "se": stats["se"], "ci_low": stats["ci_low"],
        "ci_high": stats["ci_high"], "wins": float(sum(d > 0 for d in diffs)), "losses": float(sum(d < 0 for d in diffs)),
    }
