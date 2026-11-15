"""Reward functions for `llm.grpo`, built from the same scorers that judge an evaluation.

TRL calls a reward function as `f(prompts=..., completions=..., completion_ids=..., **columns)` and
expects one number per completion; every extra dataset column (here `answer`) arrives as a keyword
argument. With chat prompts a completion is `[{"role": "assistant", "content": ...}]`.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator

from .scoring import (
    ScorerSpec,
    build_scorer,
    check_pattern,
    has_valid_reasoning,
    load_function,
    split_reasoning,
)


class RewardSpec(BaseModel):
    """One term of the reward.

        correct    1 when `scorer` accepts the answer against the reference, else 0
        format     1 when the answer (after any reasoning block) matches `pattern`
        reasoning  1 when the completion has a closed, non-empty reasoning block and an answer
        length     0 up to `max_chars`, falling linearly to -1 at twice that length
        custom     `function`: "package.module:name", called as f(completion, reference, prompt) -> float

    Keep `correct` dominant: a model can collect format and reasoning rewards without solving
    anything. `gated: true` pays a term only on completions the `correct` term accepts, which
    removes that way out: the shaping then only chooses among right answers.

    A `length` penalty is here for completeness; in the one study that isolated it, punishing
    long completions lowered accuracy.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["correct", "format", "reasoning", "length", "custom"] = "correct"
    name: str | None = None  # how the term is logged: train/rewards/<name>/mean
    weight: float = 1.0
    gated: bool = False  # paid only when the answer is correct; needs a `correct` term among the rewards
    scorer: ScorerSpec | None = None
    pattern: str | None = None
    max_chars: int | None = None
    function: str | None = None

    @model_validator(mode="after")
    def _check(self) -> RewardSpec:
        if self.kind == "format":
            if not self.pattern:
                raise ValueError("a format reward needs a pattern")
            check_pattern(self.pattern)
        if self.kind == "length" and not self.max_chars:
            raise ValueError("a length reward needs max_chars")
        if self.kind == "custom" and (not self.function or ":" not in self.function):
            raise ValueError('a custom reward needs function: "package.module:name"')
        return self


def completion_text(completion: Any) -> str:
    """The generated text of one completion, whichever shape TRL hands over."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("content") or "")
    if isinstance(completion, (list, tuple)) and completion:
        return completion_text(completion[-1])
    return ""


def _term(spec: RewardSpec) -> Callable[[str, str, Any], float]:
    if spec.kind == "correct":
        scorer = build_scorer(spec.scorer)
        return lambda text, reference, prompt: scorer(text, reference, {"prompt": prompt})
    if spec.kind == "format":
        pattern = re.compile(spec.pattern, re.DOTALL)
        return lambda text, reference, prompt: float(bool(pattern.search(split_reasoning(text)[1])))
    if spec.kind == "reasoning":
        return lambda text, reference, prompt: float(has_valid_reasoning(text))
    if spec.kind == "length":
        limit = spec.max_chars
        return lambda text, reference, prompt: -min(max(len(text) - limit, 0) / limit, 1.0)
    fn = load_function(spec.function)
    return lambda text, reference, prompt: float(fn(text, reference, prompt))


def build_rewards(specs: Sequence[RewardSpec]) -> tuple[list[Callable[..., list[float]]], list[float]]:
    """`(functions, weights)` for `GRPOTrainer(reward_funcs=...)` and `GRPOConfig(reward_weights=...)`.
    The functions return unweighted values, so the logs show each term on its own scale."""
    if not specs:
        raise ValueError("grpo needs at least one reward")
    names = [s.name or s.kind for s in specs]
    if len(set(names)) != len(names):
        raise ValueError(f"reward names must be distinct, got {names}; set name on the repeated ones")
    correct = next((_term(s) for s in specs if s.kind == "correct"), None)
    if correct is None and any(s.gated for s in specs):
        raise ValueError("a gated reward is paid only on correct answers: add a `correct` reward to gate it on")
    if any(s.gated and s.kind == "correct" for s in specs):
        raise ValueError("the `correct` reward cannot be gated on itself")
    functions = []
    for spec, name in zip(specs, names, strict=True):
        term = _term(spec)

        def reward(prompts=None, completions=None, answer=None, _term=term, _gated=spec.gated, **kwargs) -> list[float]:
            answers = answer if answer is not None else [""] * len(completions)
            prompts = prompts if prompts is not None else [None] * len(completions)
            out = []
            for c, a, q in zip(completions, answers, prompts, strict=True):
                text, reference = completion_text(c), "" if a is None else str(a)
                out.append(0.0 if _gated and correct(text, reference, q) < 1.0 else _term(text, reference, q))
            return out

        reward.__name__ = name  # TRL logs rewards/<__name__>/mean
        functions.append(reward)
    return functions, [s.weight for s in specs]
