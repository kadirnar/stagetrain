"""The prompt a decision is read from, and the verbalizer that names its options.

The model sees the state and the question once, through the tokenizer's own chat template, and
the prompt ends at the answer slot: the position right after the assistant header. Nothing is
generated. Every option owns a few single tokens (its code, with and without a leading space),
and the answer is a softmax over the options' logits at that one position.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .schema import NOUL, SCORE, Question

YES_FORMS = ("yes", "Yes", "YES")
NO_FORMS = ("no", "No", "NO")
ELLIPSIS = "\n[...]\n"


class PromptStyle(BaseModel):
    """How a state and a question are written. A trained model keeps its style in `decision.json`,
    and every later stage reads the model with the style it was trained in."""

    model_config = ConfigDict(extra="forbid")

    system: str | None = None
    state_format: Literal["json", "sections"] = "json"
    option_format: Literal["name_desc", "desc"] = "name_desc"
    question_header: str = "QUESTION:"
    answer_prefix: str = ""  # text between the assistant header and the answer slot
    template_kwargs: dict[str, Any] = Field(default_factory=dict)  # e.g. {enable_thinking: false}


# --------------------------------------------------------------------------- #
# verbalizer
# --------------------------------------------------------------------------- #


class Verbalizer:
    """Gives every option of a question a display code and the token ids that stand for it."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self._choice: dict[int, tuple[list[str], list[list[int]]]] = {}
        self._fixed: dict[tuple[str, ...], list[int]] = {}

    def _single(self, text: str) -> int | None:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        return ids[0] if len(ids) == 1 else None

    def _forms(self, surfaces: tuple[str, ...]) -> list[int]:
        """The distinct single tokens that write any of `surfaces`, bare or after a space."""
        if surfaces not in self._fixed:
            ids: list[int] = []
            for s in surfaces:
                for text in (s, f" {s}"):
                    tid = self._single(text)
                    if tid is not None and tid not in ids:
                        ids.append(tid)
            self._fixed[surfaces] = ids
        return self._fixed[surfaces]

    def _choice_codes(self, n: int) -> tuple[list[str], list[list[int]]]:
        """`n` codes whose token groups do not overlap. Codes are positional (A, B, C, ...), so
        every question with `n` options shares them and shuffling options moves the answer."""
        if n not in self._choice:
            letters = [chr(ord("A") + i) for i in range(26)]
            pool = letters if n <= 26 else [f"{i:02d}" for i in range(100)] + [a + b for a in letters for b in letters]
            codes: list[str] = []
            groups: list[list[int]] = []
            used: set[int] = set()
            for code in pool:
                ids = [i for i in self._forms((code,)) if i not in used]
                if not ids:
                    continue
                codes.append(code)
                groups.append(ids)
                used.update(ids)
                if len(codes) == n:
                    break
            if len(codes) < n:
                raise RuntimeError(f"the tokenizer has single-token codes for {len(codes)} options, {n} needed")
            self._choice[n] = (codes, groups)
        return self._choice[n]

    def groups(self, question: Question) -> tuple[list[str], list[list[int]]]:
        """`(codes, token id groups)`, one per option, in the question's order."""
        if question.type == NOUL:
            yes, no = self._forms(YES_FORMS), self._forms(NO_FORMS)
            no = [i for i in no if i not in yes]
            if not yes or not no:
                raise RuntimeError("the tokenizer has no single-token yes / no")
            return ["yes", "no"], [yes, no]
        if question.type == SCORE:
            groups = [self._forms((str(i),)) for i in range(len(question.names))]
            if any(not g for g in groups):
                raise RuntimeError("the tokenizer has no single-token digits for a score question")
            return [str(i) for i in range(len(groups))], groups
        return self._choice_codes(len(question.names))


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #


def render_state(state: Any, fmt: str = "json") -> str:
    if state is None:
        return ""
    if isinstance(state, str):
        return state
    if fmt == "sections" and isinstance(state, dict):
        parts = []
        for key, value in state.items():
            text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
            parts.append(f"[{key}]\n{text}")
        return "\n\n".join(parts)
    return json.dumps(state, ensure_ascii=False, indent=2)


def render_question(question: Question, codes: list[str], option_format: str = "name_desc") -> str:
    if question.type == NOUL:
        lines = ""
        if any(question.descriptions):
            lines = f"\nyes: {question.descriptions[0]}\nno: {question.descriptions[1]}"
        return f"{question.instructions}{lines}\n\nAnswer yes or no."
    if question.type == SCORE:
        levels = "\n".join(f"{i}: {d}" for i, d in enumerate(question.descriptions))
        return f"{question.instructions}\n\n{levels}\n\nAnswer with one number from 0 to {len(question.names) - 1}."
    lines = []
    for code, name, desc in zip(codes, question.names, question.descriptions, strict=True):
        shown = name.replace("_", " ")
        if not desc:
            lines.append(f"{code}. {shown}")
        elif option_format == "desc" or desc == name:
            lines.append(f"{code}. {desc}")
        else:
            lines.append(f"{code}. {shown}: {desc}")
    return f"{question.instructions}\n\nOptions:\n" + "\n".join(lines) + "\n\nAnswer with the code of one option."


def render_user(state_text: str, question_text: str, style: PromptStyle) -> str:
    if not state_text:
        return question_text
    return f"{state_text}\n\n{style.question_header}\n{question_text}"


def render_prompt(tokenizer, state_text: str, question_text: str, style: PromptStyle) -> str:
    """The whole prompt as text, ending at the answer slot."""
    user = render_user(state_text, question_text, style)
    if getattr(tokenizer, "chat_template", None) is None:
        head = f"{style.system}\n\n" if style.system else ""
        return f"{head}{user}\n\nAnswer:{style.answer_prefix}"
    messages = [{"role": "user", "content": user}]
    if style.system:
        messages.insert(0, {"role": "system", "content": style.system})
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **style.template_kwargs
        )
    except Exception:
        if not style.system:
            raise
        # Some templates have no system role: the system text then leads the user turn.
        messages = [{"role": "user", "content": f"{style.system}\n\n{user}"}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **style.template_kwargs
        )
    return text + style.answer_prefix


def encode_prompt(
    tokenizer, state: Any, question: Question, codes: list[str], style: PromptStyle, max_length: int | None = None
) -> tuple[list[int], bool]:
    """Token ids of the prompt and whether the state was cut to fit `max_length`.

    Only the state is ever cut, from its middle: the question, the options and the answer slot
    are always whole.
    """
    # A chat template writes its own BOS; a tokenizer without one adds it here.
    special = getattr(tokenizer, "chat_template", None) is None
    question_text = render_question(question, codes, style.option_format)
    state_text = render_state(state, style.state_format)
    truncated = False
    for _ in range(6):
        ids = tokenizer.encode(render_prompt(tokenizer, state_text, question_text, style), add_special_tokens=special)
        if max_length is None or len(ids) <= max_length:
            return ids, truncated
        state_ids = tokenizer.encode(state_text, add_special_tokens=False)
        marker = len(tokenizer.encode(ELLIPSIS, add_special_tokens=False))
        keep = len(state_ids) - (len(ids) - max_length) - marker - 4
        if keep <= 0:
            break
        head = keep - keep // 2
        state_text = (
            tokenizer.decode(state_ids[:head], skip_special_tokens=True) + ELLIPSIS
            + tokenizer.decode(state_ids[len(state_ids) - keep // 2:], skip_special_tokens=True)
        )
        truncated = True
    raise ValueError(
        f"the question alone does not fit in max_length={max_length}; raise it or shorten the option descriptions"
    )
