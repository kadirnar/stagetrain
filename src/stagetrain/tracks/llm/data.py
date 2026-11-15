"""Turning source rows into the three shapes the llm stages train on. No model, no tokenizer.

    chat        {"messages": [{"role": "system" | "user" | "assistant" | "tool", "content": str}, ...]}
    text        {"text": str}                                   raw documents for continued pretraining
    preference  {"prompt": [messages], "chosen": [assistant message], "rejected": [assistant message]}
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable
from typing import Any

ROLES = {
    "system": "system",
    "user": "user", "human": "user",
    "assistant": "assistant", "gpt": "assistant", "bot": "assistant", "model": "assistant",
    "tool": "tool", "function": "tool", "observation": "tool",
}
CHAT_FORMATS = ("messages", "sharegpt", "alpaca", "prompt_completion")


class Rejected(ValueError):
    """A source row that cannot be used; the message is the reason counted in the stats."""


def _text(value: Any, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Rejected(f"empty {what}")
    return value.strip()


def _turns(value: Any, role_key: str, content_key: str) -> list[dict[str, str]]:
    if isinstance(value, str):  # some exports store the list as a JSON string
        try:
            value = json.loads(value)
        except json.JSONDecodeError as e:
            raise Rejected("messages are not a list") from e
    if not isinstance(value, list) or not value:
        raise Rejected("messages are not a list")
    out = []
    for turn in value:
        if not isinstance(turn, dict):
            raise Rejected("a message is not a mapping")
        role = ROLES.get(str(turn.get(role_key, "")).strip().lower())
        if role is None:
            raise Rejected(f"unknown role {turn.get(role_key)!r}")
        out.append({"role": role, "content": _text(turn.get(content_key), f"{role} message")})
    return out


def validate_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """A conversation a chat template can render and a model can learn from: an optional system
    message first, then user and assistant turns that alternate, ending with the assistant. A
    tool message may follow any turn, and the assistant may speak again after it."""
    body = messages[1:] if messages and messages[0]["role"] == "system" else messages
    if not body or body[0]["role"] != "user":
        raise Rejected("conversation does not start with a user message")
    if messages[-1]["role"] != "assistant":
        raise Rejected("conversation does not end with an assistant message")
    last = None
    for m in body:
        if m["role"] == "system":
            raise Rejected("system message is not first")
        if m["role"] == "tool":  # a tool result separates the assistant's call from its answer
            last = "tool"
            continue
        if m["role"] == last:
            raise Rejected("user and assistant turns do not alternate")
        last = m["role"]
    return messages


def normalise_chat(row: dict[str, Any], fmt: str = "messages", system: str | None = None) -> list[dict[str, str]]:
    """One source row as validated chat messages. `system` is prepended when the row has none."""
    if fmt == "messages":
        messages = _turns(row.get("messages"), "role", "content")
    elif fmt == "sharegpt":
        messages = _turns(row.get("conversations"), "from", "value")
    elif fmt == "alpaca":
        instruction = _text(row.get("instruction"), "instruction")
        extra = row.get("input")
        if isinstance(extra, str) and extra.strip():
            instruction = f"{instruction}\n\n{extra.strip()}"
        messages = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": _text(row.get("output"), "output")},
        ]
    elif fmt == "prompt_completion":
        messages = [
            {"role": "user", "content": _text(row.get("prompt"), "prompt")},
            {"role": "assistant", "content": _text(row.get("completion"), "completion")},
        ]
    else:
        raise ValueError(f"unknown chat format {fmt!r}; known: {CHAT_FORMATS}")
    row_system = row.get("system")
    if messages[0]["role"] != "system":
        prefix = row_system if isinstance(row_system, str) and row_system.strip() else system
        if prefix:
            messages = [{"role": "system", "content": prefix.strip()}, *messages]
    return validate_messages(messages)


def normalise_preference(row: dict[str, Any], system: str | None = None) -> dict[str, Any]:
    """A prompt with a preferred and a rejected answer, in TRL's conversational preference format.
    `prompt` is a string or messages; `chosen` and `rejected` are strings or message lists whose
    last assistant turn is the answer."""
    prompt = row.get("prompt")
    if isinstance(prompt, str):
        messages = [{"role": "user", "content": _text(prompt, "prompt")}]
    elif prompt is None and isinstance(row.get("chosen"), list):
        # No prompt column: chosen and rejected are whole conversations that share their beginning.
        messages = _turns(row["chosen"], "role", "content")[:-1]
        if not messages:
            raise Rejected("chosen has no prompt before its answer")
    else:
        messages = _turns(prompt, "role", "content")
    row_system = row.get("system")
    if messages[0]["role"] != "system":
        prefix = row_system if isinstance(row_system, str) and row_system.strip() else system
        if prefix:
            messages = [{"role": "system", "content": prefix.strip()}, *messages]

    def answer(value: Any, what: str) -> list[dict[str, str]]:
        if isinstance(value, list):  # a message list: the answer is its last assistant turn
            turns = [t for t in _turns(value, "role", "content") if t["role"] == "assistant"]
            if not turns:
                raise Rejected(f"{what} has no assistant message")
            return [turns[-1]]
        return [{"role": "assistant", "content": _text(value, what)}]

    chosen, rejected = answer(row.get("chosen"), "chosen"), answer(row.get("rejected"), "rejected")
    if chosen[0]["content"] == rejected[0]["content"]:
        raise Rejected("chosen and rejected are the same")
    validate_messages([*messages, chosen[0]])
    return {"prompt": messages, "chosen": chosen, "rejected": rejected}


def chunk_text(text: str, chunk_chars: int | None = None, overlap: int = 0) -> list[str]:
    """A document as pieces of at most `chunk_chars` characters, cut at paragraph or line breaks
    where there is one in the second half of a piece. `overlap` characters repeat between pieces."""
    text = text.strip()
    if not text:
        return []
    if not chunk_chars or len(text) <= chunk_chars:
        return [text]
    if not 0 <= overlap < chunk_chars // 2:
        raise ValueError("chunk_overlap must be at least 0 and below half of chunk_chars")
    out, start = [], 0
    while start < len(text):
        end = min(len(text), start + chunk_chars)
        if end < len(text):
            window = text[start:end]
            for sep in ("\n\n", "\n", ". ", " "):
                cut = window.rfind(sep)
                if cut >= chunk_chars // 2:
                    end = start + cut + len(sep)
                    break
        piece = text[start:end].strip()
        if piece:
            out.append(piece)
        if end >= len(text):
            break
        start = end - overlap
    return out


def chat_chars(messages: list[dict[str, str]]) -> int:
    return sum(len(m["content"]) for m in messages)


def row_key(row: dict[str, Any]) -> str:
    """What makes two rows the same row: their content, whitespace and case aside."""
    canonical = json.dumps(row, sort_keys=True, ensure_ascii=False)
    return " ".join(canonical.lower().split())


def question_key(row: dict[str, Any]) -> str:
    """What a conversation asks: everything before its last assistant turn, the system message
    aside. Rows that ask the same thing share a key, whatever their answers say."""
    turns = [[m["role"], m["content"]] for m in row["messages"][:-1] if m["role"] != "system"]
    return " ".join(json.dumps(turns, ensure_ascii=False).lower().split())


def _unit(key: str, seed: int) -> float:
    return int(hashlib.sha1(f"{seed}:{key}".encode()).hexdigest()[:12], 16) / 16**12


def split_rows(
    rows: list[dict[str, Any]], eval_fraction: float, max_eval: int | None, seed: int,
    key: Callable[[dict[str, Any]], str] = row_key,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """`(train, eval)`. A row's side depends only on its content and the seed, not on its position
    or on the other rows, so adding data later never moves an old row across the split. With a
    positive `eval_fraction` and at least two rows, eval holds at least one row."""
    if eval_fraction <= 0 or len(rows) < 2:
        return list(rows), []
    scored = [(_unit(key(r), seed), i) for i, r in enumerate(rows)]
    picked = sorted(s for s in scored if s[0] < eval_fraction)
    if not picked:
        picked = [min(scored)]
    if max_eval is not None:
        picked = picked[:max_eval]
    eval_index = {i for _, i in picked}
    if len(eval_index) == len(rows):  # never leave train empty
        eval_index.discard(max(picked)[1])
    return (
        [r for i, r in enumerate(rows) if i not in eval_index],
        [r for i, r in enumerate(rows) if i in eval_index],
    )


class Collector:
    """Normalised rows of one kind, with counts of what was dropped and why."""

    def __init__(self, dedupe: bool = True):
        self.rows: list[dict[str, Any]] = []
        self.rejected: dict[str, int] = {}
        self.duplicates = 0
        self.seen: set[str] | None = set() if dedupe else None

    def reject(self, reason: str) -> None:
        self.rejected[reason] = self.rejected.get(reason, 0) + 1

    def add(self, row: dict[str, Any]) -> None:
        if self.seen is not None:
            k = hashlib.sha1(row_key(row).encode()).hexdigest()
            if k in self.seen:
                self.duplicates += 1
                return
            self.seen.add(k)
        self.rows.append(row)

    def extend(self, source: Iterable[dict[str, Any]], convert: Callable[[dict[str, Any]], Iterable[dict[str, Any]]]) -> None:
        for raw in source:
            try:
                for row in convert(raw):
                    self.add(row)
            except Rejected as e:
                self.reject(str(e))

    def stats(self) -> dict[str, Any]:
        return {"rows": len(self.rows), "duplicates": self.duplicates, "rejected": dict(sorted(self.rejected.items()))}
