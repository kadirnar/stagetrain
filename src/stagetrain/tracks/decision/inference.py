"""Use a trained decision model: a state and named questions in, typed answers out.

    model = DecisionModel.load("runs/decision/<name>/export/model")
    model.decide("I was charged twice, please refund one.", {
        "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
        "team": {"type": "choice", "instructions": "Which team handles this?",
                 "criteria": {"billing": "payments and invoices", "tech": "bugs and outages"}},
        "urgency": {"type": "score", "instructions": "How urgent is it?", "criteria": ["low", "medium", "high"]},
    })
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .metrics import softmax
from .modeling import load_model, load_tokenizer, read_meta, score_records
from .prompt import PromptStyle
from .schema import NOUL, SCORE, Question, Record


def format_answer(question: Question, probs: Sequence[float]) -> dict[str, Any]:
    top = max(range(len(probs)), key=probs.__getitem__)
    if question.type == NOUL:
        return {"noul": probs[0]}
    if question.type == SCORE:
        return {
            "score": top, "confidence": probs[top], "probabilities": list(probs),
            "expected": sum(i * q for i, q in enumerate(probs)), "legend": list(question.descriptions),
        }
    return {
        "choice": question.names[top], "confidence": probs[top],
        "probabilities": dict(zip(question.names, probs, strict=True)),
    }


class DecisionModel:
    def __init__(self, model, tokenizer, style: PromptStyle | None = None, max_length: int | None = None,
                 calibration: Mapping[str, float] | None = None, batch_size: int = 16):
        self.model, self.tokenizer = model, tokenizer
        self.style = style or PromptStyle()
        self.max_length = max_length
        self.calibration = dict(calibration or {})
        self.batch_size = batch_size

    @classmethod
    def load(cls, path: str, precision: str = "auto", device: str | None = None, trust_remote_code: bool = False):
        meta = read_meta(path)
        return cls(
            load_model(path, precision, trust_remote_code, device), load_tokenizer(path, trust_remote_code),
            PromptStyle.model_validate(meta.get("prompt") or {}), meta.get("max_length"), meta.get("calibration"),
        )

    def decide_batch(self, requests: Sequence[tuple[Any, Mapping[str, Any]]]) -> list[dict[str, Any]]:
        """One response per `(state, {name: question})` request. Every question is one forward
        pass read at one position; no token is generated."""
        records, owners = [], []
        for i, (state, questions) in enumerate(requests):
            for name, q in questions.items():
                question = Question.parse(q)
                uniform = (1.0 / len(question.names),) * len(question.names)
                records.append(Record(f"{i}:{name}", "request", state, question, uniform))
                owners.append((i, name))
        preds = score_records(self.model, self.tokenizer, records, self.style, self.max_length, self.batch_size)
        out = [{"answers": {}, "usage": {"input_tokens": 0, "output_tokens": 0}} for _ in requests]
        for (i, name), record, pred in zip(owners, records, preds, strict=True):
            probs = softmax(pred["scores"], self.calibration.get(record.question.type, 1.0))
            out[i]["answers"][name] = format_answer(record.question, probs)
            out[i]["usage"]["input_tokens"] += pred["tokens"]
        return out

    def decide(self, state: Any, questions: Mapping[str, Any]) -> dict[str, Any]:
        return self.decide_batch([(state, questions)])[0]
