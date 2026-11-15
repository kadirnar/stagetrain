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

from .metrics import probabilities
from .modeling import load_model, load_tokenizer, read_meta, score_records
from .prompt import PromptStyle
from .schema import CHOICE, NOUL, SCORE, Question, Record


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
    """`orders` above 1 asks every choice question that many times with its options in different
    orders and averages the answers. It removes what is left of the model's preference for an
    option's position, at `orders` times the cost."""

    def __init__(self, model, tokenizer, style: PromptStyle | None = None, max_length: int | None = None,
                 calibration: Mapping[str, Any] | None = None, batch_size: int = 16, orders: int = 1):
        self.model, self.tokenizer = model, tokenizer
        self.style = style or PromptStyle()
        self.max_length = max_length
        self.calibration = dict(calibration or {})
        self.batch_size = batch_size
        self.orders = max(1, orders)

    @classmethod
    def load(cls, path: str, precision: str = "auto", device: str | None = None, trust_remote_code: bool = False,
             orders: int = 1):
        meta = read_meta(path)
        return cls(
            load_model(path, precision, trust_remote_code, device), load_tokenizer(path, trust_remote_code),
            PromptStyle.model_validate(meta.get("prompt") or {}), meta.get("max_length"), meta.get("calibration"),
            orders=orders,
        )

    def _probabilities(self, records: list[Record]) -> tuple[list[list[float]], list[int]]:
        """Per record: the probability of every option in the record's own order, and the tokens read."""
        runs = [score_records(self.model, self.tokenizer, records, self.style, self.max_length, self.batch_size)]
        for k in range(1, self.orders):
            runs.append(score_records(
                self.model, self.tokenizer, records, self.style, self.max_length, self.batch_size, permute_seed=k
            ))
        out, tokens = [], []
        for i, record in enumerate(records):
            n = len(record.question.names)
            total = [0.0] * n
            used = 0
            for run in runs if record.question.type == CHOICE else runs[:1]:
                pred = run[i]
                for shown, p in enumerate(probabilities(pred, self.calibration)):
                    total[pred["order"][shown]] += p
                used += 1
            out.append([p / used for p in total])
            tokens.append(sum(run[i]["tokens"] for run in runs[:used]))
        return out, tokens

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
        probs, tokens = self._probabilities(records)
        out = [{"answers": {}, "usage": {"input_tokens": 0, "output_tokens": 0}} for _ in requests]
        for (i, name), record, p, used in zip(owners, records, probs, tokens, strict=True):
            out[i]["answers"][name] = format_answer(record.question, p)
            out[i]["usage"]["input_tokens"] += used
        return out

    def decide(self, state: Any, questions: Mapping[str, Any]) -> dict[str, Any]:
        return self.decide_batch([(state, questions)])[0]
