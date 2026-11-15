"""`llm.generate_data`: a model answers training prompts several times, and a scorer sorts the answers.

The same stage builds three things, depending on which model it is pointed at:

- **Verified answers to train on.** A larger teacher (or the model itself) answers each prompt
  without seeing the reference; only answers the scorer accepts are kept, with their reasoning.
  Generating blind matters: a trace written toward a shown answer passes the same filter and
  still damages the student.
- **Preference pairs.** For a prompt with both right and wrong answers, a right one is `chosen`
  and a wrong one `rejected`: the model's own mistakes, ready for `llm.dpo`.
- **A reading of where the model stands.** `generate/mixed_fraction` is the share of prompts the
  model sometimes solves and sometimes does not, which is exactly where reinforcement learning
  has something to learn; near 0 it has nothing to work with.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from stagetrain.core import Stage, StageContext, StageResult, register, write_json, write_jsonl
from stagetrain.core.training import seed_everything

from .backend import get_backend
from .evaluate import GenerationParams, build_prompt
from .scoring import build_scorer, has_valid_reasoning
from .train import read_rows


class GenerateDataParams(GenerationParams):
    data: str  # chat JSONL whose last assistant turn is the reference answer
    max_samples: int | None = None
    samples_per_prompt: int = Field(default=8, ge=1)
    temperature: float = Field(default=0.8, gt=0.0)  # sampling: the answers of one prompt must differ
    top_p: float = 0.95
    max_new_tokens: int = 512
    threshold: float = Field(default=1.0, gt=0.0, le=1.0)  # the score from which an answer counts as right
    keep_per_prompt: int = Field(default=1, ge=0)  # distinct right answers kept per prompt as training rows
    pairs_per_prompt: int = Field(default=1, ge=0)  # (right, wrong) pairs per prompt
    require_reasoning: bool = False  # keep only right answers that carry a valid reasoning block
    system_prompt: str | None = None  # replaces the rows' system message while generating


def distinct(texts: list[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for text in texts:
        key = " ".join(text.split())
        if key and key not in seen:
            seen.add(key)
            out.append(text)
    return out


def sort_answers(
    replies: list[str], reference: str, row: dict[str, Any], judge, threshold: float = 1.0, require_reasoning: bool = False
) -> tuple[list[str], list[str], int]:
    """`(right, wrong, n_right)`: the distinct right and wrong answers, and how many of all
    replies were right (repeats included)."""
    scored = [(reply, judge(reply, reference, row) >= threshold) for reply in replies]
    right = distinct([r for r, ok in scored if ok and (not require_reasoning or has_valid_reasoning(r))])
    wrong = distinct([r for r, ok in scored if not ok])
    return right, wrong, sum(ok for _, ok in scored)


@register("llm.generate_data")
class GenerateData(Stage):
    """Sample answers to training prompts and keep the verified ones, as training rows and preference pairs."""

    Params = GenerateDataParams

    def run(self, ctx: StageContext) -> StageResult:
        p: GenerateDataParams = self.params
        seed_everything(self.seed(ctx))
        rows = [r for r in read_rows(p.data, p.max_samples) if len(r.get("messages") or []) >= 2]
        if not rows:
            raise ValueError(f"{p.data} holds no conversation with a prompt and a reference answer")
        backend = get_backend(p.backend)
        model, tokenizer = backend.load(p.load_spec())
        if p.chat_template:
            tokenizer = backend.set_chat_template(tokenizer, p.chat_template)
        prompts = [build_prompt(r["messages"][:-1], None, p.system_prompt) for r in rows]
        replies = backend.sample(
            model, tokenizer, prompts, n=p.samples_per_prompt, max_new_tokens=p.max_new_tokens,
            temperature=p.temperature, top_p=p.top_p, template_kwargs=p.chat_template_kwargs or None,
        )
        model = tokenizer = None
        backend.release()

        judge = build_scorer(p.scorer)
        sft, pairs = [], []
        solved = mixed = total_right = 0
        for row, group in zip(rows, replies, strict=True):
            original = row["messages"][:-1]  # training rows keep the data's own prompt
            right, wrong, n_right = sort_answers(
                group, row["messages"][-1]["content"], row, judge, p.threshold, p.require_reasoning
            )
            total_right += n_right
            solved += n_right > 0
            mixed += 0 < n_right < len(group)
            for answer in right[: p.keep_per_prompt]:
                sft.append({"messages": [*original, {"role": "assistant", "content": answer}]})
            for good, bad in list(zip(right, wrong, strict=False))[: p.pairs_per_prompt]:
                pairs.append({
                    "prompt": original, "chosen": [{"role": "assistant", "content": good}],
                    "rejected": [{"role": "assistant", "content": bad}],
                })
        n_samples = sum(len(group) for group in replies)
        metrics = {
            "generate/prompts": float(len(rows)), "generate/samples": float(n_samples),
            "generate/pass_at_1": total_right / n_samples,  # the chance that one sample is right
            "generate/pass_at_k": solved / len(rows),  # prompts solved by at least one sample
            "generate/mixed_fraction": mixed / len(rows),  # prompts with both right and wrong samples
            "generate/sft_rows": float(len(sft)), "generate/pairs": float(len(pairs)),
        }
        artifacts = {
            "sft": ctx.path("sft.jsonl"), "preference": ctx.path("preference.jsonl"),
            "stats": write_json(ctx.path("stats.json"), metrics),
        }
        write_jsonl(artifacts["sft"], sft)
        write_jsonl(artifacts["preference"], pairs)
        return StageResult(artifacts=artifacts, metrics=metrics)
