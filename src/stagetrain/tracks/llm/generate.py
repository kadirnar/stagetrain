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
- **The prompts to spend RL on.** `rl` holds the prompts inside that range, with a tenth of the
  always-solved ones. The measurement belongs to the model it was made with: run this stage
  again before a second RL stage.

Repeating this stage and `llm.sft` on its `sft` output, two or three times, is the cheap
alternative to RL: in the comparisons available it reached most of GRPO's gain. It stops paying
after about three rounds, and `generate/pass_at_k` falling from one round to the next is the sign.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from stagetrain.core import Stage, StageContext, StageResult, register, write_json, write_jsonl
from stagetrain.core.training import seed_everything

from .backend import get_backend
from .evaluate import GenerationParams, build_prompt
from .quality import jaccard, shingles
from .scoring import ScorerSpec, build_scorer, has_valid_reasoning
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
    # Which right answers are kept as training rows: in the order sampled, or the shortest first
    # (for a reasoning model: the same answer reached with less reasoning).
    prefer: Literal["first", "shortest"] = "first"
    # Which wrong answer is paired with a right one: the first sampled, or the one closest to it in
    # wording (a near miss teaches more than an answer that is wrong in every way).
    negatives: Literal["first", "nearest"] = "first"
    # Prompts for reinforcement learning (`rl`): those the model solves in more than
    # `rl_min_pass_rate` and less than `rl_max_pass_rate` of its samples, plus a share of the ones it
    # always solves. A group whose samples all score the same gives RL nothing to learn from;
    # cutting only one end of the range was measured to be worse than cutting neither.
    rl_min_pass_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    rl_max_pass_rate: float = Field(default=0.9, ge=0.0, le=1.0)
    rl_keep_solved: float = Field(default=0.1, ge=0.0, le=1.0)
    # A second, more lenient scorer. Answers the task scorer rejects and this one accepts are
    # probably right answers in a form the scorer does not read: a reward that punishes them
    # teaches the wrong thing, and more samples do not average it out.
    audit_scorer: ScorerSpec | None = None


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


def nearest(answer: str, candidates: list[str]) -> str:
    """The candidate whose wording is closest to `answer`."""
    target = shingles(answer, 3)
    return max(candidates, key=lambda c: jaccard(target, shingles(c, 3)))


def rl_prompts(
    rated: list[dict[str, Any]], low: float = 0.0, high: float = 0.9, keep_solved: float = 0.1, seed: int = 0
) -> list[dict[str, Any]]:
    """The prompts worth spending rollouts on: pass rate strictly between `low` and `high`, plus
    `keep_solved` of the prompts above `high`, so that what the model can do is still rehearsed."""
    import random

    band = [r for r in rated if low < r["pass_rate"] < high]
    solved = [r for r in rated if r["pass_rate"] >= high]
    extra = random.Random(seed).sample(solved, round(keep_solved * len(solved))) if solved else []
    chosen = {id(r) for r in [*band, *extra]}
    return [r for r in rated if id(r) in chosen]


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
        lenient = build_scorer(p.audit_scorer) if p.audit_scorer else None
        sft, pairs, rated = [], [], []
        solved = mixed = total_right = disputed = rejected = 0
        for row, group in zip(rows, replies, strict=True):
            original = row["messages"][:-1]  # training rows keep the data's own prompt
            reference = row["messages"][-1]["content"]
            right, wrong, n_right = sort_answers(group, reference, row, judge, p.threshold, p.require_reasoning)
            total_right += n_right
            solved += n_right > 0
            mixed += 0 < n_right < len(group)
            rated.append({**row, "pass_rate": n_right / len(group), "samples": len(group)})
            if lenient is not None:
                rejected += len(wrong)
                disputed += sum(lenient(answer, reference, row) >= p.threshold for answer in wrong)
            if p.prefer == "shortest":
                right = sorted(right, key=len)
            for answer in right[: p.keep_per_prompt]:
                sft.append({"messages": [*original, {"role": "assistant", "content": answer}]})
            for good in right[: p.pairs_per_prompt]:
                if not wrong:
                    break
                bad = nearest(good, wrong) if p.negatives == "nearest" else wrong[0]
                wrong = [w for w in wrong if w is not bad]
                pairs.append({
                    "prompt": original, "chosen": [{"role": "assistant", "content": good}],
                    "rejected": [{"role": "assistant", "content": bad}],
                })
        rl = rl_prompts(rated, p.rl_min_pass_rate, p.rl_max_pass_rate, p.rl_keep_solved, self.seed(ctx))
        n_samples = sum(len(group) for group in replies)
        metrics = {
            "generate/prompts": float(len(rows)), "generate/samples": float(n_samples),
            "generate/pass_at_1": total_right / n_samples,  # the chance that one sample is right
            "generate/pass_at_k": solved / len(rows),  # prompts solved by at least one sample
            "generate/mixed_fraction": mixed / len(rows),  # prompts with both right and wrong samples
            "generate/sft_rows": float(len(sft)), "generate/pairs": float(len(pairs)),
            "generate/rl_rows": float(len(rl)),
        }
        if lenient is not None:
            # Of the distinct answers the task scorer rejected, the share the lenient scorer accepts.
            metrics["generate/scorer_disagreement"] = disputed / rejected if rejected else 0.0
        artifacts = {
            "sft": ctx.path("sft.jsonl"), "preference": ctx.path("preference.jsonl"), "rl": ctx.path("rl.jsonl"),
            "prompts": ctx.path("prompts.jsonl"), "stats": write_json(ctx.path("stats.json"), metrics),
        }
        write_jsonl(artifacts["sft"], sft)
        write_jsonl(artifacts["preference"], pairs)
        write_jsonl(artifacts["rl"], [{k: v for k, v in r.items() if k not in ("pass_rate", "samples")} for r in rl])
        write_jsonl(artifacts["prompts"], rated)  # every prompt with the share of its samples that were right
        return StageResult(artifacts=artifacts, metrics=metrics)
