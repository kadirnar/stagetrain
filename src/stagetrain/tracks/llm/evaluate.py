"""`llm.eval`: the task metric of a model on held-out rows, measured so that the number can be trusted.

What it does that a loss cannot:

- it scores generated answers with a task scorer, and reports the standard error with the mean;
- it compares against another evaluation of the same rows item by item (`compare_to`), which
  says whether a difference is real with far fewer rows than two separate accuracies would need;
- it runs the same harness on the base model and on a prompt-only baseline (`few_shot`,
  `system_prompt`), so fine-tuning is credited only with what prompting does not already give;
- for reasoning models it counts the answers that still carry a valid reasoning block.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from pydantic import Field, model_validator

from stagetrain.core import Stage, StageContext, StageResult, register, write_json, write_jsonl
from stagetrain.core.training import seed_everything

from .backend import LlmTrainArgs, ResponsesOnly, get_backend, sft_config_kwargs
from .scoring import (
    ScorerSpec,
    build_scorer,
    has_valid_reasoning,
    mean_with_error,
    paired_difference,
    split_reasoning,
)
from .train import ModelParams, read_rows, render_chat, stage_dir


class GenerationParams(ModelParams):
    """How answers are generated and judged; shared by every stage that scores a model."""

    scorer: ScorerSpec = Field(default_factory=ScorerSpec)
    max_new_tokens: int = 256  # large enough that a reasoning trace is never cut before the answer
    temperature: float = Field(default=0.0, ge=0.0)  # 0 is greedy, which keeps the metric repeatable
    top_p: float = 1.0


class EvalParams(GenerationParams):
    eval_data: str  # chat JSONL, e.g. ${data.chat_eval}
    max_samples: int | None = 500
    batch_size: int = 4
    measure_loss: bool = True
    responses_only: bool = True  # the same masking as llm.sft, so the losses compare
    instruction_part: str | None = None
    response_part: str | None = None
    generate_samples: int | None = Field(default=None, ge=0)  # rows answered and scored; None: all, 0: none
    strip_thinking: bool = True  # drop <think>...</think> before the exact / contains comparison
    # Prompt-only baselines: what the untuned model reaches with a better prompt.
    system_prompt: str | None = None  # replaces the rows' system message
    few_shot: int = Field(default=0, ge=0)  # solved examples put before every question
    few_shot_data: str | None = None  # where they come from: training rows, never the eval rows
    compare_to: str | None = None  # the `scores` artifact of another llm.eval over the same rows
    expect_reasoning: bool = False  # also report the share of answers with a valid reasoning block

    @model_validator(mode="after")
    def _consistent(self) -> EvalParams:
        if (self.instruction_part is None) != (self.response_part is None):
            raise ValueError("give both instruction_part and response_part, or neither")
        if self.few_shot and not self.few_shot_data:
            raise ValueError("few_shot needs few_shot_data (use training rows, e.g. ${data.chat_train})")
        return self


def normalise(text: str, strip_thinking: bool = True) -> str:
    if strip_thinking:
        text = split_reasoning(text)[1]
    return " ".join(text.lower().split())


def score(reference: str, output: str, strip_thinking: bool = True) -> dict[str, bool]:
    """`exact`: the answer is the reference; `contains`: the reference appears in the answer."""
    ref, out = normalise(reference, False), normalise(output, strip_thinking)
    return {"exact": ref == out, "contains": bool(ref) and ref in out}


def prompt_key(messages: list[dict[str, str]]) -> str:
    """Identifies a question across evaluations of the same data, whatever prompt was built around it."""
    blob = json.dumps([[m["role"], " ".join(m["content"].split())] for m in messages if m["role"] != "system"])
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def build_prompt(
    messages: list[dict[str, str]], examples: list[dict[str, Any]] | None = None, system_prompt: str | None = None
) -> list[dict[str, str]]:
    """The conversation the model answers: an optional system message, solved examples, the question."""
    system = [m for m in messages[:1] if m["role"] == "system"]
    question = messages[len(system):]
    if system_prompt:
        system = [{"role": "system", "content": system_prompt}]
    shots = [m for example in examples or [] for m in example["messages"] if m["role"] != "system"]
    return [*system, *shots, *question]


def answer_rows(
    backend, model, tokenizer, rows: list[dict[str, Any]], p: GenerationParams, *,
    examples: list[dict[str, Any]] | None = None, system_prompt: str | None = None,
) -> list[dict[str, Any]]:
    """One record per row: the prompt, the reference, the model's answer and its score."""
    judge = build_scorer(p.scorer)
    prompts = [build_prompt(r["messages"][:-1], examples, system_prompt) for r in rows]
    outputs = backend.generate(
        model, tokenizer, prompts, max_new_tokens=p.max_new_tokens, temperature=p.temperature, top_p=p.top_p,
        template_kwargs=p.chat_template_kwargs or None,
    )
    records = []
    for row, prompt, output in zip(rows, prompts, outputs, strict=True):
        reference = row["messages"][-1]["content"]
        records.append({
            "key": prompt_key(row["messages"][:-1]), "prompt": prompt, "reference": reference, "output": output,
            "score": judge(output, reference, row),
        })
    return records


@register("llm.eval")
class Evaluate(Stage):
    """Task metric with its error, a paired comparison against a baseline, and held-out loss."""

    Params = EvalParams

    def run(self, ctx: StageContext) -> StageResult:
        p: EvalParams = self.params
        seed = self.seed(ctx)
        seed_everything(seed)
        rows = read_rows(p.eval_data, p.max_samples)
        if not rows:
            raise ValueError(f"{p.eval_data} holds no evaluation row")
        backend = get_backend(p.backend)
        model, tokenizer = backend.load(p.load_spec())
        if p.chat_template:
            tokenizer = backend.set_chat_template(tokenizer, p.chat_template)

        metrics: dict[str, float] = {"eval/rows": float(len(rows))}
        artifacts: dict[str, Any] = {}
        if p.measure_loss:
            config = sft_config_kwargs(
                LlmTrainArgs(per_device_eval_batch_size=p.batch_size), output_dir=stage_dir(ctx, "trainer"), seed=seed,
                has_eval=True, max_seq_length=p.max_seq_length, run_name=f"{ctx.pipeline}-{ctx.stage}",
                bf16=backend.bf16_supported(),
            )
            masking = ResponsesOnly(p.instruction_part, p.response_part) if p.responses_only else None
            texts = render_chat(tokenizer, rows, p.chat_template_kwargs)
            measured = backend.evaluate_loss(
                model, tokenizer, texts, config, responses_only=masking, conversations=[r["messages"] for r in rows]
            )
            if "eval/loss" in measured:
                metrics["eval/loss"] = measured["eval/loss"]
                if math.isfinite(measured["eval/loss"]):
                    metrics["eval/perplexity"] = math.exp(min(measured["eval/loss"], 50.0))

        picked = rows if p.generate_samples is None else rows[: p.generate_samples]
        if picked:
            examples = read_rows(p.few_shot_data, p.few_shot) if p.few_shot else None
            samples = answer_rows(backend, model, tokenizer, picked, p, examples=examples, system_prompt=p.system_prompt)
            for s in samples:
                s.update(score(s["reference"], s["output"], p.strip_thinking))
            artifacts["samples"] = ctx.path("samples.jsonl")
            write_jsonl(artifacts["samples"], samples)
            scores = {s["key"]: s["score"] for s in samples}
            artifacts["scores"] = write_json(ctx.path("scores.json"), scores)
            stats = mean_with_error([s["score"] for s in samples])
            metrics.update({
                "eval/generated": float(len(samples)),
                "eval/score": stats["mean"], "eval/score_se": stats["se"],
                "eval/score_ci_low": stats["ci_low"], "eval/score_ci_high": stats["ci_high"],
                "eval/exact_match": sum(s["exact"] for s in samples) / len(samples),
                "eval/contains": sum(s["contains"] for s in samples) / len(samples),
                # An answer that is empty once the reasoning is removed: the trace used up the tokens.
                "eval/no_answer": sum(not split_reasoning(s["output"])[1] for s in samples) / len(samples),
            })
            if p.expect_reasoning:
                metrics["eval/valid_reasoning"] = sum(has_valid_reasoning(s["output"]) for s in samples) / len(samples)
            if p.compare_to:
                baseline = json.loads(open(p.compare_to, encoding="utf-8").read())
                diff = paired_difference(scores, baseline)
                if not diff["n"]:
                    raise ValueError(f"{p.compare_to} shares no question with this evaluation; compare runs over the same rows")
                metrics.update({
                    "eval/delta": diff["delta"], "eval/delta_se": diff["se"], "eval/delta_ci_low": diff["ci_low"],
                    "eval/delta_ci_high": diff["ci_high"], "eval/delta_n": diff["n"],
                })
        model = tokenizer = None
        backend.release()
        artifacts["report"] = write_json(ctx.path("report.json"), {"model": p.init_from, "metrics": metrics})
        return StageResult(artifacts=artifacts, metrics=metrics)
