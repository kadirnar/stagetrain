"""`llm.eval`: held-out loss and perplexity, plus generated answers compared with the references."""

from __future__ import annotations

import math
import re
from typing import Any

from pydantic import Field, model_validator

from stagetrain.core import Stage, StageContext, StageResult, register, write_json, write_jsonl
from stagetrain.core.training import seed_everything

from .backend import LlmTrainArgs, ResponsesOnly, get_backend, sft_config_kwargs
from .train import ModelParams, read_rows, render_chat, stage_dir

THINKING = re.compile(r"<think>.*?</think>", re.DOTALL)


class EvalParams(ModelParams):
    eval_data: str  # chat JSONL, e.g. ${data.chat_eval}
    max_samples: int | None = 500
    batch_size: int = 4
    responses_only: bool = True  # the same masking as llm.sft, so the losses compare
    instruction_part: str | None = None
    response_part: str | None = None
    generate_samples: int = Field(default=20, ge=0)  # 0 skips generation
    max_new_tokens: int = 256
    temperature: float = Field(default=0.0, ge=0.0)  # 0 is greedy, which keeps the metric repeatable
    top_p: float = 1.0
    strip_thinking: bool = True  # drop <think>...</think> before comparing

    @model_validator(mode="after")
    def _parts(self) -> EvalParams:
        if (self.instruction_part is None) != (self.response_part is None):
            raise ValueError("give both instruction_part and response_part, or neither")
        return self


def normalise(text: str, strip_thinking: bool = True) -> str:
    if strip_thinking:
        text = THINKING.sub(" ", text)
    return " ".join(text.lower().split())


def score(reference: str, output: str, strip_thinking: bool = True) -> dict[str, bool]:
    """`exact`: the answer is the reference; `contains`: the reference appears in the answer."""
    ref, out = normalise(reference, False), normalise(output, strip_thinking)
    return {"exact": ref == out, "contains": bool(ref) and ref in out}


@register("llm.eval")
class Evaluate(Stage):
    """Held-out loss / perplexity and sample generations of a base model or a trained stage."""

    Params = EvalParams

    def run(self, ctx: StageContext) -> StageResult:
        p: EvalParams = self.params
        seed = self.seed(ctx)
        seed_everything(seed)
        rows = read_rows(p.eval_data, p.max_samples)
        if not rows:
            raise ValueError(f"{p.eval_data} holds no evaluation row")
        backend = get_backend()
        model, tokenizer = backend.load(p.load_spec())
        if p.chat_template:
            tokenizer = backend.set_chat_template(tokenizer, p.chat_template)

        config = sft_config_kwargs(
            LlmTrainArgs(per_device_eval_batch_size=p.batch_size), output_dir=stage_dir(ctx, "trainer"), seed=seed,
            has_eval=True, max_seq_length=p.max_seq_length, run_name=f"{ctx.pipeline}-{ctx.stage}",
            bf16=backend.bf16_supported(),
        )
        masking = ResponsesOnly(p.instruction_part, p.response_part) if p.responses_only else None
        measured = backend.evaluate_loss(model, tokenizer, render_chat(tokenizer, rows), config, responses_only=masking)
        metrics: dict[str, float] = {"eval/rows": float(len(rows))}
        if "eval/loss" in measured:
            metrics["eval/loss"] = measured["eval/loss"]
            if math.isfinite(measured["eval/loss"]):
                metrics["eval/perplexity"] = math.exp(min(measured["eval/loss"], 50.0))

        artifacts: dict[str, Any] = {}
        picked = rows[: p.generate_samples]
        if picked:
            outputs = backend.generate(
                model, tokenizer, [r["messages"][:-1] for r in picked], max_new_tokens=p.max_new_tokens,
                temperature=p.temperature, top_p=p.top_p,
            )
            samples = []
            for row, output in zip(picked, outputs, strict=True):
                reference = row["messages"][-1]["content"]
                samples.append({
                    "prompt": row["messages"][:-1], "reference": reference, "output": output,
                    **score(reference, output, p.strip_thinking),
                })
            artifacts["samples"] = ctx.path("samples.jsonl")
            write_jsonl(artifacts["samples"], samples)
            metrics["eval/generated"] = float(len(samples))
            metrics["eval/exact_match"] = sum(s["exact"] for s in samples) / len(samples)
            metrics["eval/contains"] = sum(s["contains"] for s in samples) / len(samples)
        model = tokenizer = None
        backend.release()
        artifacts["report"] = write_json(ctx.path("report.json"), {"model": p.init_from, "metrics": metrics})
        return StageResult(artifacts=artifacts, metrics=metrics)
