"""`llm.select_checkpoint`: the checkpoint of a training stage with the best task metric.

Validation loss is a poor guide here: it typically bottoms out after about two epochs while the
quality of generated answers keeps improving, so the lowest-loss checkpoint is not the best model.
This stage answers the validation questions with every checkpoint and keeps the best one.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from pydantic import Field

from stagetrain.core import Stage, StageContext, StageResult, register, write_json
from stagetrain.core.history import append_history
from stagetrain.core.training import seed_everything

from .backend import get_backend
from .evaluate import GenerationParams, answer_rows
from .scoring import mean_with_error
from .train import read_rows, stage_dir

STEP = re.compile(r"checkpoint-(\d+)$")


class SelectCheckpointParams(GenerationParams):
    init_from: str  # a training stage's `checkpoints` artifact, e.g. ${sft.checkpoints}
    eval_data: str  # validation rows: not the test rows the final llm.eval reports on
    max_samples: int | None = 200
    every: int = Field(default=1, ge=1)  # judge every n-th checkpoint; the last one always
    # The earliest checkpoint within this much of the best score wins. Above 0 this prefers a less
    # trained model, which matters before reinforcement learning: a model fine-tuned until its
    # answers are near-deterministic has little left to explore.
    tolerance: float = Field(default=0.0, ge=0.0)


def list_checkpoints(directory: str | Path) -> list[tuple[int, Path]]:
    """`(step, path)` of every `checkpoint-<step>` directory, in step order."""
    found = []
    for path in Path(directory).glob("checkpoint-*"):
        match = STEP.search(path.name)
        if match and path.is_dir():
            found.append((int(match.group(1)), path))
    return sorted(found)


def choose(scores: list[tuple[int, float]], tolerance: float = 0.0) -> tuple[int, float]:
    """`(step, score)`: the earliest step whose score is within `tolerance` of the best."""
    best = max(score for _, score in scores)
    return next((step, score) for step, score in sorted(scores) if score >= best - tolerance)


@register("llm.select_checkpoint")
class SelectCheckpoint(Stage):
    """Judge every checkpoint of a training stage on the task metric and keep the best one."""

    Params = SelectCheckpointParams

    def run(self, ctx: StageContext) -> StageResult:
        p: SelectCheckpointParams = self.params
        seed_everything(self.seed(ctx))
        checkpoints = list_checkpoints(p.init_from)
        if not checkpoints:
            raise ValueError(f"{p.init_from} holds no checkpoint-<step> directory; is it a training stage's `checkpoints`?")
        picked = [c for i, c in enumerate(checkpoints) if i % p.every == 0 or i == len(checkpoints) - 1]
        rows = read_rows(p.eval_data, p.max_samples)
        if not rows:
            raise ValueError(f"{p.eval_data} holds no evaluation row")
        backend = get_backend(p.backend)
        table: list[dict[str, Any]] = []
        for step, path in picked:
            model, tokenizer = backend.load(p.load_spec(str(path)))
            if p.chat_template:
                tokenizer = backend.set_chat_template(tokenizer, p.chat_template)
            stats = mean_with_error([r["score"] for r in answer_rows(backend, model, tokenizer, rows, p)])
            model = tokenizer = None
            backend.release()
            table.append({"step": step, "score": stats["mean"], "se": stats["se"]})
            append_history(ctx.out_dir, {"step": step, "eval/score": stats["mean"]})  # the report's curve
            ctx.log.info("select_checkpoint: step %d scores %.4f (se %.4f)", step, stats["mean"], stats["se"])

        step, value = choose([(row["step"], row["score"]) for row in table], p.tolerance)
        chosen = dict(checkpoints)[step]
        model, tokenizer = backend.load(p.load_spec(str(chosen)))
        model_dir = stage_dir(ctx, "model")
        backend.save(model, tokenizer, model_dir)
        model = tokenizer = None
        backend.release()
        scores = [row["score"] for row in table]
        best_step = max(table, key=lambda row: (row["score"], -row["step"]))["step"]
        metrics = {
            "eval/score": value, "eval/rows": float(len(rows)),
            "select/checkpoints": float(len(table)), "select/chosen_step": float(step),
            "select/best_step": float(best_step), "select/spread": max(scores) - min(scores),
        }
        return StageResult(
            artifacts={"model": model_dir, "table": write_json(ctx.path("checkpoints.json"), table)},
            metrics=metrics, notes={"chosen": str(chosen), "checkpoints": table},
        )
