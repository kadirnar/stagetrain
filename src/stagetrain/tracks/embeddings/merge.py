"""`embeddings.merge`: a weighted average of models' weights.

Two uses, both measured (docs/research/embeddings.md):

- **Interpolating a fine-tuned model with the model it started from.** Domain fine-tuning lowers
  general retrieval ability; averaging the tuned weights with the starting weights gives most of
  it back for about a point of in-domain score (LM-Cocktail, arXiv 2311.13534).
- **Averaging runs that differ** in data mixture or task. Averaging checkpoints of one run, or
  seeds, has no measured gain.

Every model must have the same architecture: the same modules with the same tensor shapes.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field, model_validator

from ...core.select import group_runs
from ...core.stage import Stage, StageContext, StageParams, StageResult, register
from .evaluate import retrieval_metrics
from .model import EncoderSpec, load_encoder


class MergeParams(StageParams):
    models: list[str] = Field(default_factory=list)  # fine-tuned models, e.g. ["${finetune.model}"]
    group: str | None = None  # or every run of this matrix
    weights: list[float] | None = None  # relative weights of `models`; equal when unset
    base: str | None = None  # the model fine-tuning started from
    # The base's share of the result. 0 leaves the fine-tuned average alone; 0.5 is the midpoint.
    # Make it a matrix key and select on the validation set to choose it.
    base_weight: float = Field(default=0.5, ge=0.0, lt=1.0)
    eval_set: str | None = None  # evaluate the merged model, so that a stage can select or gate on it
    eval_dims: list[int] = Field(default_factory=list)
    eval_batch_size: int = 64
    encoder: EncoderSpec = Field(default_factory=EncoderSpec)

    @model_validator(mode="after")
    def _check(self) -> MergeParams:
        if not self.models and not self.group:
            raise ValueError("merge needs `models` or a `group`")
        if self.weights is not None:
            if self.group or len(self.weights) != len(self.models):
                raise ValueError("weights must match `models`, one each; a group's runs are weighted equally")
            if min(self.weights) < 0 or sum(self.weights) <= 0:
                raise ValueError("weights must not be negative and must sum to a positive number")
        return self


def member_weights(n_models: int, weights: list[float] | None, base: bool, base_weight: float) -> tuple[list[float], float]:
    """`(weight of each fine-tuned model, weight of the base)`; together they sum to 1."""
    relative = weights or [1.0] * n_models
    share = 1.0 - (base_weight if base else 0.0)
    return [share * w / sum(relative) for w in relative], (base_weight if base else 0.0)


def add_weighted(total: dict[str, Any] | None, state: dict[str, Any], weight: float, reference: dict[str, Any] | None = None):
    """`total + weight * state` over the floating-point tensors, in fp32."""
    import torch

    if reference is not None:
        if state.keys() != reference.keys():
            diff = sorted(set(state) ^ set(reference))[:5]
            raise ValueError(f"the models are not one architecture; tensors that differ: {diff}")
        for key, value in state.items():
            if value.shape != reference[key].shape:
                raise ValueError(f"{key}: shape {tuple(value.shape)} against {tuple(reference[key].shape)}")
    scaled = {k: v.to(torch.float32) * weight for k, v in state.items() if v.is_floating_point()}
    if total is None:
        return scaled
    for key, value in scaled.items():
        total[key] += value
    return total


@register("embeddings.merge")
class Merge(Stage):
    """Average the weights of fine-tuned models, optionally interpolated with the model they started from."""

    Params = MergeParams

    def run(self, ctx: StageContext) -> StageResult:
        p: MergeParams = self.params
        paths = list(p.models)
        if p.group:
            runs = group_runs(ctx.upstream, p.group)
            if not runs:
                raise ValueError(f"merge: {p.group!r} is not a matrix this stage depends on")
            paths += [m.artifact("model", ctx.run_dir) for m in runs.values()]
        weights, base_weight = member_weights(len(paths), p.weights, p.base is not None, p.base_weight)

        # The first fine-tuned model gives the result its modules, prompts and tokenizer.
        target = load_encoder(paths[0], p.encoder)
        reference = {k: v.detach().cpu() for k, v in target.state_dict().items()}
        total = add_weighted(None, reference, weights[0])
        others = list(zip(paths[1:], weights[1:], strict=True))
        if p.base is not None:
            others.append((p.base, base_weight))
        for path, weight in others:
            other = load_encoder(path, p.encoder, device="cpu")
            total = add_weighted(total, other.state_dict(), weight, reference)
            del other
        merged = {k: total[k].to(v.dtype) if k in total else v for k, v in reference.items()}
        target.load_state_dict(merged)
        out = ctx.out_dir / "model"
        target.save_pretrained(str(out))

        metrics: dict[str, float] = {"merge/members": float(len(paths)), "merge/base_weight": base_weight}
        if p.eval_set:
            metrics.update(retrieval_metrics(target, p.eval_set, p.eval_dims, p.eval_batch_size))
        notes = {"models": paths, "weights": weights, "base": p.base, "base_weight": base_weight}
        return StageResult(artifacts={"model": out}, metrics=metrics, notes=notes)
