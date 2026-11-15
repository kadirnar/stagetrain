"""Stage `decision.soup`: average the weights of several fine-tuned checkpoints."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Literal

from pydantic import model_validator

from stagetrain.core import Stage, StageContext, StageParams, StageResult, register
from stagetrain.core.soup import average_checkpoints

from .metrics import summarize
from .modeling import load_model, load_records, load_tokenizer, read_meta, score_records
from .prompt import PromptStyle


class SoupParams(StageParams):
    models: list[str]  # full checkpoints of one architecture: `decision.train` stages' `model`
    method: Literal["uniform", "greedy"] = "uniform"
    weights: list[float] | None = None  # uniform only
    # greedy: held-out records; a checkpoint joins the soup only if the soup then scores no worse
    data: str | None = None
    metric: Literal["nll", "accuracy"] = "nll"
    max_samples: int | None = 500
    batch_size: int = 16
    precision: str = "auto"
    trust_remote_code: bool = False

    @model_validator(mode="after")
    def _check(self) -> SoupParams:
        if not self.models:
            raise ValueError("a soup needs at least one model")
        if self.method == "greedy" and not self.data:
            raise ValueError("a greedy soup needs held-out data")
        if self.method == "greedy" and self.weights:
            raise ValueError("weights apply to a uniform soup only")
        return self


@register("decision.soup")
class Soup(Stage):
    """Merge checkpoints trained with different seeds and data mixtures into one model."""

    Params = SoupParams

    def score(self, model_dir: Path | str) -> float:
        """The held-out metric of a checkpoint, as a number where higher is better."""
        import gc

        p = self.params
        meta = read_meta(model_dir)
        style = PromptStyle.model_validate(meta.get("prompt") or {})
        tokenizer = load_tokenizer(str(model_dir), p.trust_remote_code)
        model = load_model(str(model_dir), p.precision, p.trust_remote_code)
        records = load_records(p.data, p.max_samples)
        result = summarize(score_records(model, tokenizer, records, style, meta.get("max_length"), p.batch_size))
        del model
        gc.collect()
        return result["accuracy"] if p.metric == "accuracy" else -result["nll"]

    def run(self, ctx: StageContext) -> StageResult:
        p = self.params
        out = ctx.out_dir / "model"
        shutil.rmtree(out, ignore_errors=True)
        notes: dict = {"models": list(p.models)}
        if p.method == "uniform":
            average_checkpoints(p.models, out, p.weights)
            members = list(p.models)
            metrics = {"soup/members": float(len(members))}
        else:
            single = {m: self.score(m) for m in p.models}
            ranked = sorted(p.models, key=single.__getitem__, reverse=True)
            members, best = [ranked[0]], single[ranked[0]]
            average_checkpoints(members, out)
            candidate = ctx.out_dir / "candidate"
            for m in ranked[1:]:
                shutil.rmtree(candidate, ignore_errors=True)
                average_checkpoints([*members, m], candidate)
                value = self.score(candidate)
                ctx.log.info("soup: %d members + %s -> %.5f (best %.5f)", len(members), m, value, best)
                if value >= best:
                    members, best = [*members, m], value
                    shutil.rmtree(out)
                    candidate.rename(out)
            shutil.rmtree(candidate, ignore_errors=True)
            sign = 1.0 if p.metric == "accuracy" else -1.0
            metrics = {"soup/members": float(len(members)), f"soup/{p.metric}": sign * best,
                       f"soup/best_single_{p.metric}": sign * single[ranked[0]]}
            notes["single"] = {m: sign * v for m, v in single.items()}
        notes["members"] = members
        return StageResult(artifacts={"model": out}, metrics=metrics, notes=notes)
