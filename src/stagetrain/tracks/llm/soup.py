"""`llm.soup`: one adapter whose update is the average of several runs' updates.

Runs of one recipe that differ only in their seed end in different places, and the average of
their updates is usually better than any of them: in the one study that isolated it, three seeds
averaged gained 2.6 and 3.3 points on GSM8K over a single run, more than a higher rank or more
epochs bought, at the inference cost of one model. Runs that differ in data or recipe are another
matter: their plain average was measured to be worse than the best member.

So the members should be the seeds of one configuration. With `selected_by`, they are the runs
of the configuration an `llm.select` stage chose (give that stage `average_over: [seed]`).
Evaluate the result next to a single run (`llm.eval` with `compare_to`) before relying on it.
"""

from __future__ import annotations

import shutil

from pydantic import Field, model_validator

from stagetrain.core import Stage, StageContext, StageParams, StageResult, group_runs, register
from stagetrain.core.soup import average_adapters

from .backend import is_adapter_dir


class SoupParams(StageParams):
    models: list[str] = Field(default_factory=list)  # LoRA adapters: training stages' `model`
    group: str | None = None  # or: the runs of this matrix
    selected_by: str | None = None  # with `group`: only the runs of the configuration this llm.select stage chose
    weights: list[float] | None = None  # for `models`; equal weights when unset

    @model_validator(mode="after")
    def _check(self) -> SoupParams:
        if not self.models and not self.group:
            raise ValueError("a soup needs models, or the group of a matrix")
        if self.selected_by and not self.group:
            raise ValueError("selected_by narrows the runs of a group: set group as well")
        if self.weights and (self.group or len(self.weights) != len(self.models)):
            raise ValueError("weights go with models, one weight per model")
        return self


@register("llm.soup")
class Soup(Stage):
    """Average the LoRA updates of several runs of one recipe into one adapter."""

    Params = SoupParams

    def run(self, ctx: StageContext) -> StageResult:
        p: SoupParams = self.params
        members = list(p.models)
        if p.group:
            runs = group_runs(ctx.upstream, p.group)
            if not runs:
                raise ValueError(f"soup: {p.group!r} is not a matrix this stage depends on; it sees {sorted(ctx.upstream)}")
            names = list(runs)
            if p.selected_by:
                chosen = ctx.upstream.get(p.selected_by)
                if chosen is None or "winner_runs" not in chosen.notes:
                    raise ValueError(f"soup: {p.selected_by!r} is not a select stage this stage depends on")
                names = [n for n in chosen.notes["winner_runs"] if n in runs]
            members += [str(runs[n].artifact("model", ctx.run_dir)) for n in names]
        if len(members) < 2:
            raise ValueError(
                f"soup: {len(members)} member; averaging needs at least two runs. With selected_by, give the "
                "matrix several seeds and the select stage average_over: [seed]"
            )
        full = [m for m in members if not is_adapter_dir(m)]
        if full:
            raise ValueError(f"soup: {full} are not LoRA adapters; full checkpoints are not averaged by this stage")
        out = ctx.out_dir / "model"
        shutil.rmtree(out, ignore_errors=True)
        average_adapters(members, out, p.weights)
        import json

        rank = json.loads((out / "adapter_config.json").read_text())["r"]
        return StageResult(
            artifacts={"model": out},
            metrics={"soup/members": float(len(members)), "soup/rank": float(rank)},
            notes={"members": members, "mode": "lora"},
        )
