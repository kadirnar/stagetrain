"""The training stages: `llm.cpt`, `llm.sft` and `llm.dpo`.

All three start from `init_from`, which is a base model or the `model` artifact of an earlier
stage, so they chain in any order and any of them can appear more than once in a pipeline.

When `init_from` is a LoRA adapter, `on_adapter` decides how training goes on:

    continue   load the adapter and keep training it. Cheap, and what Unsloth documents. The
               adapter's shape (rank, target modules, trained embeddings) is the earlier stage's;
               this stage's `lora` is not used.
    merge      fold the adapter into a 16-bit copy of the base weights (written under this stage's
               directory) and attach a fresh LoRA from this stage's `lora`. Use it to change the
               adapter's shape, or to freeze what the earlier stage learned, e.g. the embeddings
               trained by continued pretraining.

`llm.cpt` and `llm.sft` default to `continue`, `llm.dpo` to `merge`.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from stagetrain.core import Stage, StageContext, StageParams, StageResult, read_jsonl, register, write_json
from stagetrain.core.training import seed_everything

from .backend import (
    CptLora,
    CptTrainArgs,
    DpoTrainArgs,
    LlmLora,
    LlmTrainArgs,
    LoadSpec,
    ResponsesOnly,
    TrainOutput,
    adapter_trains_embeddings,
    dpo_config_kwargs,
    get_backend,
    load_kwargs,
    peft_kwargs,
    sft_config_kwargs,
)


class ModelParams(StageParams):
    """How a stage loads its model."""

    init_from: str  # a Hub id, or an earlier stage's artifact such as ${sft.model}
    max_seq_length: int = 2048
    load_in: Literal["4bit", "8bit", "16bit"] = "4bit"
    full_finetuning: bool = False
    trust_remote_code: bool = False
    chat_template: str | None = None  # one of Unsloth's named templates, for a tokenizer without its own

    @model_validator(mode="after")
    def _loadable(self) -> ModelParams:
        load_kwargs(self.load_spec())  # raises on an impossible combination
        return self

    def load_spec(self, model_name: str | None = None) -> LoadSpec:
        return LoadSpec(
            model_name=model_name or self.init_from, max_seq_length=self.max_seq_length, load_in=self.load_in,
            full_finetuning=self.full_finetuning, trust_remote_code=self.trust_remote_code,
        )


class TrainParams(ModelParams):
    train_data: str
    eval_data: str | None = None
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    on_adapter: Literal["continue", "merge"] = "continue"
    lora: LlmLora = Field(default_factory=LlmLora)
    train: LlmTrainArgs = Field(default_factory=LlmTrainArgs)
    embedding_learning_rate: float | None = None  # for embed_tokens / lm_head when they are trained
    resume: bool = True  # after an interruption, go on from this stage's last checkpoint


class CptParams(TrainParams):
    lora: CptLora = Field(default_factory=CptLora)
    train: CptTrainArgs = Field(default_factory=CptTrainArgs)
    train_embeddings: bool = True  # also train embed_tokens and lm_head, so new vocabulary can be learnt
    packing: bool = False


class SftParams(TrainParams):
    responses_only: bool = True  # loss on the assistant turns only
    instruction_part: str | None = None  # the template's user and assistant markers;
    response_part: str | None = None  # leave both unset to detect them from the chat template
    last_response_only: bool = False
    packing: bool = False

    @model_validator(mode="after")
    def _parts(self) -> SftParams:
        if (self.instruction_part is None) != (self.response_part is None):
            raise ValueError("give both instruction_part and response_part, or neither")
        return self

    def masking(self) -> ResponsesOnly | None:
        if not self.responses_only:
            return None
        return ResponsesOnly(self.instruction_part, self.response_part, self.last_response_only)


class DpoParams(TrainParams):
    # DPO measures itself against the policy it started from. After `merge` that policy is exactly
    # the new adapter's base model, whatever the TRL release does with an already-trained adapter.
    on_adapter: Literal["continue", "merge"] = "merge"
    train: DpoTrainArgs = Field(default_factory=DpoTrainArgs)
    beta: float = 0.1
    loss_type: str | list[str] | None = None  # TRL's default (sigmoid) when unset

    @model_validator(mode="after")
    def _one_learning_rate(self) -> DpoParams:
        if self.embedding_learning_rate is not None:
            raise ValueError("embedding_learning_rate applies to cpt and sft; TRL's DPO trainer has one learning rate")
        return self


# --------------------------------------------------------------------------- #
# shared steps
# --------------------------------------------------------------------------- #


def stage_dir(ctx: StageContext, name: str) -> Path:
    path = ctx.out_dir / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_rows(path: str | None, limit: int | None = None) -> list[dict[str, Any]]:
    if not path:
        return []
    if not Path(path).is_file():
        raise FileNotFoundError(f"no data file at {path}")
    rows = []
    for row in read_jsonl(path):
        rows.append(row)
        if limit is not None and len(rows) >= limit:
            break
    return rows


def render_chat(tokenizer, rows: list[dict[str, Any]]) -> list[str]:
    """Conversations as the text the model is trained on, through the tokenizer's chat template."""
    if not getattr(tokenizer, "chat_template", True):
        raise ValueError(
            "the tokenizer has no chat template; set chat_template to one of Unsloth's names (e.g. chatml)"
        )
    return [
        tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=False) for row in rows
    ]


def render_text(tokenizer, rows: list[dict[str, Any]]) -> list[str]:
    """Raw documents, each ended with the end-of-sequence token so the model learns to stop."""
    eos = getattr(tokenizer, "eos_token", None) or ""
    return [row["text"] if eos and row["text"].endswith(eos) else row["text"] + eos for row in rows]


def prepare_model(p: TrainParams, ctx: StageContext, seed: int, train_embeddings: bool = False):
    """`(model, tokenizer, info)` ready to train, following `on_adapter` (see the module docstring)."""
    backend = get_backend()
    model, tokenizer = backend.load(p.load_spec())
    info: dict[str, Any] = {"init_from": p.init_from, "trains_embeddings": train_embeddings and not p.full_finetuning}
    started_from_adapter = backend.has_adapter(model)
    if started_from_adapter and (p.on_adapter == "merge" or p.full_finetuning):
        merged = stage_dir(ctx, "merged_base")
        ctx.log.info("stage %s: merging the adapter of %s into %s", ctx.stage, p.init_from, merged)
        backend.save_merged(model, tokenizer, merged, "merged_16bit")
        model = tokenizer = None  # free the old model before the merged one is loaded
        backend.release()
        model, tokenizer = backend.load(p.load_spec(str(merged)))
        info["merged_base"] = str(merged)
        started_from_adapter = False
    if p.chat_template:
        tokenizer = backend.set_chat_template(tokenizer, p.chat_template)
    if p.full_finetuning:
        info["mode"] = "full"
    elif started_from_adapter:
        info["mode"] = "adapter:continue"
        info["trains_embeddings"] = adapter_trains_embeddings(p.init_from)
        ctx.log.info("stage %s: continuing the adapter of %s; this stage's lora settings are not used", ctx.stage, p.init_from)
    else:
        model = backend.attach_lora(model, peft_kwargs(p.lora, seed, train_embeddings))
        info["mode"] = "adapter:merge" if "merged_base" in info else "adapter:new"
    return model, tokenizer, info


def embedding_lr(p: TrainParams, trains_embeddings: bool) -> float | None:
    """Unsloth's recipe trains the embedding matrices 2-10x slower than the LoRA weights. That
    also holds for a later stage that continues an adapter whose embeddings are trained."""
    if p.embedding_learning_rate is not None:
        return p.embedding_learning_rate
    return p.train.learning_rate / 10 if trains_embeddings else None


def finish(ctx: StageContext, model, tokenizer, out: TrainOutput, info: dict[str, Any], rows: dict[str, int]) -> StageResult:
    backend = get_backend()
    model_dir = stage_dir(ctx, "model")
    backend.save(model, tokenizer, model_dir)
    model = tokenizer = None
    backend.release()
    metrics = dict(out.metrics)
    if "eval/loss" in metrics and math.isfinite(metrics["eval/loss"]):
        metrics["eval/perplexity"] = math.exp(min(metrics["eval/loss"], 50.0))
    metrics.update({f"{k}/rows": float(v) for k, v in rows.items()})
    artifacts: dict[str, Any] = {"model": model_dir, "log": write_json(ctx.path("train_log.json"), out.log_history)}
    if "merged_base" in info:
        artifacts["base"] = info["merged_base"]  # the new adapter's base model
    return StageResult(artifacts=artifacts, metrics=metrics, notes=info)


def _text_stage(stage: Stage, ctx: StageContext, render, masking: ResponsesOnly | None, train_embeddings: bool) -> StageResult:
    p: TrainParams = stage.params
    seed = stage.seed(ctx)
    seed_everything(seed)
    train_rows = read_rows(p.train_data, p.max_train_samples)
    eval_rows = read_rows(p.eval_data, p.max_eval_samples)
    if not train_rows:
        raise ValueError(f"{p.train_data} holds no training row")
    backend = get_backend()
    model, tokenizer, info = prepare_model(p, ctx, seed, train_embeddings)
    config = sft_config_kwargs(
        p.train, output_dir=stage_dir(ctx, "checkpoints"), seed=seed, has_eval=bool(eval_rows),
        max_seq_length=p.max_seq_length, run_name=f"{ctx.pipeline}-{ctx.stage}", packing=p.packing,
        embedding_learning_rate=embedding_lr(p, info["trains_embeddings"]), bf16=backend.bf16_supported(),
    )
    out = backend.train_text(
        model, tokenizer, render(tokenizer, train_rows), render(tokenizer, eval_rows), config,
        responses_only=masking, resume=p.resume,
    )
    return finish(ctx, model, tokenizer, out, info, {"train": len(train_rows), "eval": len(eval_rows)})


# --------------------------------------------------------------------------- #
# stages
# --------------------------------------------------------------------------- #


@register("llm.cpt")
class ContinuedPretraining(Stage):
    """Continued pretraining on raw domain text, with the embeddings and the output head trained too."""

    Params = CptParams

    def run(self, ctx: StageContext) -> StageResult:
        return _text_stage(self, ctx, render_text, None, self.params.train_embeddings)


@register("llm.sft")
class SupervisedFineTuning(Stage):
    """Chat fine-tuning through the tokenizer's chat template, with the loss on assistant turns only."""

    Params = SftParams

    def run(self, ctx: StageContext) -> StageResult:
        return _text_stage(self, ctx, render_chat, self.params.masking(), False)


@register("llm.dpo")
class PreferenceOptimisation(Stage):
    """Direct preference optimisation on prompt / chosen / rejected pairs (TRL's DPOTrainer)."""

    Params = DpoParams

    def run(self, ctx: StageContext) -> StageResult:
        p: DpoParams = self.params
        seed = self.seed(ctx)
        seed_everything(seed)
        train_rows = read_rows(p.train_data, p.max_train_samples)
        eval_rows = read_rows(p.eval_data, p.max_eval_samples)
        if not train_rows:
            raise ValueError(f"{p.train_data} holds no preference pair")
        backend = get_backend()
        model, tokenizer, info = prepare_model(p, ctx, seed)
        config = dpo_config_kwargs(
            p.train, output_dir=stage_dir(ctx, "checkpoints"), seed=seed, has_eval=bool(eval_rows),
            max_seq_length=p.max_seq_length, beta=p.beta, loss_type=p.loss_type,
            run_name=f"{ctx.pipeline}-{ctx.stage}", bf16=backend.bf16_supported(),
        )
        out = backend.train_dpo(model, tokenizer, train_rows, eval_rows, config, resume=p.resume)
        return finish(ctx, model, tokenizer, out, info, {"train": len(train_rows), "eval": len(eval_rows)})
