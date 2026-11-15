"""The training stages: `llm.cpt`, `llm.sft`, `llm.dpo` and `llm.grpo`.

All of them start from `init_from`, which is a base model or the `model` artifact of an earlier
stage, so they chain in any order and any of them can appear more than once in a pipeline.

When `init_from` is a LoRA adapter, `on_adapter` decides how training goes on:

    continue   load the adapter and keep training it. Cheap, and what Unsloth documents. The
               adapter's shape (rank, target modules, trained embeddings) is the earlier stage's;
               this stage's `lora` is not used.
    merge      fold the adapter into a 16-bit copy of the base weights (written under this stage's
               directory) and attach a fresh LoRA from this stage's `lora`. Use it to change the
               adapter's shape, or to freeze what the earlier stage learned, e.g. the embeddings
               trained by continued pretraining.

`llm.cpt`, `llm.sft` and `llm.grpo` default to `continue`, `llm.dpo` to `merge`.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from stagetrain.core import Stage, StageContext, StageParams, StageResult, read_jsonl, register, write_json
from stagetrain.core.history import save_history
from stagetrain.core.training import seed_everything

from .backend import (
    CptLora,
    CptTrainArgs,
    DpoTrainArgs,
    GrpoTrainArgs,
    LlmLora,
    LlmTrainArgs,
    LoadSpec,
    ResponsesOnly,
    SftTrainArgs,
    TrainOutput,
    adapter_trains_embeddings,
    dpo_config_kwargs,
    get_backend,
    grpo_config_kwargs,
    load_kwargs,
    peft_kwargs,
    sft_config_kwargs,
)
from .rewards import RewardSpec, build_rewards


class ModelParams(StageParams):
    """How a stage loads its model."""

    init_from: str  # a Hub id, or an earlier stage's artifact such as ${sft.model}
    max_seq_length: int = 2048
    load_in: Literal["4bit", "8bit", "16bit"] = "4bit"
    full_finetuning: bool = False
    trust_remote_code: bool = False
    chat_template: str | None = None  # one of Unsloth's named templates, for a tokenizer without its own
    # Passed to the chat template wherever a prompt is rendered, e.g. {enable_thinking: false}.
    # Use the same value in every stage of a pipeline: a model is read the way it was trained.
    chat_template_kwargs: dict[str, Any] = Field(default_factory=dict)

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
    # A row longer than max_seq_length is dropped, not cut: a cut row teaches an answer (or a
    # reasoning trace) that stops in the middle. `truncate` keeps the library's behaviour.
    overlength: Literal["drop", "truncate"] = "drop"


class CptParams(TrainParams):
    lora: CptLora = Field(default_factory=CptLora)
    train: CptTrainArgs = Field(default_factory=CptTrainArgs)
    train_embeddings: bool = True  # also train embed_tokens and lm_head, so new vocabulary can be learnt
    packing: bool = False


class SftParams(TrainParams):
    train: SftTrainArgs = Field(default_factory=SftTrainArgs)
    responses_only: bool = True  # loss on the assistant turns only
    instruction_part: str | None = None  # the template's user and assistant markers;
    response_part: str | None = None  # leave both unset to detect them from the chat template
    last_response_only: bool = False
    mask_out_tokens: list[str] | None = None  # tokens inside a response that carry no loss
    packing: bool = False

    @model_validator(mode="after")
    def _parts(self) -> SftParams:
        if (self.instruction_part is None) != (self.response_part is None):
            raise ValueError("give both instruction_part and response_part, or neither")
        return self

    def masking(self) -> ResponsesOnly | None:
        if not self.responses_only:
            return None
        return ResponsesOnly(
            self.instruction_part, self.response_part, self.last_response_only,
            tuple(self.mask_out_tokens) if self.mask_out_tokens else None,
        )


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


class GrpoParams(TrainParams):
    """Reinforcement learning on prompts whose answers can be checked.

    `train_data` is chat JSONL: the conversation up to the last assistant turn is the prompt and
    that turn's content is the reference the rewards compare completions with. Start from a model
    that already solves part of the prompts (see `llm.generate_data`'s `generate/mixed_fraction`):
    when every completion of a group gets the same reward there is nothing to learn from.
    """

    train: GrpoTrainArgs = Field(default_factory=GrpoTrainArgs)
    rewards: list[RewardSpec] = Field(default_factory=lambda: [RewardSpec()])
    num_generations: int = 8  # completions sampled per prompt
    max_completion_length: int = 512  # tokens; long enough for the reasoning the task needs
    max_prompt_tokens: int | None = None  # longer prompts are dropped; None: max_seq_length - max_completion_length
    temperature: float = 1.0
    top_p: float = 1.0
    loss_type: Literal["dapo", "grpo", "dr_grpo", "bnpo", "cispo"] = "dapo"
    beta: float = 0.0  # KL penalty toward the starting policy; the current recipes run without it
    importance_sampling_level: Literal["token", "sequence"] = "token"  # sequence is GSPO
    mask_truncated_completions: bool | None = None
    fast_inference: bool = False  # generate with vLLM: much faster, LoRA on a 4bit or 16bit base only
    gpu_memory_utilization: float = 0.5

    @model_validator(mode="after")
    def _consistent(self) -> GrpoParams:
        if self.eval_data:
            raise ValueError("grpo takes no eval_data: measure the held-out reward with llm.eval, which decodes greedily")
        if self.embedding_learning_rate is not None:
            raise ValueError("embedding_learning_rate applies to cpt and sft")
        build_rewards(self.rewards)  # distinct names, importable custom functions
        grpo_config_kwargs(  # the batch rule and the objective's name
            self.train, output_dir="unused", seed=0, num_generations=self.num_generations,
            max_completion_length=self.max_completion_length, temperature=self.temperature, top_p=self.top_p,
            loss_type=self.loss_type, beta=self.beta, importance_sampling_level=self.importance_sampling_level,
        )
        if self.max_completion_length >= self.max_seq_length:
            raise ValueError("max_completion_length must leave room for the prompt inside max_seq_length")
        return self

    def load_spec(self, model_name: str | None = None) -> LoadSpec:
        return LoadSpec(
            model_name=model_name or self.init_from, max_seq_length=self.max_seq_length, load_in=self.load_in,
            full_finetuning=self.full_finetuning, trust_remote_code=self.trust_remote_code,
            fast_inference=self.fast_inference, max_lora_rank=max(self.lora.r, 8),
            gpu_memory_utilization=self.gpu_memory_utilization,
        )


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


def render_chat(tokenizer, rows: list[dict[str, Any]], template_kwargs: dict[str, Any] | None = None) -> list[str]:
    """Conversations as the text the model is trained on, through the tokenizer's chat template."""
    if not getattr(tokenizer, "chat_template", True):
        raise ValueError(
            "the tokenizer has no chat template; set chat_template to one of Unsloth's names (e.g. chatml)"
        )
    return [
        tokenizer.apply_chat_template(
            row["messages"], tokenize=False, add_generation_prompt=False, **(template_kwargs or {})
        )
        for row in rows
    ]


def render_prompt(tokenizer, messages: list[dict[str, str]], template_kwargs: dict[str, Any] | None = None) -> str:
    """A conversation up to the point where the model answers."""
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **(template_kwargs or {}))


def within_length(backend, tokenizer, texts: list[str], limit: int) -> list[bool]:
    """Which rendered texts fit in `limit` tokens."""
    return [n <= limit for n in backend.count_tokens(tokenizer, texts)]


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
    save_history(ctx.out_dir, out.log_history)  # what `stagetrain report` draws the curves from
    artifacts: dict[str, Any] = {"model": model_dir, "log": write_json(ctx.path("train_log.json"), out.log_history)}
    if any((ctx.out_dir / "checkpoints").glob("checkpoint-*")):
        artifacts["checkpoints"] = ctx.out_dir / "checkpoints"  # for llm.select_checkpoint
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
    train_texts, eval_texts = render(tokenizer, train_rows), render(tokenizer, eval_rows)
    dropped = 0
    if p.overlength == "drop":
        keep = within_length(backend, tokenizer, train_texts, p.max_seq_length)
        dropped = len(keep) - sum(keep)
        train_texts = [t for t, ok in zip(train_texts, keep, strict=True) if ok]
        eval_texts = [t for t, ok in zip(eval_texts, within_length(backend, tokenizer, eval_texts, p.max_seq_length), strict=True) if ok]
        if not train_texts:
            raise ValueError(f"every training row is longer than max_seq_length={p.max_seq_length} tokens")
        if dropped:
            ctx.log.warning("stage %s: dropped %d of %d rows longer than %d tokens", ctx.stage, dropped, len(keep), p.max_seq_length)
    config = sft_config_kwargs(
        p.train, output_dir=stage_dir(ctx, "checkpoints"), seed=seed, has_eval=bool(eval_texts),
        max_seq_length=p.max_seq_length, run_name=f"{ctx.pipeline}-{ctx.stage}", packing=p.packing,
        embedding_learning_rate=embedding_lr(p, info["trains_embeddings"]), bf16=backend.bf16_supported(),
    )
    out = backend.train_text(
        model, tokenizer, train_texts, eval_texts, config, responses_only=masking, resume=p.resume and ctx.resumed,
        history_dir=ctx.out_dir,
    )
    result = finish(ctx, model, tokenizer, out, info, {"train": len(train_texts), "eval": len(eval_texts)})
    result.metrics["train/dropped_overlength"] = float(dropped)
    return result


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
        kwargs = self.params.chat_template_kwargs

        def render(tokenizer, rows):
            return render_chat(tokenizer, rows, kwargs)

        return _text_stage(self, ctx, render, self.params.masking(), False)


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
        out = backend.train_dpo(
            model, tokenizer, train_rows, eval_rows, config, resume=p.resume and ctx.resumed, history_dir=ctx.out_dir
        )
        return finish(ctx, model, tokenizer, out, info, {"train": len(train_rows), "eval": len(eval_rows)})


def reward_trend(log_history: list[dict[str, Any]]) -> dict[str, float]:
    """The mean reward over the first and the last tenth of a run, and the difference: a run that
    learned ends higher than it started."""
    rewards = [float(e["reward"]) for e in log_history if isinstance(e.get("reward"), (int, float))]
    if not rewards:
        return {}
    width = max(1, len(rewards) // 10)
    start, end = sum(rewards[:width]) / width, sum(rewards[-width:]) / width
    return {"train/reward_start": start, "train/reward": end, "train/reward_gain": end - start}


@register("llm.grpo")
class ReinforcementLearning(Stage):
    """Reinforcement learning with verifiable rewards: GRPO and its variants (DAPO, Dr. GRPO, BNPO, GSPO)."""

    Params = GrpoParams

    def run(self, ctx: StageContext) -> StageResult:
        p: GrpoParams = self.params
        seed = self.seed(ctx)
        seed_everything(seed)
        chats = [r for r in read_rows(p.train_data, p.max_train_samples) if len(r.get("messages") or []) >= 2]
        if not chats:
            raise ValueError(f"{p.train_data} holds no conversation with a prompt and a reference answer")
        backend = get_backend()
        model, tokenizer, info = prepare_model(p, ctx, seed)
        if p.fast_inference and info["trains_embeddings"]:
            raise ValueError(
                "fast_inference cannot serve an adapter that trains the embeddings (it came from llm.cpt): "
                "use on_adapter: merge, or fast_inference: false"
            )
        # TRL has no prompt-length limit any more, so the stage applies one itself.
        limit = p.max_prompt_tokens or p.max_seq_length - p.max_completion_length
        prompts = [render_prompt(tokenizer, r["messages"][:-1], p.chat_template_kwargs) for r in chats]
        keep = within_length(backend, tokenizer, prompts, limit)
        rows = [
            {"prompt": r["messages"][:-1], "answer": r["messages"][-1]["content"]}
            for r, ok in zip(chats, keep, strict=True) if ok
        ]
        if not rows:
            raise ValueError(f"every prompt is longer than {limit} tokens")
        functions, weights = build_rewards(p.rewards)
        config = grpo_config_kwargs(
            p.train, output_dir=stage_dir(ctx, "checkpoints"), seed=seed, num_generations=p.num_generations,
            max_completion_length=p.max_completion_length, temperature=p.temperature, top_p=p.top_p,
            loss_type=p.loss_type, beta=p.beta, importance_sampling_level=p.importance_sampling_level,
            reward_weights=weights, mask_truncated_completions=p.mask_truncated_completions,
            template_kwargs=p.chat_template_kwargs, run_name=f"{ctx.pipeline}-{ctx.stage}", bf16=backend.bf16_supported(),
        )
        out = backend.train_grpo(
            model, tokenizer, rows, functions, config, resume=p.resume and ctx.resumed, history_dir=ctx.out_dir
        )
        result = finish(ctx, model, tokenizer, out, info, {"train": len(rows)})
        result.metrics.update(reward_trend(out.log_history))
        result.metrics["train/dropped_overlength"] = float(len(chats) - len(rows))
        return result
