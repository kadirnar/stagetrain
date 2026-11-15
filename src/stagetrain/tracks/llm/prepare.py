"""`llm.prepare_data`: sources in whatever shape they come, out as the files the later stages read."""

from __future__ import annotations

import random
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from stagetrain.core import (
    DataSource,
    Stage,
    StageContext,
    StageParams,
    StageResult,
    register,
    write_json,
    write_jsonl,
)

from .data import (
    Collector,
    Rejected,
    chat_chars,
    chunk_text,
    normalise_chat,
    normalise_preference,
    question_key,
    row_key,
    split_rows,
)


class ChatSource(BaseModel):
    """Conversations. Columns by format: `messages` (role/content), `sharegpt` -> `conversations`
    (from/value), `alpaca` -> `instruction`, `input`, `output`, `prompt_completion` -> `prompt`,
    `completion`. An optional `system` column, or `system` here, becomes the system message."""

    model_config = ConfigDict(extra="forbid")

    source: DataSource
    format: Literal["messages", "sharegpt", "alpaca", "prompt_completion"] = "messages"
    system: str | None = None
    # General-purpose data mixed into the training rows so the model keeps what it could already
    # do: this many rows per task row (0.05 = 5 %). Replay rows never reach the eval or test split.
    replay_ratio: float | None = Field(default=None, gt=0.0)


class TextSource(BaseModel):
    """Raw documents (column `text`) for continued pretraining, optionally cut into pieces."""

    model_config = ConfigDict(extra="forbid")

    source: DataSource
    chunk_chars: int | None = None
    chunk_overlap: int = 0


class PreferenceSource(BaseModel):
    """Preference pairs: columns `prompt` (a string or messages), `chosen`, `rejected`."""

    model_config = ConfigDict(extra="forbid")

    source: DataSource
    system: str | None = None


class PrepareParams(StageParams):
    chat: list[ChatSource] = Field(default_factory=list)
    text: list[TextSource] = Field(default_factory=list)
    preference: list[PreferenceSource] = Field(default_factory=list)
    eval_fraction: float = Field(default=0.02, ge=0.0, le=0.5)  # validation: picks checkpoints and settings
    # A second held-out split that nothing is selected on, for the number that gets reported.
    # Choosing the best of many checkpoints on one split makes that split's score optimistic.
    test_fraction: float = Field(default=0.0, ge=0.0, le=0.5)
    # Questions set aside for reinforcement learning, which no SFT or distillation stage trains on.
    # RL on questions the model has already been fine-tuned on has less left to learn.
    rl_fraction: float = Field(default=0.0, ge=0.0, le=0.9)
    max_eval: int | None = 1000
    min_chars: int = 1  # rows shorter or longer than this, counted over message contents, are dropped
    max_chars: int | None = None
    dedupe: bool = True

    @model_validator(mode="after")
    def _some_source(self) -> PrepareParams:
        if not (self.chat or self.text or self.preference):
            raise ValueError("prepare_data needs at least one source under chat, text or preference")
        return self


@register("llm.prepare_data")
class PrepareData(Stage):
    """Normalise chat, raw-text and preference sources; dedupe, filter by length, split train/eval."""

    Params = PrepareParams

    def run(self, ctx: StageContext) -> StageResult:
        from stagetrain.core import load_source

        p: PrepareParams = self.params
        seed = self.seed(ctx)

        def sized(n: int) -> None:
            if n < p.min_chars:
                raise Rejected("too short")
            if p.max_chars is not None and n > p.max_chars:
                raise Rejected("too long")

        def chat_rows(src: ChatSource):
            def convert(raw: dict[str, Any]):
                messages = normalise_chat(raw, src.format, src.system)
                sized(chat_chars(messages))
                yield {"messages": messages}

            return convert

        def text_rows(src: TextSource):
            def convert(raw: dict[str, Any]):
                text = raw.get("text")
                if not isinstance(text, str) or not text.strip():
                    raise Rejected("empty text")
                for piece in chunk_text(text, src.chunk_chars, src.chunk_overlap):
                    try:
                        sized(len(piece))
                    except Rejected as e:
                        collectors["text"].reject(str(e))
                        continue
                    yield {"text": piece}

            return convert

        def preference_rows(src: PreferenceSource):
            def convert(raw: dict[str, Any]):
                row = normalise_preference(raw, src.system)
                sized(chat_chars(row["prompt"]) + len(row["chosen"][0]["content"]))
                yield row

            return convert

        collectors = {kind: Collector(p.dedupe) for kind in ("chat", "text", "preference")}
        plan = (("chat", p.chat, chat_rows), ("text", p.text, text_rows), ("preference", p.preference, preference_rows))
        result = StageResult()
        stats: dict[str, Any] = {}
        for kind, sources, converter in plan:
            if not sources:
                continue
            collector = collectors[kind]
            task_sources = [src for src in sources if not getattr(src, "replay_ratio", None)]
            for src in task_sources:
                collector.extend(load_source(src.source), converter(src))
            if not collector.rows:
                raise ValueError(f"{kind}: no usable row in {len(sources)} source(s); dropped: {collector.rejected}")
            # A conversation's side is decided by its question alone, so two rows that ask the same
            # thing with different answers never end up on both sides of a split.
            key = question_key if kind == "chat" else row_key
            train, test = split_rows(collector.rows, p.test_fraction, p.max_eval, seed + 1, key=key)
            train, held_out = split_rows(train, p.eval_fraction, p.max_eval, seed, key=key)
            reserved: list[dict[str, Any]] = []
            if kind == "chat" and p.rl_fraction > 0:  # before the replay rows join: RL gets task questions only
                train, reserved = split_rows(train, p.rl_fraction, None, seed + 2, key=key)
            replayed = 0
            for src in sources:
                if src in task_sources:
                    continue
                pool = Collector(p.dedupe)
                pool.extend(load_source(src.source), converter(src))
                rows = pool.rows[: round(src.replay_ratio * len(train))]
                replayed += len(rows)
                train = train + rows
            if replayed:
                random.Random(seed).shuffle(train)
            parts = {"train": train, "eval": held_out}
            if reserved:
                parts["rl"] = reserved
            if p.test_fraction > 0:
                parts["test"] = test
            for part, rows in parts.items():
                result.artifacts[f"{kind}_{part}"] = ctx.path(f"{kind}.{part}.jsonl")
                write_jsonl(result.artifacts[f"{kind}_{part}"], rows)
                result.metrics[f"{kind}/{part}_rows"] = len(rows)
            stats[kind] = {**collector.stats(), **{part: len(rows) for part, rows in parts.items()}, "replay": replayed}
            result.metrics.update({
                f"{kind}/duplicates": collector.duplicates,
                f"{kind}/rejected": sum(collector.rejected.values()),
            })
            if replayed:
                result.metrics[f"{kind}/replay_rows"] = replayed
            ctx.log.info("prepare_data: %s %s dropped=%s", kind, {k: len(v) for k, v in parts.items()}, collector.rejected)
        result.artifacts["stats"] = write_json(ctx.path("stats.json"), stats)
        result.notes = stats
        return result
