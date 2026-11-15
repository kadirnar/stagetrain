"""`llm.prepare_data`: sources in whatever shape they come, out as the files the later stages read."""

from __future__ import annotations

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
    eval_fraction: float = Field(default=0.02, ge=0.0, le=0.5)
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
            for src in sources:
                collector.extend(load_source(src.source), converter(src))
            if not collector.rows:
                raise ValueError(f"{kind}: no usable row in {len(sources)} source(s); dropped: {collector.rejected}")
            train, held_out = split_rows(collector.rows, p.eval_fraction, p.max_eval, seed)
            result.artifacts[f"{kind}_train"] = ctx.path(f"{kind}.train.jsonl")
            result.artifacts[f"{kind}_eval"] = ctx.path(f"{kind}.eval.jsonl")
            write_jsonl(result.artifacts[f"{kind}_train"], train)
            write_jsonl(result.artifacts[f"{kind}_eval"], held_out)
            stats[kind] = {**collector.stats(), "train": len(train), "eval": len(held_out)}
            result.metrics.update({
                f"{kind}/train_rows": len(train),
                f"{kind}/eval_rows": len(held_out),
                f"{kind}/duplicates": collector.duplicates,
                f"{kind}/rejected": sum(collector.rejected.values()),
            })
            ctx.log.info("prepare_data: %s train=%d eval=%d dropped=%s", kind, len(train), len(held_out), collector.rejected)
        result.artifacts["stats"] = write_json(ctx.path("stats.json"), stats)
        result.notes = stats
        return result
