"""Loading a bi-encoder: a saved SentenceTransformer, or a plain encoder with a pooling head put on it."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

QUERY, DOCUMENT = "query", "document"


class PromptSpec(BaseModel):
    """Instruction prefixes, e.g. `{query: "query: ", document: "passage: "}`. They are stored in
    the model under the names `query` and `document`, which `encode_query` / `encode_document`,
    the retrieval evaluator and the miner pick up."""

    model_config = ConfigDict(extra="forbid")

    query: str | None = None
    document: str | None = None

    def named(self) -> dict[str, str]:
        return {k: v for k, v in ((QUERY, self.query), (DOCUMENT, self.document)) if v}

    def for_columns(self, columns: list[str], query_column: str = "query") -> dict[str, str]:
        """Per training column: the query prompt on the query column, the document prompt on the rest."""
        out = {}
        for col in columns:
            prompt = self.query if col == query_column else self.document
            if prompt:
                out[col] = prompt
        return out


class EncoderSpec(BaseModel):
    """How to load the encoder.

    `pooling: auto` loads whatever the checkpoint is (a SentenceTransformer directory or Hub
    repository, or a plain encoder, which the library wraps with mean pooling). Any other value
    builds `Transformer -> Pooling [-> Normalize]` explicitly; it only applies to a plain encoder
    and is ignored for a checkpoint that already is a SentenceTransformer.
    """

    model_config = ConfigDict(extra="forbid")

    pooling: Literal["auto", "mean", "cls", "lasttoken", "max"] = "auto"
    normalize: bool | None = None  # True: make sure the model ends in a Normalize module
    max_seq_length: int | None = None
    prompts: PromptSpec | None = None  # None: keep the prompts saved with the checkpoint
    similarity: Literal["cosine", "dot"] | None = None
    trust_remote_code: bool = False
    model_kwargs: dict[str, Any] | None = None
    device: str | None = None  # None: the library picks (cuda, mps, cpu)


def is_sentence_transformer_dir(path: str) -> bool:
    return (Path(path) / "modules.json").is_file()


def load_encoder(path: str, spec: EncoderSpec | None = None, device: str | None = None, truncate_dim: int | None = None):
    """A `SentenceTransformer` from a Hub id or a local directory."""
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.sentence_transformer.modules import Normalize, Pooling, Transformer

    spec = spec or EncoderSpec()
    common: dict[str, Any] = {"device": device or spec.device, "truncate_dim": truncate_dim}
    if spec.similarity:
        common["similarity_fn_name"] = spec.similarity
    if spec.prompts is not None and spec.prompts.named():
        common["prompts"] = spec.prompts.named()

    if spec.pooling == "auto" or is_sentence_transformer_dir(path):
        model = SentenceTransformer(
            path, trust_remote_code=spec.trust_remote_code, model_kwargs=spec.model_kwargs, **common
        )
    else:
        remote = {"trust_remote_code": True} if spec.trust_remote_code else {}
        transformer = Transformer(
            path, model_kwargs={**remote, **(spec.model_kwargs or {})} or None,
            processor_kwargs=remote or None, config_kwargs=remote or None,
        )
        pooling = Pooling(transformer.get_embedding_dimension(), pooling_mode=spec.pooling)
        model = SentenceTransformer(modules=[transformer, pooling], **common)
    if spec.normalize and not isinstance(model[len(model) - 1], Normalize):
        model.append(Normalize())
    if spec.max_seq_length is not None:
        model.max_seq_length = spec.max_seq_length
    return model


def prompt_names(model) -> dict[str, str]:
    """`{query_prompt_name, corpus_prompt_name}` for the prompts the model carries, as the
    evaluator and the miner take them."""
    prompts = getattr(model, "prompts", None) or {}
    out = {}
    if prompts.get(QUERY):
        out["query_prompt_name"] = QUERY
    if prompts.get(DOCUMENT):
        out["corpus_prompt_name"] = DOCUMENT
    return out
