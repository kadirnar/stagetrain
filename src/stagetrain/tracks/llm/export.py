"""`llm.export`: the trained model in the formats it will be served from."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from stagetrain.core import Stage, StageContext, StageResult, register

from .backend import get_backend
from .train import ModelParams, stage_dir

Format = Literal["lora", "merged_16bit", "merged_4bit", "gguf"]
EXPORT_ORDER = ["lora", "merged_16bit", "gguf", "merged_4bit"]


class ExportParams(ModelParams):
    formats: list[Format] = Field(default_factory=lambda: ["merged_16bit"])
    gguf_quantization: list[str] = Field(default_factory=lambda: ["q4_k_m"])  # llama.cpp quantisations
    push_to_hub: str | None = None  # a repo id such as user/model; nothing is uploaded when unset
    private: bool = True

    @model_validator(mode="after")
    def _formats(self) -> ExportParams:
        if not self.formats or len(set(self.formats)) != len(self.formats):
            raise ValueError("formats must name at least one format, each once")
        if "gguf" in self.formats and not self.gguf_quantization:
            raise ValueError("gguf export needs at least one entry in gguf_quantization")
        return self


def hub_repo(repo: str, fmt: str) -> str:
    """One repository per format: the merged model under the name itself, the others suffixed."""
    return {"merged_16bit": repo, "merged_4bit": f"{repo}-4bit", "lora": f"{repo}-lora", "gguf": f"{repo}-GGUF"}[fmt]


@register("llm.export")
class Export(Stage):
    """Save the model as a LoRA adapter, a merged checkpoint and/or GGUF files; optionally push them."""

    Params = ExportParams

    def run(self, ctx: StageContext) -> StageResult:
        p: ExportParams = self.params
        backend = get_backend(p.backend)
        model, tokenizer = backend.load(p.load_spec())
        if p.chat_template:
            tokenizer = backend.set_chat_template(tokenizer, p.chat_template)
        if "lora" in p.formats and not backend.has_adapter(model):
            raise ValueError(f"{p.init_from} is a full model with no adapter, so there is no 'lora' to export")

        artifacts: dict[str, Any] = {}
        metrics: dict[str, float] = {"export/formats": float(len(p.formats))}
        # A fixed order, whatever the config lists: the adapter first, and the 4-bit merge last
        # because it is the one export that may rewrite the loaded weights.
        for fmt in sorted(p.formats, key=EXPORT_ORDER.index):
            out = stage_dir(ctx, fmt)
            ctx.log.info("stage %s: exporting %s to %s", ctx.stage, fmt, out)
            if fmt == "lora":
                backend.save(model, tokenizer, out)
                artifacts[fmt] = out
            elif fmt == "gguf":
                directory, files = backend.save_gguf(model, tokenizer, out, p.gguf_quantization)
                artifacts[fmt] = directory
                metrics["export/gguf_files"] = float(len(files))
            else:
                backend.save_merged(model, tokenizer, out, fmt)
                artifacts[fmt] = out
            if p.push_to_hub:
                repo = hub_repo(p.push_to_hub, fmt)
                ctx.log.info("stage %s: pushing %s to %s (private=%s)", ctx.stage, fmt, repo, p.private)
                backend.push(model, tokenizer, repo, fmt, private=p.private, quantization=p.gguf_quantization)
        model = tokenizer = None
        backend.release()
        # `model` is what a later stage would load: the merged checkpoint if there is one, else the adapter.
        for fmt in ("merged_16bit", "merged_4bit", "lora"):
            if fmt in artifacts:
                artifacts["model"] = artifacts[fmt]
                break
        return StageResult(artifacts=artifacts, metrics=metrics, notes={"pushed_to": p.push_to_hub})
