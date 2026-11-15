"""The model as it ships: one directory, optionally truncated, optionally with an ONNX copy."""

from __future__ import annotations

import importlib.util
from typing import Any

from pydantic import Field

from ...core.data import write_json
from ...core.stage import Stage, StageContext, StageParams, StageResult, register
from .model import EncoderSpec, load_encoder

ONNX_LIBRARIES = ("optimum", "onnxruntime")


class ExportParams(StageParams):
    model: str
    truncate_dim: int | None = None  # ship a Matryoshka model at a smaller dimension
    normalize: bool = False  # end the model in a Normalize module, so dot product equals cosine
    half: bool = False  # save fp16 weights
    onnx: bool = False  # also write an ONNX copy; skipped with a note when optimum / onnxruntime are missing
    hub_repo: str | None = None  # push the exported model to this repository
    hub_private: bool = True
    encoder: EncoderSpec = Field(default_factory=EncoderSpec)


def missing_onnx_libraries() -> list[str]:
    return [lib for lib in ONNX_LIBRARIES if importlib.util.find_spec(lib) is None]


@register("embeddings.export")
class Export(Stage):
    """Write the final model directory (optionally truncated or normalised), an optional ONNX copy, and a summary."""

    Params = ExportParams

    def run(self, ctx: StageContext) -> StageResult:
        from sentence_transformers import SentenceTransformer

        p: ExportParams = self.params
        spec = p.encoder.model_copy(update={"normalize": True}) if p.normalize else p.encoder
        model = load_encoder(p.model, spec, truncate_dim=p.truncate_dim)
        with model.truncate_embeddings(None):
            full = model.get_embedding_dimension()
        if p.truncate_dim is not None and full is not None and p.truncate_dim > full:
            raise ValueError(f"truncate_dim={p.truncate_dim} is larger than the model's dimension {full}")
        if p.half:
            model.half()
        out = ctx.out_dir / "model"
        model.save_pretrained(str(out))

        # Load what was written and encode once: the export is only done if the saved files work.
        reloaded = SentenceTransformer(str(out))
        vectors = reloaded.encode(["export check", "a second text"], show_progress_bar=False)
        dim = int(vectors.shape[-1])
        if dim != reloaded.get_embedding_dimension():
            raise RuntimeError(f"exported model encodes to {dim} dimensions but reports {reloaded.get_embedding_dimension()}")

        notes: dict[str, Any] = {"dimension": dim, "full_dimension": full, "prompts": dict(reloaded.prompts or {})}
        artifacts: dict[str, Any] = {"model": out}
        if p.onnx:
            missing = missing_onnx_libraries()
            if missing:
                notes["onnx"] = f"skipped: {', '.join(missing)} not installed (pip install 'sentence-transformers[onnx]')"
                ctx.log.warning("export: ONNX %s", notes["onnx"])
            else:
                onnx_dir = ctx.out_dir / "onnx"
                SentenceTransformer(str(out), backend="onnx").save_pretrained(str(onnx_dir))
                artifacts["onnx"] = onnx_dir
                notes["onnx"] = "written"
        if p.hub_repo:
            notes["hub_url"] = reloaded.push_to_hub(p.hub_repo, private=p.hub_private, exist_ok=True)
        artifacts["summary"] = write_json(ctx.path("export.json"), notes)
        metrics = {"export/dimension": float(dim), "export/onnx": float(notes.get("onnx") == "written")}
        return StageResult(artifacts=artifacts, metrics=metrics, notes=notes)
