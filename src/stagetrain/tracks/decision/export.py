"""Stage `decision.export`: the final model directory, with its prompt style and calibration."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Literal

from stagetrain.core import Stage, StageContext, StageParams, StageResult, register

from .modeling import META_FILE, read_meta, write_meta
from .prompt import PromptStyle


class ExportParams(StageParams):
    model: str
    calibration: str | None = None  # a `decision.calibrate` stage's `calibration`
    precision: Literal["keep", "bf16", "fp16", "fp32"] = "keep"
    push_to_hub: str | None = None  # a repo id; nothing is uploaded unless this is set
    private: bool = True
    trust_remote_code: bool = False


@register("decision.export")
class Export(Stage):
    """Package the model with `decision.json` so `DecisionModel.load` reads it as it was trained."""

    Params = ExportParams

    def run(self, ctx: StageContext) -> StageResult:
        p = self.params
        out = ctx.out_dir / "model"
        shutil.rmtree(out, ignore_errors=True)
        meta = read_meta(p.model)
        if p.precision == "keep":
            shutil.copytree(p.model, out)
        else:
            from .modeling import load_model, load_tokenizer

            model = load_model(p.model, p.precision, p.trust_remote_code, device="cpu")
            model.save_pretrained(str(out))
            load_tokenizer(p.model, p.trust_remote_code).save_pretrained(str(out))
        calibration = meta.get("calibration")
        if p.calibration:
            fitted = json.loads(Path(p.calibration).read_text(encoding="utf-8"))
            calibration = {"temperature": fitted["temperature"], "bands": fitted.get("bands") or {}}
        # `init_from` is a path on the training machine; it has no place in a published model.
        kept = ("format", "prompt", "max_length", "calibration", "init_from")
        extra = {k: v for k, v in meta.items() if k not in kept}
        write_meta(out, PromptStyle.model_validate(meta.get("prompt") or {}), meta.get("max_length"), calibration, extra)
        size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
        metrics = {"export/size_gb": size / 1e9, "export/calibrated": float(bool(calibration))}
        if p.push_to_hub:
            from huggingface_hub import HfApi

            api = HfApi()
            api.create_repo(p.push_to_hub, private=p.private, exist_ok=True)
            api.upload_folder(repo_id=p.push_to_hub, folder_path=str(out))
        return StageResult(artifacts={"model": out, "meta": out / META_FILE}, metrics=metrics)
