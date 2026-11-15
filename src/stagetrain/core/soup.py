"""Weight averaging of checkpoints that share one architecture ("model soup")."""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from pathlib import Path


def weight_files(model_dir: Path) -> list[str]:
    files = sorted(p.name for p in Path(model_dir).glob("*.safetensors") if not p.name.startswith("adapter_"))
    if not files:
        raise FileNotFoundError(
            f"{model_dir}: no *.safetensors weights; a soup needs full checkpoints, so merge adapters first"
        )
    return files


def average_checkpoints(
    model_dirs: Sequence[str | Path], out_dir: str | Path, weights: Sequence[float] | None = None
) -> Path:
    """Write to `out_dir` the weighted average of the checkpoints in `model_dirs`.

    Floating-point tensors are averaged in fp32 and cast back; integer tensors and every other
    file (config, tokenizer) are taken from the first checkpoint. One shard is held in memory at
    a time.
    """
    import torch
    from safetensors.torch import load_file, save_file

    dirs = [Path(d) for d in model_dirs]
    if not dirs:
        raise ValueError("nothing to average")
    if weights is None:
        weights = [1.0] * len(dirs)
    if len(weights) != len(dirs) or sum(weights) <= 0:
        raise ValueError("weights must match the checkpoints and sum to a positive number")
    norm = [w / sum(weights) for w in weights]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = weight_files(dirs[0])
    for d in dirs[1:]:
        if weight_files(d) != files:
            raise ValueError(f"{d} and {dirs[0]} hold different weight files; they are not one architecture")
    for name in files:
        first = load_file(dirs[0] / name)
        acc = {k: v.to(torch.float32) * norm[0] for k, v in first.items() if v.is_floating_point()}
        for d, w in zip(dirs[1:], norm[1:], strict=True):
            other = load_file(d / name)
            if other.keys() != first.keys():
                raise ValueError(f"{d / name}: tensor names differ from {dirs[0] / name}")
            for k in acc:
                if other[k].shape != first[k].shape:
                    raise ValueError(f"{k}: shape {tuple(other[k].shape)} in {d}, {tuple(first[k].shape)} in {dirs[0]}")
                acc[k] += other[k].to(torch.float32) * w
            del other
        merged = {k: acc[k].to(v.dtype) if k in acc else v for k, v in first.items()}
        save_file(merged, out_dir / name, metadata={"format": "pt"})
    for p in dirs[0].iterdir():
        if p.is_file() and p.suffix != ".safetensors" and not (out_dir / p.name).exists():
            shutil.copy2(p, out_dir / p.name)
    return out_dir
