"""Weight averaging of checkpoints that share one architecture ("model soup")."""

from __future__ import annotations

import json
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


def lora_scale(config: dict) -> float:
    """What PEFT multiplies `B @ A` by: alpha / r, or alpha / sqrt(r) with rank stabilisation."""
    r, alpha = config["r"], config["lora_alpha"]
    return alpha / r**0.5 if config.get("use_rslora") else alpha / r


def average_adapters(
    adapter_dirs: Sequence[str | Path], out_dir: str | Path, weights: Sequence[float] | None = None
) -> Path:
    """Write to `out_dir` one LoRA adapter whose update is the weighted average of the updates
    of the adapters in `adapter_dirs`.

    The update of a LoRA layer is the product `B @ A`, and the average of products is not the
    product of the averages: averaging the `A` and `B` matrices of two runs gives a different,
    smaller update. The exact average is written by stacking instead. With `A` the rows of every
    member's `A` (each scaled by its weight and its own alpha / r) and `B` the columns of every
    member's `B`, `B @ A` is the weighted sum of the members' updates. The result is an ordinary
    adapter of rank `sum(r)` with scale 1, so no base model has to be loaded or copied.

    Every other trained tensor (`modules_to_save`, biases) enters its layer linearly and is
    averaged directly. The members must be adapters of one base model on the same modules.
    """
    import torch
    from safetensors.torch import load_file, save_file

    dirs = [Path(d) for d in adapter_dirs]
    if not dirs:
        raise ValueError("nothing to average")
    if weights is None:
        weights = [1.0] * len(dirs)
    if len(weights) != len(dirs) or sum(weights) <= 0:
        raise ValueError("weights must match the adapters and sum to a positive number")
    norm = [w / sum(weights) for w in weights]

    configs = []
    for d in dirs:
        path = d / "adapter_config.json"
        if not path.exists():
            raise FileNotFoundError(f"{d} is not a LoRA adapter directory (no adapter_config.json)")
        config = json.loads(path.read_text())
        if str(config.get("peft_type", "LORA")).upper() != "LORA":
            raise ValueError(f"{d}: only LoRA adapters can be averaged, this one is {config.get('peft_type')}")
        for key, why in (
            ("use_dora", "DoRA rescales the merged weight, so its update is not a sum of products"),
            ("lora_bias", "a bias on the B matrices cannot be stacked"),
            ("rank_pattern", "per-module ranks are not supported"),
            ("alpha_pattern", "per-module alphas are not supported"),
        ):
            if config.get(key):
                raise ValueError(f"{d}: {key} is set; {why}. Merge the adapters and average full checkpoints instead")
        configs.append(config)
    first = configs[0]
    for d, config in zip(dirs[1:], configs[1:], strict=True):
        if config.get("base_model_name_or_path") != first.get("base_model_name_or_path"):
            raise ValueError(
                f"{d} adapts {config.get('base_model_name_or_path')!r}, {dirs[0]} adapts "
                f"{first.get('base_model_name_or_path')!r}: updates of different base models cannot be averaged"
            )

    tensors = [load_file(d / "adapter_model.safetensors") for d in dirs]
    for d, t in zip(dirs[1:], tensors[1:], strict=True):
        if t.keys() != tensors[0].keys():
            raise ValueError(f"{d} and {dirs[0]} train different modules; they are not adapters of one recipe")
    merged = {}
    for name, ref in tensors[0].items():
        parts = [t[name].to(torch.float32) for t in tensors]
        if "lora_A" in name or "lora_embedding_A" in name:  # (r, in): stack the rows, scaled
            scaled = [x * (w * lora_scale(c)) for x, w, c in zip(parts, norm, configs, strict=True)]
            value = torch.cat(scaled, dim=0)
        elif "lora_B" in name or "lora_embedding_B" in name:  # (out, r): stack the columns
            value = torch.cat(parts, dim=1)
        elif "lora_" in name:
            raise ValueError(f"{name}: a LoRA tensor this function does not know how to average")
        elif ref.is_floating_point():
            if any(x.shape != ref.shape for x in parts):
                raise ValueError(f"{name}: shapes differ between the adapters")
            value = sum(x * w for x, w in zip(parts, norm, strict=True))
        else:
            value = ref
        merged[name] = value.to(ref.dtype) if ref.is_floating_point() else value

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(merged, out_dir / "adapter_model.safetensors", metadata={"format": "pt"})
    rank = sum(c["r"] for c in configs)
    config = {**first, "r": rank, "lora_alpha": rank, "use_rslora": False}
    (out_dir / "adapter_config.json").write_text(json.dumps(config, indent=2, sort_keys=True))
    for p in dirs[0].iterdir():  # tokenizer, chat template, README
        if p.is_file() and p.name not in ("adapter_config.json", "adapter_model.safetensors") and p.suffix != ".safetensors":
            shutil.copy2(p, out_dir / p.name)
    return out_dir
