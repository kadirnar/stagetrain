"""Stands in for the `axolotl` and `swift` commands in the tests: it reads the config a stage
wrote, records what it was called with, and leaves the files a real run would leave."""

import json
import os
import sys
from pathlib import Path

import yaml


def main() -> int:
    argv = sys.argv[1:]
    config_path = next(a for a in argv if a.endswith(".yaml"))
    config = yaml.safe_load(Path(config_path).read_text())
    if config.get("fail"):
        print("the tool rejects this config", file=sys.stderr)
        return 3
    out = Path(config["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    seen = {
        "argv": argv, "config": config, "cwd": os.getcwd(),
        "env": {k: os.environ.get(k) for k in ("AXOLOTL_DO_NOT_TRACK", "NPROC_PER_NODE", "CUDA_VISIBLE_DEVICES",
                                                "HF_HUB_DISABLE_TELEMETRY", "EXTRA")},
    }
    (out.parent / "seen.json").write_text(json.dumps(seen))
    state = {"log_history": [
        {"step": 10, "epoch": 0.5, "loss": 1.5, "learning_rate": 1e-4},
        {"step": 20, "epoch": 1.0, "loss": 1.1, "learning_rate": 0.0},
        {"step": 20, "epoch": 1.0, "eval_loss": 1.2},
        {"step": 20, "epoch": 1.0, "train_runtime": 3.0, "train_loss": 1.3},
    ]}
    checkpoint = out / "checkpoint-20"
    checkpoint.mkdir(exist_ok=True)
    (checkpoint / "trainer_state.json").write_text(json.dumps(state))
    adapter = {"base_model_name_or_path": config.get("base_model") or config.get("model")}
    if "base_model" in config:  # axolotl leaves the adapter in output_dir
        (out / "adapter_config.json").write_text(json.dumps(adapter))
    else:  # ms-swift leaves checkpoints and a `last` link
        (checkpoint / "adapter_config.json").write_text(json.dumps(adapter))
        (out / "last").symlink_to(checkpoint, target_is_directory=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
