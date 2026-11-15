"""`python -m stagetrain.worker job.json`: one process of a launched stage.

Started by `LaunchableStage.run` under `accelerate launch` or `torchrun`. Every process of the
run executes the stage; the one that is the main process writes `result.json`.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

from .core.launch import WORKER_ENV, is_main_process, write_result
from .core.stage import StageContext, get_stage_class


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: python -m stagetrain.worker <job.json>", file=sys.stderr)
        return 2
    os.environ[WORKER_ENV] = "1"
    job = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    cls = get_stage_class(job["kind"])
    stage = cls(cls.Params.model_validate(job["params"]))
    c = job["ctx"]
    ctx = StageContext(
        track=c["track"], pipeline=c["pipeline"], stage=c["stage"], out_dir=Path(c["out_dir"]),
        run_dir=Path(c["run_dir"]), seed=c["seed"], resumed=c["resumed"],
    )
    result = stage.execute(ctx)
    if is_main_process():
        write_result(ctx.out_dir, result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
