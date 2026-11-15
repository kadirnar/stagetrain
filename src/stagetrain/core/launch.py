"""Running a stage in its own process, on one GPU or several.

A pipeline is one Python process. Training on several GPUs needs one process per GPU, started by
a launcher (`accelerate launch`, `torchrun`). A stage that can be launched writes what it was
asked to do into `job.json`, starts `python -m stagetrain.worker job.json` under the launcher,
and reads the result the main process of that run left in `result.json`.

Every process of the run executes the stage's `execute` method; the trainer inside it spreads
the work, and only the main process writes files (`is_main_process`).

A stage can also be launched with one process (`subprocess: true`). That costs a model load, and
gives the GPU back completely when the stage ends, whatever the libraries left allocated.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .stage import Stage, StageContext, StageResult

JOB, RESULT = "job.json", "result.json"
WORKER_ENV = "STAGETRAIN_WORKER"
FSDP_ENV = "STAGETRAIN_FSDP"


class LaunchSpec(BaseModel):
    """How a training stage is started.

        launch: {num_processes: 4}                       # four GPUs, one process each (DDP) under torchrun
        launch: {num_processes: 2, gpu_ids: "2,3"}
        launch: {num_processes: 4, launcher: accelerate}
        launch: {num_processes: 4, fsdp: true}           # shard one model over the GPUs (FSDP2) instead
        launch: {subprocess: true}                       # one GPU, in a process of its own

    The effective batch size is `per_device_train_batch_size * gradient_accumulation_steps *
    num_processes`: when the number of processes changes, change one of the others to keep it.
    """

    model_config = ConfigDict(extra="forbid")

    num_processes: int = Field(default=1, ge=1)  # one per GPU
    launcher: Literal["torchrun", "accelerate", "python"] = "torchrun"  # torchrun is what TRL documents
    subprocess: bool = False  # start a process even for one GPU
    gpu_ids: str | None = None  # which GPUs, e.g. "0,1"; default: the first `num_processes` visible ones
    # FSDP2: the model's parameters are sharded across the GPUs, for full fine-tuning of a model
    # that does not fit on one. The hf backend turns it on in the training arguments.
    fsdp: bool = False
    extra_args: list[str] = Field(default_factory=list)  # passed to the launcher as they are
    env: dict[str, str] = Field(default_factory=dict)

    @property
    def active(self) -> bool:
        return self.num_processes > 1 or self.subprocess

    def command(self, job: str | Path) -> list[str]:
        """The command that starts the worker. The launchers are run as modules of this
        interpreter, so they come from the same environment as the pipeline."""
        worker = ["-m", "stagetrain.worker", str(job)]
        if self.launcher == "python":
            if self.num_processes > 1:
                raise ValueError("launcher: python starts one process; use accelerate or torchrun for several")
            return [sys.executable, *worker]
        if self.launcher == "torchrun":
            return [
                sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", str(self.num_processes),
                *self.extra_args, *worker,
            ]
        args = ["--num_processes", str(self.num_processes)]
        if self.num_processes > 1:
            args.append("--multi_gpu")
        if self.gpu_ids:
            args += ["--gpu_ids", self.gpu_ids]
        return [sys.executable, "-m", "accelerate.commands.accelerate_cli", "launch", *args, *self.extra_args, *worker]

    def environment(self) -> dict[str, str]:
        env = {**os.environ, **self.env, WORKER_ENV: "1"}
        if self.gpu_ids and self.launcher != "accelerate":
            env["CUDA_VISIBLE_DEVICES"] = self.gpu_ids
        if self.fsdp:
            env[FSDP_ENV] = "1"
        return env


def in_worker() -> bool:
    return os.environ.get(WORKER_ENV) == "1"


def is_main_process() -> bool:
    """True in the process that writes a run's files: always, unless this is a secondary process
    of a distributed run."""
    if not in_worker():
        return True
    for name in ("RANK", "LOCAL_RANK"):
        if os.environ.get(name) not in (None, "", "0"):
            return False
    return True


def fsdp_requested() -> bool:
    return in_worker() and os.environ.get(FSDP_ENV) == "1"


def world_size() -> int:
    """How many processes share this run (1 outside a distributed run)."""
    for name in ("WORLD_SIZE", "ACCELERATE_NUM_PROCESSES"):
        value = os.environ.get(name)
        if in_worker() and value and value.isdigit():
            return int(value)
    return 1


def write_job(stage: Stage, ctx: StageContext, num_processes: int) -> Path:
    path = ctx.out_dir / JOB
    job = {
        "kind": stage.kind,
        "params": stage.params.model_dump(mode="json"),
        "num_processes": num_processes,
        "ctx": {
            "track": ctx.track, "pipeline": ctx.pipeline, "stage": ctx.stage, "out_dir": str(ctx.out_dir),
            "run_dir": str(ctx.run_dir), "seed": ctx.seed, "resumed": ctx.resumed,
        },
    }
    path.write_text(json.dumps(job, indent=2) + "\n", encoding="utf-8")
    return path


def read_result(out_dir: Path) -> StageResult:
    path = out_dir / RESULT
    if not path.is_file():
        raise RuntimeError(f"the launched stage left no {RESULT} in {out_dir}; see its output above")
    data = json.loads(path.read_text(encoding="utf-8"))
    return StageResult(artifacts=data["artifacts"], metrics=data["metrics"], notes=data.get("notes") or {})


def write_result(out_dir: Path, result: StageResult) -> None:
    data: dict[str, Any] = {
        "artifacts": {k: str(v) for k, v in result.artifacts.items()},
        "metrics": {k: float(v) for k, v in result.metrics.items()},
        "notes": result.notes,
    }
    tmp = out_dir / (RESULT + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")
    tmp.replace(out_dir / RESULT)


class LaunchableStage(Stage):
    """A stage that can run in the pipeline's process or be launched. Subclasses implement
    `execute`; their parameters carry `launch: LaunchSpec`."""

    def execute(self, ctx: StageContext) -> StageResult:
        raise NotImplementedError

    def run(self, ctx: StageContext) -> StageResult:
        spec: LaunchSpec | None = getattr(self.params, "launch", None)
        if spec is None or not spec.active or in_worker():
            return self.execute(ctx)
        job = write_job(self, ctx, spec.num_processes)
        (ctx.out_dir / RESULT).unlink(missing_ok=True)
        command = spec.command(job)
        ctx.log.info("stage %s: launching %d process(es): %s", ctx.stage, spec.num_processes, " ".join(command))
        # The run's output goes straight to the terminal: a training run is watched while it runs.
        done = subprocess.run(command, env=spec.environment(), check=False)
        if done.returncode != 0:
            raise RuntimeError(f"stage {ctx.stage!r}: the launched run exited with code {done.returncode}; see its output above")
        result = read_result(ctx.out_dir)
        result.notes = {**result.notes, "launch": {"num_processes": spec.num_processes, "launcher": spec.launcher}}
        return result
