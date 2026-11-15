# Training research: libraries for multi-GPU LLM training

Date: 2026-11-15. Scope: open-source libraries that fine-tune LLMs on several GPUs and support
several training methods, and how each one is driven.

**The question.** Which libraries can train the llm track's stages on more than one GPU, which
training methods does each bring, and how should the pipeline use them?

**Short answer.** Four are worth integrating, in this order: plain TRL with accelerate (as a
second backend that runs in the pipeline's own environment), Unsloth under the same launcher,
and Axolotl and ms-swift through their own command lines. The first is done and tested by running
it; the others are wired up and not run.

## How to read this

Version numbers, licences and release dates come from PyPI and the repositories on the date
above. Behaviour was read from documentation and source, and for TRL from the installed release
(1.18.0). **Nothing was run on a GPU:** every statement about DDP or FSDP behaviour is from
reading, not from a run.

## The libraries

| Library | Release (date) | Methods beyond SFT | Multi-GPU | Driven by |
|---|---|---|---|---|
| **TRL** + accelerate | 1.18.0 (2026-11-13) | continued pretraining, DPO (many losses), KTO, GRPO (8 losses), RLOO, reward models, distillation | DDP, FSDP2, DeepSpeed ZeRO 1 to 3 | Python |
| **Unsloth** | 2026.11.4 (2026-11-11) | the same trainers, patched on import; FP8 training | DDP under `torchrun`; in source also FSDP2 and tensor, pipeline and context parallelism, switched by environment variables | Python |
| **Axolotl** | 0.21.1 (2026-10-27) | DPO, IPO, ORPO, KTO, SimPO, GRPO, quantisation-aware training | DDP by default, FSDP2 (also with QLoRA), DeepSpeed, context, tensor and expert parallelism | YAML + CLI |
| **ms-swift** | 4.6.1 (2026-11-13) | DPO, ORPO, SimPO, KTO, CPO, PPO, a wide GRPO family, generalised knowledge distillation, reward models | DDP, FSDP2, DeepSpeed, Megatron | YAML + CLI |
| LLaMA-Factory | 0.9.5 (2026-05-30) | DPO, ORPO, SimPO, KTO, PPO; no GRPO | DDP, DeepSpeed, FSDP with QLoRA | YAML + CLI |
| verl | 0.9.1 (2026-09-20) | PPO, GRPO, DAPO, RLOO; RL only | Ray with FSDP2 or Megatron | Hydra CLI, Parquet data |

No longer maintained: **torchtune** (last release 2025-04-07; its README says development wound
down) and **AutoTrain Advanced** (2025-01-21; its README points to Axolotl and TRL). Not suited
here: LitGPT (no chat format, no QLoRA on several GPUs), H2O LLM Studio (built around its GUI),
Oumi (a wrapper around older TRL releases).

Every library pins its own versions of transformers and torch, so each of the command-line tools
needs an environment of its own.

## What the pipeline does with each

| Library | Integration | Status |
|---|---|---|
| TRL + accelerate | `backend: hf`: the stages call transformers, PEFT and TRL directly. With `launch: {num_processes: N}` the stage runs as `python -m stagetrain.worker` under `torchrun` or `accelerate launch` | Run here on a tiny random model on a CPU: SFT with full fine-tuning and with LoRA, checkpoint selection, self-play, GRPO, export, and a stage in a worker process. No run on more than one process |
| Unsloth | `backend: unsloth` with the same `launch` block: the trainers are TRL's, so the same worker starts them under `torchrun` | Not run. Unsloth's documentation still calls multi-GPU support upcoming; its source implements it |
| Axolotl | `llm.axolotl`: writes Axolotl's YAML, converts the data, runs `axolotl train config.yaml --num-processes N`, reads back the adapter and the training log | Config keys taken from Axolotl's documentation and example configs. Not run: the tests use a stand-in command |
| ms-swift | `llm.swift`: writes its YAML, runs `swift sft`, `swift pt` or `swift rlhf` with `NPROC_PER_NODE=N` | As above |

Not integrated: LLaMA-Factory (the other three cover what it does, and it has no GRPO), verl and
the other RL engines (they want their own data schema, own all GPUs through Ray, and return
checkpoints that are not PEFT adapters; worth it only when GRPO has to scale past one trainer
with a vLLM server).

## What multi-GPU training requires of a pipeline

Findings that shaped the launcher (`stagetrain.core.launch`):

- **One process per GPU, started by a launcher.** TRL's own documentation now uses `torchrun`;
  `accelerate launch` still works for a script of one's own. The pipeline defaults to `torchrun`.
- **A model that is already loaded must be placed by the script.** TRL forces `device_map=None`
  only when it loads the model itself. The `hf` backend loads each process's copy on that
  process's GPU and never uses `device_map="auto"`.
- **A single process that sees several GPUs is an error for TRL's trainers,** which refuse
  `DataParallel`. The `hf` backend trains on one GPU in that case.
- **`ddp_find_unused_parameters` must be set to false for LoRA.** The default turns out true
  because a PEFT model is not a `PreTrainedModel`.
- **Saving is already guarded.** The trainers write checkpoints from the main process only;
  evaluation is collective. The stage's own files (model, history, result) are written by the
  main process.
- **GRPO's batch rule counts processes.** `per_device_train_batch_size x processes x steps per
  generation` must be a multiple of `num_generations`. The stage checks it before launching.
- **FSDP2** is switched on in the training arguments (`fsdp=True`, `fsdp_config={"version": 2}`).
- **ms-swift splits one model across the GPUs unless `NPROC_PER_NODE` is set.** The stage sets it.
- **Telemetry.** TRL and Axolotl report usage by default. The stages set
  `HF_HUB_DISABLE_TELEMETRY=1` and `AXOLOTL_DO_NOT_TRACK=1`.

## Licences

All the libraries above publish under Apache-2.0, with one thing to check. Unsloth's package
metadata says Apache-2.0, but in the 2026.11.4 wheel 112 of the 205 Python files in the `unsloth`
package carry an `AGPL-3.0-only` header, among them the multi-GPU modules. Read those headers
before distributing something built on them. This is a description of the files, not legal advice.

## Not verified

- Any behaviour on a GPU, and any run with more than one process.
- Unsloth under the launcher, and GRPO or DPO under DDP in Unsloth.
- The Axolotl and ms-swift configs against the tools themselves. In particular: Axolotl's
  `ds_type: json` together with `path`, `test_datasets`, `max_steps` and `load_in_4bit`;
  ms-swift's `adapters` for continuing a LoRA; and where each tool leaves its training log.
- QLoRA under FSDP or ZeRO-3 in plain TRL.

## Sources

- TRL 1.18.0: installed source, and `docs/source/distributing_training.md` of that release
- accelerate 1.15.0: `accelerate launch --help`
- Unsloth 2026.11.4: the wheel's source and https://unsloth.ai/docs/basics/multi-gpu-training-with-unsloth
- Axolotl: https://github.com/axolotl-ai-cloud/axolotl (docs: multi-gpu, cli, rlhf, dataset formats)
- ms-swift: https://github.com/modelscope/ms-swift (docs: command-line parameters, custom dataset, RLHF)
- LLaMA-Factory: https://github.com/hiyouga/LlamaFactory
- verl: https://github.com/verl-project/verl
- torchtune and AutoTrain Advanced: the notices in their READMEs
