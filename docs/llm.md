# llm track: multi-stage fine-tuning, on one GPU or several

The track trains a causal language model in stages, with the goal of the highest task accuracy.
It trains with Unsloth or with plain TRL and PEFT, on one GPU or several, and can hand a stage to
Axolotl or ms-swift.
Which stages exist, their order and their defaults follow [docs/research/llm.md](research/llm.md):
measure first, sweep the learning rate, choose the checkpoint by the task metric, add verified
data, then post-train with preference optimisation or reinforcement learning.

```
prepare_data ─┬─> eval (base) ──> eval (prompt-only)
              └─> check_data ──> sft x N (matrix) ──> select ──> select_checkpoint ──> eval
                                                        └──> generate_data ─┬─> dpo ─┐
                                                                            └────────┴─> grpo ──> eval ──> export
```

Every stage starts from a base model or from the `model` artifact of an earlier stage, so stages
can be left out, repeated and reordered. Example pipelines in `configs/llm/`:

| Config | What it is |
|---|---|
| `multi-gpu.yaml` | LoRA SFT on four GPUs with the `hf` backend, and full fine-tuning with FSDP2 |
| `other-trainers.yaml` | The same data through Axolotl and ms-swift, each on several GPUs |
| `accuracy.yaml` | The research-driven recipe on a task with checkable answers: baselines, a learning-rate sweep, checkpoint selection, self-play, GRPO, paired evaluations |
| `qwen3-multistage.yaml` | Continued pretraining, two SFT passes, DPO, export |
| `sft-only.yaml` | The shortest useful pipeline |

## Stages

| Kind | What it does | Why it is there |
|---|---|---|
| `llm.prepare_data` | Normalises chat, raw-text and preference sources; drops invalid rows and duplicates; splits by question into train, validation (`eval`) and optionally test; mixes in general data (`replay_ratio`) | Two rows asking the same thing never sit on both sides of a split. Validation is selected on; test is only reported |
| `llm.check_data` | Checks the training rows for defects, duplicates, leaks into the held-out rows and private data; drops what is certainly useless, flags the rest, writes a report | Wrong answers and train/eval overlap are the two defects with a measured cost ([research](research/llm-data.md)) |
| `llm.eval` | Generates answers and scores them with a task scorer; reports the mean with its standard error and 95 % interval; compares with another evaluation item by item (`compare_to`); optional few-shot and system-prompt baselines; optional held-out loss | A loss cannot rank models across recipes, and an unpaired comparison needs thousands of rows to see three points |
| `llm.cpt` | Continued pretraining on raw text, with the embeddings and output head trained at a lower learning rate | For domains whose vocabulary the base model lacks. Not in the accuracy recipe: raw-text pretraining tends to damage prompting |
| `llm.sft` | Chat fine-tuning through the tokenizer's chat template, loss on assistant turns only, a checkpoint per epoch | The main stage. Run it as a `matrix` over the learning rate |
| `llm.select` | The best run of a matrix by one metric; hands on that run's artifacts | The learning rate is the most sensitive setting |
| `llm.select_checkpoint` | Answers the validation questions with every checkpoint of a training stage and keeps the best (or, with `tolerance`, the earliest one close to the best) | Validation loss bottoms out early while answers keep improving. Before RL, a less-trained checkpoint explores more |
| `llm.generate_data` | A model answers training prompts several times without seeing the references; a scorer sorts the answers | Verified answers to train on (from a teacher or the model itself), right/wrong pairs for DPO, and the share of prompts RL can still learn from |
| `llm.dpo` | Direct preference optimisation on prompt / chosen / rejected | Helps at about 7B and above in the available evidence; not below |
| `llm.grpo` | Reinforcement learning with verifiable rewards (TRL's `GRPOTrainer`): DAPO by default, also Dr. GRPO, BNPO, GRPO, CISPO, and GSPO through `importance_sampling_level: sequence` | A few points on top of SFT when answers can be checked and the model already solves part of the prompts |
| `llm.export` | LoRA adapter, merged 16-bit or 4-bit checkpoint, GGUF; optional Hub push, private by default | Runs once, after the gates |
| `llm.axolotl`, `llm.swift` | SFT, continued pretraining or DPO run by Axolotl's or ms-swift's command line: the stage writes the library's config, runs it, and reads back the adapter and the training log | Their multi-GPU modes and methods, without leaving the pipeline |

### Scorers

One scorer definition is used by `llm.eval`, `llm.select_checkpoint`, `llm.generate_data` and the
`correct` reward of `llm.grpo`, so "correct" means the same thing everywhere.

| `scorer.name` | Correct when |
|---|---|
| `exact` | the answer equals the reference, case and whitespace aside |
| `contains` | the reference appears in the answer |
| `numeric` | the last number in the answer equals the last number in the reference |
| `choice` | the option letter the answer settles on equals the reference's |
| `regex` | `pattern`'s first group in the answer equals the reference |
| `json` | the answer parses as JSON and equals the reference, on `fields` if given |
| `custom` | `function: "package.module:name"`, called as `f(output, reference, row)`, returns 0 to 1 |

By default a reasoning block (`<think>...</think>`) is removed before scoring. `eval/no_answer`
is the share of answers that are empty once it is removed, which is what a generation that ran
out of tokens inside its reasoning looks like.

### Rewards for `llm.grpo`

```yaml
rewards:
  - {kind: correct, scorer: {name: numeric}, weight: 1.0}
  - {kind: format, pattern: '####\s*-?[\d,.]+\s*$', weight: 0.1}
```

Kinds: `correct` (a scorer), `format` (the answer matches a pattern), `reasoning` (a closed,
non-empty reasoning block followed by an answer), `length` (a penalty past `max_chars`), `custom`.
Each term is logged on its own (`train/rewards/<name>/mean`); the weights combine them. Keep
`correct` dominant: the other terms can be collected without solving anything.

The stage reports `train/reward_start`, `train/reward` and `train/reward_gain` (the mean reward
over the first and last tenth of the run), so a gate can require that the run learned. The
held-out reward is measured afterwards by `llm.eval`, which decodes greedily; TRL's own
evaluation samples at the training temperature.

### What a gate can say

```yaml
gate:
  - {metric: eval/delta_ci_low, min: 0.0}            # better than the compared model, beyond chance
  - {metric: eval/score, baseline: eval_sft.eval/score, max_drop: 0.0}
  - {metric: generate/mixed_fraction, min: 0.1}       # RL has something to learn from
  - {metric: train/reward_gain, min: 0.0}
  - {metric: eval/no_answer, max: 0.05}
  - {metric: eval/valid_reasoning, min: 0.95}         # with expect_reasoning: true
```

### Reasoning models

Pass the template's switch wherever prompts are rendered, with the same value in every stage:
`chat_template_kwargs: {enable_thinking: false}`. To keep a model's reasoning ability, train on
answers that contain reasoning (`llm.generate_data` with `require_reasoning: true` produces them)
and evaluate with `expect_reasoning: true`. `mask_out_tokens` on `llm.sft` is passed to Unsloth's
`train_on_responses_only`.

## Backends

Every stage that loads a model takes `backend:`.

| | `unsloth` (default) | `hf` |
|---|---|---|
| What it is | Unsloth's loaders and kernels over TRL's trainers | transformers + PEFT + TRL + accelerate, nothing else |
| Needs | a CUDA GPU, and an environment with Unsloth's older pins | whatever transformers runs on |
| Loss on assistant turns | Unsloth's `train_on_responses_only` | the stage's own masking, from character offsets; keeps the end-of-turn token in the loss |
| GRPO generation with vLLM | yes (`fast_inference`) | no |
| Export | LoRA, merged 16-bit and 4-bit, GGUF | LoRA, merged 16-bit |
| Tested by running it | no | yes, on a tiny random model on a CPU |

A pipeline can mix them: train with one, evaluate or export with the other. An adapter saved by
either is a PEFT adapter and loads in both.

## Several GPUs

A training stage (`llm.cpt`, `llm.sft`, `llm.dpo`, `llm.grpo`) takes a `launch` block:

```yaml
launch: {num_processes: 4}                       # one process per GPU (DDP) under torchrun
launch: {num_processes: 2, gpu_ids: "2,3"}
launch: {num_processes: 4, launcher: accelerate}
launch: {num_processes: 4, fsdp: true}           # hf backend: shard one model over the GPUs (FSDP2)
launch: {subprocess: true}                       # one GPU, in a process of its own
```

The pipeline stays one process. A launched stage writes `job.json`, starts
`python -m stagetrain.worker job.json` under the launcher, and reads `result.json` back. Every
process of the run executes the stage; the trainer spreads the work and only the main process
writes files. The stages around it (evaluation, selection, export) run in the pipeline's process
on one GPU.

- The effective batch size is `per_device_train_batch_size x gradient_accumulation_steps x
  num_processes`. Change one factor when the number of GPUs changes.
- For `llm.grpo` that product must be a multiple of `num_generations`; the stage checks it before
  anything starts.
- `subprocess: true` costs a model load and returns the GPU's memory completely when the stage
  ends.
- With Unsloth the same block applies (its trainers are TRL's, and it switches DDP on under
  `torchrun`). That path was not run; see [docs/research/llm-libraries.md](research/llm-libraries.md).

## Other trainers

`llm.axolotl` and `llm.swift` hand a training stage to another library:

```yaml
- name: sft
  kind: llm.axolotl                   # or llm.swift
  params:
    method: sft                       # sft, cpt or dpo
    init_from: Qwen/Qwen3-4B          # or an earlier stage's adapter, which training then continues
    train_data: ${quality.chat}
    num_processes: 4
    lora: {r: 64, alpha: 32}
    train: {learning_rate: 2.0e-4, num_train_epochs: 3}
    executable: [/opt/axolotl/bin/axolotl]
    config: {sample_packing: true}    # the library's own keys, written as they are
```

The stage writes `<stage>/config.yaml`, converts preference pairs and raw text to the library's
row format, runs `axolotl train config.yaml --num-processes 4` (or `swift sft config.yaml` with
`NPROC_PER_NODE=4`), and returns the adapter as `model` and the trainer's log as `history.jsonl`,
so the report and every later stage treat it like any other training stage.

Each library needs an environment of its own. Neither was run when the stages were written: the
configs follow the libraries' documentation, and `config:` is there to correct what a library
rejects.

## Data quality

`llm.check_data` runs after `llm.prepare_data`. No check needs a model.

| Dropped by default | Flagged by default | Reported |
|---|---|---|
| empty messages; chat-template markers or control tokens inside a message; answers that repeat the prompt; answers stuck in a loop; questions that repeat a held-out one; with a `scorer`, references the scorer cannot check; preference pairs whose two answers are the same | near-duplicates; the same question with different answers; refusal boilerplate; placeholders; answers that look cut off; broken encoding, control characters, HTML; e-mail addresses, phone and card numbers, credentials; rows over `max_seq_length` | token-length percentiles; distinct word 2-grams; the most common answer opening and its share; rows per source; for preference pairs, how often the chosen answer is the longer one |

Each check's action is `drop`, `flag` or `off` under `checks:`; `pii: redact` replaces what it
finds with placeholders such as `[EMAIL]`. The stage writes the cleaned files (`chat`, `text`,
`preference`), `flags.jsonl` with the reason for every row it touched, and `report.json` with
rates and examples. `judge:` adds a model that grades each row from 1 to 5; a low grade flags.

Why so little is dropped: wrong answers and overlap with the evaluation data are the defects with
a measured cost, and removing rows on a guess can remove good ones. The evidence is in
[docs/research/llm-data.md](research/llm-data.md).

```yaml
gate:
  - {metric: quality/chat_drop_rate, max: 0.3}
  - {metric: quality/chat_conflict, max: 0.05}
```

## How stages chain

`init_from` is a Hub id or `${stage.model}`. When it is a LoRA adapter, `on_adapter` decides:

- `continue` (default for `cpt`, `sft` and `grpo`): the adapter is loaded on its base model and trained
  further. This is what Unsloth documents for resuming from a saved adapter. The adapter keeps
  the shape the earlier stage gave it, so this stage's `lora` block is not used. If the adapter
  carries trained embeddings (it came from `llm.cpt`), they keep training, at a tenth of the
  learning rate unless `embedding_learning_rate` says otherwise.
- `merge` (default for `dpo`): the adapter is folded into a 16-bit copy of the base weights
  (`<stage>/merged_base`, several GB) and a fresh LoRA is attached. Use it to change rank or
  target modules, to freeze what the earlier stage learned (the example does this between `cpt`
  and `sft_broad`), and for DPO, where the policy the loss is measured against must be exactly the
  model the stage started from.

`full_finetuning: true` (with `load_in: 16bit`) trains every weight instead of a LoRA; an adapter
it starts from is always merged first.

A training stage that is interrupted continues from its last checkpoint when the pipeline is run
again (`resume: true`); a finished stage is skipped; a stage whose parameters changed reruns
together with everything after it.

## Data formats

| `format` | Columns | Notes |
|---|---|---|
| `messages` | `messages`: `[{role, content}]` | roles `system`, `user`, `assistant`, `tool` |
| `sharegpt` | `conversations`: `[{from, value}]` | `human`/`gpt`/`system`/`observation` are mapped |
| `alpaca` | `instruction`, `input`, `output` | `input` is appended to the instruction |
| `prompt_completion` | `prompt`, `completion` | |
| text sources | `text` | `chunk_chars` cuts long documents at paragraph breaks |
| preference sources | `prompt`, `chosen`, `rejected` | strings or message lists; with no `prompt`, `chosen` and `rejected` are whole conversations |

Map other column names with `columns` on the source. A conversation is kept only if it starts with
a user turn (after an optional system message), alternates, and ends with the assistant; the
reasons for every dropped row are counted in `stats.json`.

## Running

Train in an environment of its own: `unsloth 2026.11.4` pins `transformers<=5.17`, `trl<=1.15`,
`datasets<5` and `torch<2.15`, which is older than what the other tracks are developed on.

```bash
pip install -e ".[llm]"                                   # on the GPU machine
stagetrain kinds llm --params                              # every stage and parameter
stagetrain plan   configs/llm/accuracy.yaml                # validate, show what would run
stagetrain run    configs/llm/accuracy.yaml                # run; rerun to resume
stagetrain report configs/llm/accuracy.yaml --open         # stages, gates, metrics, training curves
stagetrain run    configs/llm/accuracy.yaml --from grpo --set grpo.loss_type=dr_grpo
```

`report.html` is rewritten after every stage, and a training stage appends to its
`history.jsonl` as it logs, so `stagetrain report` can be run while a stage is still training.
For live dashboards set `train: {report_to: [trackio]}` (or `wandb`, `tensorboard`).


## Code layout

- `data.py`: row normalisation, chunking, the content-hash split. No model or tokenizer.
- `scoring.py`: scorers, reasoning-block handling, standard errors, paired differences.
- `rewards.py`: reward terms as the functions TRL's `GRPOTrainer` calls.
- `backend.py`: every Unsloth call, in `UnslothBackend`, and the plain functions that turn stage
  parameters into library keyword arguments. `hf.py`: the same interface on transformers, PEFT
  and TRL. Stages ask `get_backend(name)`; tests can swap in a fake with `set_backend`.
- `quality.py`: the data checks. `check.py`: the stage that runs them.
- `external.py`: the Axolotl and ms-swift stages.
- `prepare.py`, `train.py`, `evaluate.py`, `select.py`, `generate.py`, `export.py`: the stages.
- `stagetrain/core/launch.py` and `stagetrain/worker.py`: launching a stage on several GPUs.

## What was verified, and how

No real model was trained: the track was written on a machine without a CUDA device, and no
weights were downloaded. What ran, ran on a tiny random model built in the tests.

Run for real, with the `hf` backend on a CPU (`tests/test_llm_hf.py`):

- SFT with full fine-tuning and with LoRA, the loss masked to assistant turns; the model learns
  the synthetic task and the paired evaluation shows it;
- checkpoint selection by generated answers, self-play data generation, GRPO with TRL's
  `GRPOTrainer` and the stage's reward functions, export of the adapter and the merged model;
- a training stage in a worker process (`launch: {subprocess: true}`), start to result.

Run with stand-ins (`pytest tests/test_llm_*.py`):

- data normalisation, rejection reasons, chunking, the splits and replay mixing;
- every scorer, the standard error and the paired difference, the reward terms called the way
  TRL calls them;
- parameters -> keyword arguments, and that `sft_config_kwargs`, `dpo_config_kwargs` and
  `grpo_config_kwargs` build real `trl.SFTConfig`, `trl.DPOConfig` and `trl.GRPOConfig` objects
  on the installed trl 1.18 / transformers 5.23;
- the data checks and `llm.check_data` on rows with known defects;
- `llm.axolotl` and `llm.swift` against a script that stands in for their commands: the config
  written, the command and environment, the data conversion, what is read back;
- the stages through `Pipeline` with a fake backend: the whole accuracy pipeline (baselines,
  sweep, selection, checkpoint choice, self-play, DPO, GRPO, paired final evaluation and its
  gate), `continue` vs `merge`, full fine-tuning, resume/skip, export order, and that every
  shipped config validates.

Verified by reading the source of `unsloth 2026.11.4` and `unsloth_zoo 2026.11.4` only, never by
running it:

- `FastLanguageModel.from_pretrained` parameters, and that an adapter directory loads its base
  model with the adapter trainable (`PeftModel.from_pretrained(..., is_trainable=True)`);
- `FastLanguageModel.get_peft_model` parameters; a string `target_modules` is passed through;
  `embed_tokens` / `lm_head` belong in `modules_to_save`;
- `UnslothTrainingArguments(embedding_learning_rate=...)` and `UnslothTrainer.create_optimizer`;
- `train_on_responses_only(trainer, instruction_part=None, response_part=None, ...)` detects the
  markers from the chat template when both are `None`;
- `save_pretrained_merged`, `save_pretrained_gguf` (GGUF files land in `<dir>_gguf`; the call
  returns `gguf_directory` and `gguf_files`), `push_to_hub_merged`, `push_to_hub_gguf`;
- `PatchDPOTrainer()` is a no-op in this release; it is still called for older ones.

Not verified at all, to check on the first real run:

- `fit_config_kwargs` renames arguments for the older transformers / trl that Unsloth pins
  (`warmup_ratio`, `group_by_length`, `max_seq_length`); it was tested against stand-in
  dataclasses, not against those releases.
- TRL's conversational preference format (`prompt` / `chosen` / `rejected` as message lists) was
  read in trl 1.18; Unsloth pins trl <= 1.15.
- `llm.eval` measures loss through an `UnslothTrainer` that is never trained (TRL requires a train
  set, so it is handed the eval rows); on a 4-bit base model this path is untested.
- Exporting several formats from one loaded model (`lora`, `merged_16bit`, `gguf`, then
  `merged_4bit` last) assumes the 16-bit merge and the GGUF conversion leave the loaded weights
  untouched.
- `on_adapter: continue` for `llm.dpo` relies on TRL keeping a frozen copy of the starting adapter
  as the reference; `merge` (the default) does not depend on it.
- `GRPOConfig` field names are the same in trl 1.15 and 1.18 according to a diff of the two
  releases; the stage was checked against 1.18 only.
- `llm.grpo`: the reward functions were called in tests the way TRL's documentation and source
  describe (`prompts`, `completions`, extra dataset columns as keyword arguments). No GRPO step
  was run. `fast_inference` (vLLM) was not exercised at all.
- `backend.sample` generates several answers with `num_return_sequences` on one prompt at a
  time. It is correct by construction but slow; batching it is the first optimisation to make.
- Loading a `checkpoint-<step>` directory as an adapter (`llm.select_checkpoint`) relies on a
  Trainer checkpoint of a PEFT model holding `adapter_config.json` and the adapter weights.
- Training with more than one process: the launcher's commands are tested as text, and one
  worker process was run; no DDP or FSDP run was made. The secondary processes' behaviour
  (writing nothing, waiting for the merged model) is untested.
- `llm.axolotl` and `llm.swift` against the real libraries.
- `ftfy` repair in `llm.check_data` (not installed here), and the stage on a large dataset: near-
  duplicate detection is tested on small sets.
