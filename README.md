# stagetrain

Multi-stage training pipelines for four kinds of models. Each kind is an independent track with its
own stages, its own dependencies and its own example configs; they share only a small core that
runs stages in order, resumes finished ones and stops at a failed quality gate.

| Track | Model | Trained with | Stages | Design |
|---|---|---|---|---|
| `llm` | Generative LLM (LoRA / QLoRA / full) | [Unsloth](https://github.com/unslothai/unsloth), or plain TRL + PEFT; Axolotl and ms-swift through their CLIs; one or several GPUs | prepare_data → check_data → eval (baselines) → sft × N → select → soup (seeds) → select_checkpoint → distill → generate_data → sft (own verified answers) / dpo / grpo → eval → export | [docs/llm.md](docs/llm.md); research: [training](docs/research/llm.md), [RL and post-training](docs/research/llm-rl.md), [optimizer settings](docs/research/llm-optimizer.md), [other levers](docs/research/llm-levers.md), [data quality](docs/research/llm-data.md), [libraries](docs/research/llm-libraries.md) |
| `decision` | Decision model in the style of [LiquidAI/d1](https://huggingface.co/LiquidAI/d1-3B): typed answers in one forward pass, no generated tokens | transformers, PEFT | build_data (with held-out sources) → train × seeds → soup → calibrate → eval → export | [docs/decision.md](docs/decision.md), [research](docs/research/decision.md) |
| `classifiers` | Encoder text classifier, single- or multi-label | transformers | prepare_data → baseline (TF-IDF) → mlm_adapt → finetune × (model, learning rate, seed) → select → calibrate → eval → export; label audit and ensemble distillation as a second recipe | [docs/classifiers.md](docs/classifiers.md), [research](docs/research/classifiers.md) |
| `embeddings` | Text embedding model (bi-encoder) | sentence-transformers | prepare_data → bm25 and untuned baselines → mine_negatives (strong embedder) → finetune × N → select → merge → eval → export | [docs/embeddings.md](docs/embeddings.md), [research](docs/research/embeddings.md) |

## Why stages

Each of these models is trained in steps that depend on each other's outputs: an embedding model
is fine-tuned on hard negatives that another model mined, then merged with the model it started
from; a classifier is chosen among several learning rates and seeds, and its ensemble is distilled
into one model; an LLM is fine-tuned, its best checkpoint picked by the task metric, then improved
with reinforcement learning on what it can already partly do; a decision model is trained on
several seeds, averaged, and calibrated on sources it has never seen. A pipeline here is that
sequence written down: a YAML file of stages, where a stage names the earlier artifacts it starts
from.

Which stages each track has, and their defaults, follow a reading of the evidence that is kept
next to the code: `docs/research/<track>.md` says what raises accuracy, what does not, and where
no evidence was found.

```yaml
track: embeddings
name: my-retriever
vars: {base_model: answerdotai/ModernBERT-base}

stages:
  - name: pretrain
    kind: embeddings.pretrain
    params: {init_from: "${vars.base_model}", train_data: "${data.train}"}

  - name: mine
    kind: embeddings.mine_negatives
    params: {model: "${pretrain.model}", train_data: "${data.train}", corpus: "${data.corpus}"}

  - name: finetune
    kind: embeddings.finetune
    params: {init_from: "${pretrain.model}", train_data: "${mine.train}"}
    gate: {metric: eval/ndcg@10, baseline: pretrain.eval/ndcg@10, max_drop: 0.0}
```

- `${vars.x}` is filled in when the file is loaded.
- `${stage.artifact}` is the path of an earlier stage's output. It is resolved when the stage
  starts, and it makes that stage a dependency.
- `defaults:` holds parameters shared by every stage of one kind; YAML anchors work too.

### Several runs of one stage

The setting that matters most is rarely known in advance. A `matrix` runs a stage once per
combination of values, and `<track>.select` keeps the best run and hands on its artifacts:

```yaml
  - name: sft
    kind: llm.sft
    matrix:
      train.learning_rate: [5.0e-5, 1.0e-4, 2.0e-4, 5.0e-4, 1.0e-3]
      seed: [1, 2]                      # ten runs: sft-1 ... sft-10
    params: {init_from: "${vars.base_model}", train_data: "${data.chat_train}", eval_data: "${data.chat_eval}"}

  - name: best
    kind: llm.select
    params: {group: sft, metric: eval/loss}

  - name: export
    kind: llm.export
    params: {init_from: "${best.model}"}
```

Changing the matrix later reruns only the combinations that are new.

## Use

```bash
pip install -e ".[embeddings]"      # or decision, classifiers, llm: one extra per track

stagetrain kinds embeddings --params                 # the track's stages and their parameters
stagetrain plan   configs/embeddings/retrieval.yaml  # validate everything, show what would run
stagetrain run    configs/embeddings/retrieval.yaml
stagetrain status configs/embeddings/retrieval.yaml  # state and metrics of every stage
stagetrain report configs/embeddings/retrieval.yaml --open   # the run as one HTML page

stagetrain run configs/embeddings/retrieval.yaml --from mine            # this stage and everything after it
stagetrain run configs/embeddings/retrieval.yaml --only eval --force    # one stage again
stagetrain run configs/embeddings/retrieval.yaml --set finetune.train.learning_rate=1e-5 --set seed=7
```

Install the `llm` extra in an environment of its own: Unsloth pins older releases of transformers,
trl, datasets and torch than the other tracks are developed on (see [docs/llm.md](docs/llm.md)).

A run writes to `runs/<track>/<name>/<stage>/`. Every stage directory holds its outputs, a
`stage.log`, a `stage.json` manifest (parameters, artifacts, metrics, gate results, timing,
package versions, the git commit) and, for training stages, `history.jsonl`: every logged step.

### The report

`runs/<track>/<name>/report.html` is one self-contained page, rewritten after every stage:

- every stage's state, duration, gate results and metrics;
- each eval metric across the stages that report it, in pipeline order: the base model, each
  run of a sweep, the selected model, the model after post-training;
- the training curves of every training stage: train and eval loss on one chart, each other
  metric (reward, accuracy, learning rate) on its own.

Training stages append to `history.jsonl` as they log, so `stagetrain report` also works while a
stage is still running. The page needs no network; it follows the system's light or dark mode,
and every chart has a data table. For live dashboards, `train: {report_to: [wandb]}` (or
`tensorboard`, `trackio`) is passed through to the trainer.

Example configs:

| Track | Config | What it trains |
|---|---|---|
| llm | `configs/llm/multi-gpu.yaml` | LoRA SFT on four GPUs without Unsloth; full fine-tuning with FSDP2 |
| llm | `configs/llm/other-trainers.yaml` | the same data through Axolotl and ms-swift |
| llm | `configs/llm/accuracy.yaml` | the research-driven recipe: baselines, a learning-rate sweep, checkpoint selection, self-play, GRPO, paired evaluations |
| llm | `configs/llm/sweep-soup.yaml` | a learning-rate sweep over three seeds, the best configuration's seeds averaged into one adapter |
| llm | `configs/llm/post-training.yaml` | after SFT: on-policy distillation, two self-training rounds, RL in two stages on reserved questions |
| llm | `configs/llm/qwen3-multistage.yaml` | CPT, two SFT passes, DPO, eval, export |
| llm | `configs/llm/sft-only.yaml` | the shortest useful pipeline |
| decision | `configs/decision/d1-style.yaml` | a d1-style model on 14 public datasets, three seeds, calibrated and judged on three held-out datasets |
| classifiers | `configs/classifiers/modernbert-agnews.yaml` | the research-driven recipe: a TF-IDF baseline, plain and domain-adapted encoders over three learning rates and three seeds, selected on the seed mean, a paired gate against the baseline |
| classifiers | `configs/classifiers/audit-distill.yaml` | a cross-validated label audit, then a seed ensemble distilled into one model |
| classifiers | `configs/classifiers/xlmr-go-emotions.yaml` | multi-label, with per-label thresholds |
| embeddings | `configs/embeddings/retrieval.yaml` | the research-driven recipe: a published embedder adapted with mined negatives, a small sweep, a merge with the starting model, paired evaluation |
| embeddings | `configs/embeddings/distill.yaml` | the same with a cross-encoder teacher |
| embeddings | `configs/embeddings/pretrain.yaml` | weakly-supervised pre-training of a plain encoder, for when it is worth it |

## How a run behaves

**Validation first.** Before any stage runs, every stage's kind and parameters are checked. A typo
in a parameter name is an error at second zero, not after the first stage has trained for six
hours.

**Resume.** A stage is skipped when it already finished with the same parameters, the same seed and
the same upstream outputs. Change a stage's parameters and that stage and everything after it run
again; everything before it is reused. `--force` reruns a stage, and the stages after it then count
as stale.

**Interrupted stages.** If a stage was interrupted, the next `run` continues it from its last
checkpoint. Leftover files are only ever continued when they belong to an unfinished attempt of the
same configuration; in every other case the stage directory is emptied first. Rerunning a stage
therefore replaces its previous output: give the pipeline a new `name` to keep both.

**Gates.** A stage can carry conditions on its metrics. If one fails, the pipeline stops there
(exit code 2), and the stage is not retrained on the next run: only the gate is checked again, so
loosening it lets the pipeline continue.

```yaml
gate:
  - {metric: eval/accuracy, min: 0.90}                                       # absolute
  - {metric: eval/ndcg@10, baseline: pretrain.eval/ndcg@10, max_drop: 0.0}   # against an earlier stage
  - {metric: eval/ece, max: 0.05}
```

**Several GPUs.** A training stage of the llm track can be given `launch: {num_processes: 4}`. The
pipeline stays one process: it starts that stage under `torchrun` (or `accelerate launch`) with one
process per GPU and reads its result back. See [docs/llm.md](docs/llm.md#several-gpus).

## Adding a stage

```python
from stagetrain.core import Stage, StageContext, StageParams, StageResult, register

class MyParams(StageParams):          # unknown keys in the YAML are rejected
    model: str
    threshold: float = 0.5

@register("classifiers.my_stage")
class MyStage(Stage):
    """One line that `stagetrain kinds` shows."""

    Params = MyParams

    def run(self, ctx: StageContext) -> StageResult:
        import torch                  # heavy imports go inside run()
        out = ctx.path("result.json")
        ...
        return StageResult(artifacts={"result": out}, metrics={"eval/score": 0.9})
```

Import the module from the track's `__init__.py` and the kind is available in configs.

## Layout

```
src/stagetrain/
  core/            config (YAML, references, gates, matrix), stage registry, pipeline runner,
                   manifests, run selection, training history, the HTML report, data sources,
                   shared training arguments, checkpoint averaging
  tracks/
    llm/           Unsloth and hf backends, data normalisation and quality checks, scorers and rewards,
                   cpt / sft / dpo / grpo, checkpoint selection, data generation, eval, export,
                   Axolotl and ms-swift stages
    decision/      question schema, prompt and verbalizer, answer-slot loss, soup, calibration, probes
    classifiers/   label maps, MLM adaptation, fine-tuning, pseudo-labelling, calibration, export
    embeddings/    pair cleaning, contrastive training, hard-negative mining, retrieval eval, export
  cli.py
  worker.py        one process of a stage launched on several GPUs
configs/<track>/   example pipelines
docs/<track>.md    the design of each track
docs/research/     the evidence behind a track's recipe
tests/
```

## What has been verified

The test suite (291 tests) runs on CPU in under a minute and downloads nothing:

```bash
pip install -e ".[decision,classifiers,embeddings,dev]"
pytest
```

- **Core**: configuration, references, ordering, resume, staleness, gates, matrix runs and
  selection, training history, the report's data, the CLI. The report page itself was looked at
  in a browser, in light and dark mode, with simulated curves.
- **decision, classifiers, embeddings**: the real stages run end to end on tiny random models
  built inside the tests, through the same `Pipeline` a real run uses.
- **llm**: the stages run end to end on a tiny random model with the `hf` backend (transformers,
  PEFT and TRL): SFT, checkpoint selection, self-play, GRPO, on-policy distillation, LoRA+, the
  average of several seeds' adapters, export, and a stage in a worker process. The Unsloth backend was not run: its calls were written against the Unsloth source and
  are driven in the tests through a stand-in. `llm.axolotl` and `llm.swift` are tested against a
  script standing in for those commands. The data checks are tested on rows with known defects.
  See [docs/llm.md](docs/llm.md) for what that leaves unchecked.

Not verified: no real checkpoint has been trained with these pipelines. The example configs are
validated (`stagetrain plan`) and their dataset and model ids were checked against the Hub, but
their hyperparameters and gate thresholds are starting points, not results.

Training on several GPUs is wired up for the llm track and was not run on more than one process:
this was written on a machine without a CUDA device. The other three tracks train in the
pipeline's own process, on one GPU.
