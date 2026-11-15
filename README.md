# stagetrain

Multi-stage training pipelines for four kinds of models. Each kind is an independent track with its
own stages, its own dependencies and its own example configs; they share only a small core that
runs stages in order, resumes finished ones and stops at a failed quality gate.

| Track | Model | Trained with | Stages | Design |
|---|---|---|---|---|
| `llm` | Generative LLM (LoRA / QLoRA) | [Unsloth](https://github.com/unslothai/unsloth), TRL | prepare_data → cpt → sft (one or more) → dpo → eval → export | [docs/llm.md](docs/llm.md) |
| `decision` | Decision model in the style of [LiquidAI/d1](https://huggingface.co/LiquidAI/d1-3B): typed answers in one forward pass, no generated tokens | transformers, PEFT | build_data → train (align) → train (full, per seed and mixture) → soup → calibrate → eval → export | [docs/decision.md](docs/decision.md) |
| `classifiers` | Encoder text classifier, single- or multi-label | transformers | prepare_data → mlm_adapt → finetune → pseudo_label → finetune → calibrate → eval → export | [docs/classifiers.md](docs/classifiers.md) |
| `embeddings` | Text embedding model (bi-encoder) | sentence-transformers | prepare_data → pretrain → mine_negatives → finetune → eval → export | [docs/embeddings.md](docs/embeddings.md) |

## Why stages

Each of these models is trained in steps that depend on each other's outputs: an embedding model
mines its own hard negatives after a first contrastive pass; a classifier labels unlabeled text
before a second fine-tune; an LLM is adapted to raw domain text before it is taught to answer; a
decision model is aligned on short inputs, trained several times on long ones, and averaged. A
pipeline here is that sequence written down: a YAML file of stages, where a stage names the
earlier artifacts it starts from.

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

## Use

```bash
pip install -e ".[embeddings]"      # or decision, classifiers, llm: one extra per track

stagetrain kinds embeddings --params                 # the track's stages and their parameters
stagetrain plan   configs/embeddings/retrieval.yaml  # validate everything, show what would run
stagetrain run    configs/embeddings/retrieval.yaml
stagetrain status configs/embeddings/retrieval.yaml  # state and metrics of every stage

stagetrain run configs/embeddings/retrieval.yaml --from mine            # this stage and everything after it
stagetrain run configs/embeddings/retrieval.yaml --only eval --force    # one stage again
stagetrain run configs/embeddings/retrieval.yaml --set finetune.train.learning_rate=1e-5 --set seed=7
```

Install the `llm` extra in an environment of its own: Unsloth pins older releases of transformers,
trl, datasets and torch than the other tracks are developed on (see [docs/llm.md](docs/llm.md)).

A run writes to `runs/<track>/<name>/<stage>/`. Every stage directory holds its outputs, a
`stage.log`, and a `stage.json` manifest: parameters, artifacts, metrics, gate results, timing,
package versions and the git commit.

Example configs:

| Track | Config | What it trains |
|---|---|---|
| llm | `configs/llm/qwen3-multistage.yaml` | every stage: CPT, two SFT passes, DPO, eval, export |
| llm | `configs/llm/sft-only.yaml` | the shortest useful pipeline |
| decision | `configs/decision/d1-style.yaml` | a d1-style model on eight public datasets, three soup members |
| classifiers | `configs/classifiers/modernbert-agnews.yaml` | single-label, with domain adaptation and self-training |
| classifiers | `configs/classifiers/xlmr-go-emotions.yaml` | multi-label, with per-class thresholds |
| embeddings | `configs/embeddings/retrieval.yaml` | the full recipe from a plain encoder |
| embeddings | `configs/embeddings/distill.yaml` | adapting a published model with a cross-encoder teacher |

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
  core/            config (YAML, references, gates), stage registry, pipeline runner, manifests,
                   data sources, shared training arguments, checkpoint averaging
  tracks/
    llm/           Unsloth backend, data normalisation, cpt / sft / dpo, eval, export
    decision/      question schema, prompt and verbalizer, answer-slot loss, soup, calibration, probes
    classifiers/   label maps, MLM adaptation, fine-tuning, pseudo-labelling, calibration, export
    embeddings/    pair cleaning, contrastive training, hard-negative mining, retrieval eval, export
  cli.py
configs/<track>/   example pipelines
docs/<track>.md    the design of each track
tests/
```

## What has been verified

The test suite (158 tests) runs on CPU in under a minute and downloads nothing:

```bash
pip install -e ".[decision,classifiers,embeddings,dev]"
pytest
```

- **Core**: configuration, references, ordering, resume, staleness, gates, the CLI.
- **decision, classifiers, embeddings**: the real stages run end to end on tiny random models
  built inside the tests, through the same `Pipeline` a real run uses.
- **llm**: Unsloth needs an NVIDIA GPU and could not be imported where this was written. Its calls
  sit behind one backend module, written against the Unsloth source, and the tests drive the
  stages through a stand-in backend. See [docs/llm.md](docs/llm.md) for what that leaves unchecked.

Not verified: no real checkpoint has been trained with these pipelines. The example configs are
validated (`stagetrain plan`) and their dataset and model ids were checked against the Hub, but
their hyperparameters and gate thresholds are starting points, not results.

One pipeline is one process. Multi-GPU training through `accelerate launch` or `torchrun` is not
wired up: every rank would run the whole pipeline.
