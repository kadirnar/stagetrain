# Classifiers track

Multi-stage training of encoder text classifiers (ModernBERT, BERT, XLM-R, DeBERTa: anything
`AutoModelForSequenceClassification` loads), single-label and multi-label. Every stage is a
registered kind under `classifiers.*`; a pipeline is a YAML file that chains them.

```
prepare_data ─► mlm_adapt ─► finetune (teacher) ─► pseudo_label ─► finetune (student) ─► calibrate ─► eval ─► export
     │              ▲              ▲                    ▲                  ▲                 ▲          ▲
     └── train / eval / test / unlabeled JSONL + labels.json ──────────────┴─────────────────┴──────────┘
```

Only `prepare_data` and one `finetune` are required. The rest are optional improvements, each
of which can be gated so that it must pay for itself.

## Stages

| Kind | What it does | Why | Reads | Artifacts | Metrics worth a gate |
|---|---|---|---|---|---|
| `classifiers.prepare_data` | Loads the sources, builds the label map, normalises whitespace, removes duplicates and texts labelled two ways, removes train rows that also occur in eval/test, carves stratified eval/test splits when none are given, computes balanced class weights | Leakage and label conflicts are the cheapest errors to remove and the hardest to see later | `DataSource`s with `text` + `label` (or `labels`), optional `text_pair`, optional unlabeled `text` | `train`, `eval`, `test`, `unlabeled` (JSONL; only the ones that exist), `labels` (`labels.json`) | `n_labels`, `min_class_count`, `imbalance_ratio`, `removed/*` |
| `classifiers.mlm_adapt` | Continues masked-LM pretraining of the encoder on in-domain text (DAPT/TAPT); long texts become overlapping windows | A general encoder has not seen the domain's vocabulary; this needs no labels | `init_from`, JSONL files with `text` | `model` | `mlm/loss_drop` (before minus after, on held-out text), `eval/perplexity` |
| `classifiers.finetune` | Supervised training. Loss over target distributions: class-weighted CE or focal (BCE or focal BCE for multi-label), label smoothing, soft targets. Layer-wise lr decay, a separate head lr, optional head-only warm-up (LP-FT) | The main stage; used once for the teacher and again for the student | `init_from` (Hub id or an earlier `model`), `labels`, `train_data`, `eval_data`, `extra_train_data` | `model` (weights, tokenizer, `labels.json`) | `eval/accuracy`, `eval/macro_f1`, `eval/micro_f1`, `eval/loss` |
| `classifiers.pseudo_label` | Runs a trained teacher over the unlabeled pool and keeps confident predictions (`min_confidence`, `max_per_class`; for multi-label every label must be far from 0.5), written with the teacher's distribution as soft labels | Self-training / distillation: more training signal without annotation | teacher `model`, optional `calibration`, unlabeled JSONL | `data` (JSONL with `probs`, `confidence`, `source: pseudo`) | `pseudo/kept`, `pseudo/keep_rate`, `pseudo/classes_covered` |
| `classifiers.calibrate` | Fits one temperature on held-out logits by minimising NLL; for multi-label also a decision threshold per label (best F1). Weights do not change | Fine-tuned encoders are over-confident; thresholds at 0.5 are rarely right for rare labels | `model`, labelled held-out JSONL | `calibration` (`calibration.json`) | `calibration/ece_after`, `calibration/nll_drop`, `calibration/at_bound` |
| `classifiers.eval` | Accuracy, macro/micro/weighted F1, NLL, ECE, per-class precision/recall/F1, confusion matrix, optional predictions | The number the pipeline is judged by, on a split nothing was tuned on | `model`, optional `calibration`, labelled JSONL | `report`, `confusion_matrix` (single-label), `predictions` | `eval/macro_f1`, `eval/min_class_f1`, `eval/ece` |
| `classifiers.export` | Copies the model into one directory with `labels.json`, `calibration.json` and `classifier_config.json`; optional ONNX | One self-contained artifact to deploy | `model`, optional `calibration` | `model`, `onnx` (when written) | `export/onnx`, `export/size_mb` |

## How the stages chain

Stages pass files, never Python objects. A stage refers to an earlier stage's artifact as
`${stage.artifact}`, which also makes it a dependency:

```yaml
- name: student
  kind: classifiers.finetune
  params:
    init_from: ${adapt.model}          # the domain-adapted encoder, not the teacher
    labels: ${data.labels}
    train_data: ${data.train}
    eval_data: ${data.eval}
    extra_train_data:
      - {path: "${pseudo.data}", weight: 0.5}
  gate: {metric: eval/macro_f1, baseline: teacher.eval/macro_f1, max_drop: 0.005}
```

Data contracts:

- **Labelled row**: `{"text": ..., "label": 2}` (single-label) or `{"text": ..., "labels": [0, 3]}`
  (multi-label); class indices follow `labels.json`. Optional `text_pair`, `weight`.
- **Soft row** (from `pseudo_label`): the same plus `"probs": [...]`. `finetune` trains towards
  `probs` as they are; label smoothing applies to gold rows only. `soft_label_weight` and the
  per-file `weight` scale their loss.
- **Model directory**: a normal `save_pretrained` directory plus `labels.json`, so every later
  stage finds the label names and problem type next to the weights.

Which split is used where: `train` trains; `eval` selects the best checkpoint and fits the
calibration; `test` is only read by `classifiers.eval`. If you calibrate and evaluate on the
same split, the reported ECE and the multi-label F1 are optimistic.

## Choices that are easy to get wrong

- **`init_from` of the student** is the adapted encoder, not the teacher: a student that starts
  from the teacher mostly re-learns the teacher's mistakes.
- **`llrd`** multiplies the learning rate by this factor per layer going down from the top
  (`0.9` on a 22-layer ModernBERT leaves the embeddings at ~9% of `learning_rate`). When `llrd`
  or `head_lr_multiplier` is not 1, the stage builds its own AdamW and `train.optim` is ignored.
- **`head_warmup_epochs`** trains only the head first. Use it when the head is new (a Hub
  encoder, or the output of `mlm_adapt`); it is pointless when `init_from` is already a
  classifier with the same labels.
- **`class_weights: balanced`** uses the weights `prepare_data` computed on the final train
  split: `n / (classes * count)` for single-label, `negatives / positives` (BCE `pos_weight`)
  for multi-label, both clipped by `max_class_weight`.
- **`metric_for_best_model`** takes the Trainer's name (`eval_macro_f1`); gates take the stage
  metric name (`eval/macro_f1`).
- **`calibration/at_bound`** is 1 when the temperature fit ran into `min_temperature` or
  `max_temperature`. That happens when the held-out split is classified entirely right (or is
  tiny); the temperature is then meaningless and should not be shipped. Gate on it.
- **Resume**: `mlm_adapt` and `finetune` continue from their last checkpoint only when the
  pipeline resumes an unfinished attempt of the same run (`ctx.resumed`). A changed or forced
  stage starts from an empty directory.

## Running

```bash
pip install -e ".[classifiers]"
stagetrain kinds classifiers --params                         # every kind and its parameters
stagetrain plan configs/classifiers/modernbert-agnews.yaml    # validate, show what would run
stagetrain run  configs/classifiers/modernbert-agnews.yaml    # run; finished stages are skipped
stagetrain run  configs/classifiers/modernbert-agnews.yaml --from student
stagetrain run  configs/classifiers/modernbert-agnews.yaml --set vars.base_model=answerdotai/ModernBERT-large
```

Shipped configs:

- `configs/classifiers/modernbert-agnews.yaml`: single-label, all seven kinds (nine stages,
  teacher and student).
- `configs/classifiers/xlmr-go-emotions.yaml`: multi-label, the short path
  (`prepare_data → finetune → calibrate → eval → export`).

Using the exported model:

```python
from stagetrain.tracks.classifiers.inference import Classifier

clf = Classifier("runs/classifiers/modernbert-agnews/export/model")
clf.predict(["Stocks rallied after the central bank held rates."])
# [{"label": "Business", "confidence": 0.97, "probabilities": {...}}]
```

## What the tests cover

`pytest tests/test_classifiers_units.py tests/test_classifiers_pipeline.py` (about 12 s on CPU,
no network, no downloaded weights).

Covered:

- Pure logic: label map, raw-label parsing, dedupe and conflicts, leakage removal, stratified
  split (deterministic, every class kept), class weights, target construction.
- Metrics against scikit-learn (accuracy, macro/micro/weighted F1, per-class report, confusion
  matrix, hamming loss), ECE on hand-computed cases, temperature scaling recovering a known
  temperature, per-label thresholds.
- The loss against `F.cross_entropy` / `F.binary_cross_entropy_with_logits` (class weights,
  label smoothing, soft targets, sample weights); layer-wise lr groups for BERT- and
  ModernBERT-style parameter names.
- End to end through `Pipeline`, with a 2-layer random BERT and a word-level tokenizer built in
  the test: all eight stages single-label (gates included, second run fully cached) and the
  multi-label path; the exported directory loaded by `Classifier`.
- `finetune` on a tiny random ModernBERT (backbone prefix `model.`, `head` + `classifier`),
  checkpoint resume, `ClassLabel` / list-of-`ClassLabel` Hub features, long-text chunking for
  MLM, sentence pairs, row order of batched inference.
- Both shipped configs pass `Pipeline.validate()`.

Not covered:

- No real checkpoint or Hub dataset was downloaded or trained. The shipped configs are validated
  for structure only; their gate thresholds are starting points, not measured results.
- GPU paths: bf16/fp16, multi-GPU, `group_by_length` at scale.
- ONNX export is not in the test suite's environment (`onnx` / `onnxscript` are not installed, so
  the suite checks the "skipped" note). It was run once by hand on the tiny BERT with both
  packages on the path: dynamic batch and sequence axes, logits equal to PyTorch to 1e-6 under
  onnxruntime. Other architectures are untested.
- Tokenizers other than the test's word-level one (e.g. `token_type_ids` handling of real BERT
  tokenizers, ModernBERT's tokenizer).
