# Classifiers track

Multi-stage training of encoder text classifiers (ModernBERT, DeBERTa-v3, XLM-R, mmBERT, BERT:
anything `AutoModelForSequenceClassification` loads), single-label and multi-label, with the goal
of the highest accuracy. Which stages exist, their order and their defaults follow
[docs/research/classifiers.md](research/classifiers.md): measure against a baseline, sweep the
learning rate and the encoder over several seeds, set the input length from the data, then
improve the data (label audit, ensemble distillation).

```
prepare_data ─┬─> baseline                         TF-IDF + logistic regression: the bar
              ├─> mlm_adapt ─┐
              └──────────────┴─> finetune x N (matrix: encoder x learning rate x seed)
                                     └─> select (seed mean) ──> calibrate ──> eval (paired against the baseline) ──> export

finetune x K (matrix over fold) ──> audit_labels                       which training labels to check by hand
finetune x S (matrix over seed) ──> pseudo_label (ensemble) ──> finetune (student)
```

Only `prepare_data` and one `finetune` are required. Example pipelines in `configs/classifiers/`:

| Config | What it is |
|---|---|
| `modernbert-agnews.yaml` | The accuracy recipe, single-label: baseline, a sweep of encoder x learning rate x seed selected on the seed mean, calibration on its own split, a paired test against the baseline |
| `audit-distill.yaml` | A cross-validated label audit, and a three-seed ensemble distilled into one student |
| `xlmr-go-emotions.yaml` | Multi-label on a multilingual encoder: a learning-rate sweep, per-label thresholds |

## Stages

| Kind | What it does | Why it is there |
|---|---|---|
| `classifiers.prepare_data` | Builds the label map; normalises whitespace; removes duplicates and texts labelled two ways; removes rows that also occur in a more protected split; carves stratified `eval`, `calibration` and `test` splits when none are given; computes class weights | Leakage and label conflicts are the cheapest errors to remove. Four splits, because each selection needs rows the previous one did not see |
| `classifiers.baseline` | TF-IDF + logistic regression (one-vs-rest for multi-label) on the same splits, with the same report as `classifiers.eval` | Minutes on a CPU. If a fine-tuned encoder does not clearly beat it, the problem is the data, not the hyperparameters |
| `classifiers.mlm_adapt` | Continues masked-LM pretraining on in-domain text (DAPT/TAPT); long texts become overlapping windows | Helps when the domain is far from web and news text. It has to earn its place: train with and without it and let `select` decide |
| `classifiers.finetune` | Supervised training towards target distributions, so gold, smoothed and soft-labelled rows share one loss. Measures the input length; early stopping; optionally holds one fold out and writes its predictions | The main stage. Run it as a `matrix` |
| `classifiers.select` | The best setting of a matrix by a metric; with `average_over: [seed]` by its mean over seeds, handing on its most typical run | A single run rewards a lucky seed as much as a good setting |
| `classifiers.audit_labels` | Reads the out-of-fold predictions of a fold matrix and ranks the training rows by loss; writes the top slice as suspects, optionally a cleaned training file | A row a model gets confidently wrong without having trained on it is the likeliest label error |
| `classifiers.pseudo_label` | A teacher, or an ensemble whose probabilities are averaged, labels text. Keeps confident rows, or with `keep_all` every row with its soft labels | Distilling a seed ensemble into one student. A single same-architecture teacher at a confidence threshold mostly adds what the model already knows |
| `classifiers.calibrate` | One temperature fitted on held-out logits; for multi-label a decision threshold per label, shared among labels with too few positives | Probabilities that mean what they say; thresholds at 0.5 are rarely right for rare labels |
| `classifiers.eval` | Accuracy, macro / micro / weighted F1, NLL, ECE, per-class report, confusion matrix; 95 % bootstrap intervals; the paired difference to another evaluation (`compare_to`) | The reported number, on a split nothing was tuned on, with how far it could move |
| `classifiers.export` | One directory: weights, tokenizer, `labels.json`, `calibration.json`, `classifier_config.json`; optional ONNX | What gets deployed |

### What `finetune` does about the things that decide the result

- **Learning rate.** The default is 5e-5, inside the range ModernBERT's authors selected for the
  base model (5e-5 to 8e-5). It is the most sensitive setting and its best value depends on the
  encoder: sweep it. ModernBERT-large 1e-5 to 5e-5; DeBERTa-v3 and XLM-R 1e-5 to 3e-5.
- **Epochs.** `early_stopping_patience: 3` with `train.eval_steps` set to a few evaluations per
  epoch stops when the eval metric (macro-F1 unless `train.metric_for_best_model` says otherwise)
  has not improved, and keeps the best checkpoint. `num_train_epochs` is then an upper bound.
- **Input length.** `max_length: auto` (the default) measures the training rows and takes the
  length that fits `length_percentile` (99) percent of them, up to the model's limit and
  `max_length_cap`. The stage reports `data/max_length` and `data/truncated_fraction`, and
  writes `preprocessing.json` into the model directory: every later stage cuts inputs the way the
  model was trained. On an 8k-token encoder with long documents, set `max_length_cap`: cost
  grows with length.
- **Truncation.** `head` keeps the start of an over-long text. `head_tail` keeps the first
  quarter and the last three quarters of the window. Sentence pairs are always cut from the
  longer side.
- **Pooling.** `pooling: mean` or `cls` sets ModernBERT's `classifier_pooling`. Left unset, the
  model keeps its own (the published ModernBERT checkpoints use mean).
- **Imbalance.** `loss: ce` is plain cross-entropy (BCE for multi-label). `loss:
  logit_adjusted` adds `logit_adjustment_tau * log(class frequency)` to the logits during
  training only, so the raw logits used for prediction are balanced across classes.
  `class_weights: balanced` and `loss: focal` remain as options.
- **Off by default, there to search.** `llrd`, `head_lr_multiplier`, `head_warmup_epochs`,
  `label_smoothing`, `classifier_dropout`. None has controlled support on current encoders.
  `classifier_dropout` is set on the field the architecture has (`classifier_dropout`, or
  DeBERTa's `cls_dropout`); for a head without dropout the stage's notes say it was not set.

### Sweeps and seeds

```yaml
- name: finetune
  kind: classifiers.finetune
  matrix:
    init_from: ["${vars.base_model}", "${adapt.model}"]   # with and without adaptation
    train.learning_rate: [3.0e-5, 5.0e-5, 8.0e-5]
    seed: [1, 2, 3]
- name: best
  kind: classifiers.select
  params: {group: finetune, metric: eval/macro_f1, average_over: [seed]}
```

`select/mean` is the winning setting's mean, `select/std` its spread over seeds: a difference
between settings smaller than that is noise. `${best.model}` is the winning setting's most
typical run.

### Comparing two models

```yaml
- name: test
  kind: classifiers.eval
  params: {model: "${best.model}", data: "${data.test}", compare_to: "${baseline.predictions}"}
  gate: {metric: eval/delta_macro_f1_ci_low, min: 0.0}    # better beyond chance
```

`compare_to` takes the `predictions` artifact of another `classifiers.eval` or
`classifiers.baseline` over the same rows. Both models are scored on the same resampled rows, so
the interval of the difference is much tighter than the two models' own intervals.
`eval/delta_macro_f1_ci_low > 0` means better beyond chance; `eval/delta_macro_f1_ci_high >= 0`
means not worse beyond chance, which is the gate for a student against its teacher.

### The label audit

```yaml
- name: cv
  kind: classifiers.finetune
  matrix: {fold: [0, 1, 2, 3, 4]}
  params: {folds: 5}
- name: audit
  kind: classifiers.audit_labels
  params: {group: cv, labels: "${data.labels}", train_data: "${data.train}", review_fraction: 0.02}
```

A row's fold is decided by a hash of its text, so it is the same in every run. Each run trains
without its fold and writes that fold's predictions and losses (`oof`). `audit_labels` merges
them, ranks by loss, and writes `suspects.jsonl` with the given and the predicted label.
`audit/oof_macro_f1` is the recipe's quality on all of the training rows, each scored by a model
that did not train on it. `drop_fraction` writes `train_clean` without the worst rows; the
default keeps everything, because relabelling the suspects by hand is the better fix.

### The ensemble teacher

`classifiers.pseudo_label` takes one `model`, a list of `models`, or `group: <matrix name>`.
With a group, `ensemble: all` averages every run and `ensemble: winner` only the runs of the
setting a `classifiers.select` stage chose (give that stage's `model`). `keep_all: true` keeps
every row with the averaged distribution instead of thresholding confidence.

## How the stages chain

Stages pass files, never Python objects. A stage refers to an earlier stage's artifact as
`${stage.artifact}`, which also makes it a dependency.

Data contracts:

- **Labelled row**: `{"text": ..., "label": 2}` (single-label) or `{"text": ..., "labels": [0, 3]}`
  (multi-label); class indices follow `labels.json`. Optional `text_pair`, `weight`.
- **Soft row** (from `pseudo_label`, or labels from any other source such as an LLM): the same
  plus `"probs": [...]`. `finetune` trains towards `probs` as they are; label smoothing applies
  to gold rows only. `soft_label_weight` and the per-file `weight` scale their loss.
- **Model directory**: a normal `save_pretrained` directory plus `labels.json` and
  `preprocessing.json`, so every later stage finds the label names, the problem type and how
  inputs are cut next to the weights.
- **Predictions** (from `eval` and `baseline`): `{"id", "text", "gold", "pred", "predicted",
  "probs"}`; `id` identifies a row across evaluations of the same file.

Which split is used where:

| Split | Used for |
|---|---|
| `train` | training; the folds of a label audit |
| `eval` | early stopping, the best checkpoint, choosing among settings |
| `calibration` | the temperature and the thresholds |
| `test` | `classifiers.eval` and `classifiers.baseline` only |

## Choices that are easy to get wrong

- **Calibrate on `calibration`, not on `eval`.** The split that picked the checkpoint flatters
  that checkpoint.
- **A gate needs an interval.** A margin of a few thousandths of macro-F1 on a few thousand rows
  is below the standard error. Gate on `eval/delta_macro_f1_ci_low` or `_ci_high`.
- **`init_from` of a student** is the plain (or adapted) encoder, not the teacher.
- **A teacher that is distilled or thresholded** is trained with `label_smoothing: 0`.
- **`llrd`** multiplies the learning rate per layer going down from the top: `0.9` on a 22-layer
  ModernBERT leaves the embeddings at about 9 % of `learning_rate`. When `llrd` or
  `head_lr_multiplier` is not 1, the stage builds its own AdamW and `train.optim` is ignored.
- **`metric_for_best_model`** takes the Trainer's name (`eval_macro_f1`); gates take the stage
  metric name (`eval/macro_f1`).
- **`calibration/at_bound`** is 1 when the temperature fit ran into its bounds, which happens
  when the split is classified entirely right or is tiny. Gate on it.
- **Precision.** Use `train: {precision: bf16}` on a GPU. fp16 has produced NaN losses with
  ModernBERT.
- **Resume**: `mlm_adapt` and `finetune` continue from their last checkpoint only when the
  pipeline resumes an unfinished attempt of the same run.

## Running

```bash
pip install -e ".[classifiers]"
stagetrain kinds classifiers --params                         # every kind and its parameters
stagetrain plan   configs/classifiers/modernbert-agnews.yaml  # validate, show what would run
stagetrain run    configs/classifiers/modernbert-agnews.yaml  # run; finished stages are skipped
stagetrain report configs/classifiers/modernbert-agnews.yaml --open
stagetrain run    configs/classifiers/modernbert-agnews.yaml --set vars.base_model=microsoft/deberta-v3-base
stagetrain run    configs/classifiers/audit-distill.yaml --until audit
```

Using the exported model:

```python
from stagetrain.tracks.classifiers.inference import Classifier

clf = Classifier("runs/classifiers/modernbert-agnews/export/model")
clf.predict(["Stocks rallied after the central bank held rates."])
# [{"label": "Business", "confidence": 0.97, "probabilities": {...}}]     (numbers are illustrative)
```

## What the tests cover

`pytest tests/test_classifiers_*.py` (about 20 s on CPU, no network, no downloaded weights).

Covered:

- Pure logic: label map, raw-label parsing, dedupe and conflicts, leakage removal, stratified
  splits, class weights, target construction, fold assignment, suspect ranking.
- Metrics against scikit-learn; ECE on hand-computed cases; temperature scaling recovering a
  known temperature; per-label thresholds and the shared threshold of rare labels; bootstrap
  intervals; the paired difference separating two models whose own intervals overlap.
- The loss against `F.cross_entropy` / `F.binary_cross_entropy_with_logits`, including the
  logit-adjusted form; layer-wise lr groups.
- End to end through `Pipeline`, with a 2-layer random BERT and a word-level tokenizer built in
  the test:
  - the recipe: four splits, the baseline, a learning-rate x seed sweep with early stopping
    selected on the seed mean, calibration, a test evaluation paired against the baseline, export;
  - the label audit: six flipped labels among 210 rows, at least five of them among the top
    suspects; then a two-seed ensemble distilled into a student;
  - the earlier teacher / pseudo-label / student path and the multi-label path.
- `finetune` on a tiny random ModernBERT; head dropout and pooling on tiny DeBERTa-v2,
  ModernBERT and EuroBERT models; head+tail truncation; measured input length; checkpoint resume.
- Every shipped config passes `Pipeline.validate()`.

Not covered:

- No real checkpoint or Hub dataset was downloaded or trained. The shipped configs are validated
  for structure only; their gate thresholds are starting points, not measured results. The
  dataset and model ids in them were checked to exist.
- Whether any of this raises accuracy on a real task. The tiny model proves the plumbing. In
  particular the logit-adjusted loss is tested as arithmetic, not for its effect on rare classes.
- GPU paths: bf16/fp16, multi-GPU, `group_by_length` at scale.
- ONNX export is not in the test suite's environment (`onnx` / `onnxscript` are not installed, so
  the suite checks the "skipped" note). It was run once by hand on the tiny BERT with both
  packages on the path, before the changes described here: logits equal to PyTorch to 1e-6 under
  onnxruntime. Other architectures are untested.
- Real tokenizers (`token_type_ids` of BERT tokenizers, ModernBERT's and DeBERTa's tokenizers),
  including head+tail truncation on them.
