# Embeddings track

Multi-stage training of a text embedding model (bi-encoder) with sentence-transformers. The track
is independent of the other three: its pipelines hold only `embeddings.*` stages.

```
data_weak ─► pretrain ─┬─► mine_negatives ─► finetune ─► eval ─► export
data_sup ──────────────┴───────────────────────┘
```

The idea behind the staging: contrastive learning needs many pairs to shape the embedding space,
and hard negatives to sharpen it, and a model can only find good hard negatives once it already
retrieves reasonably. So the model is first trained on cheap pairs with in-batch negatives, then
used to mine its own hard negatives on the supervised data, then fine-tuned on them.

## Stages

| Kind | What it does | Why | Reads | Writes (artifacts) | Typical gate |
|---|---|---|---|---|---|
| `embeddings.prepare_data` | Loads (query, positive[, negative]) sources, normalises whitespace, drops empty / too short / too long / identical / duplicate pairs, splits off eval queries, builds a retrieval eval set | Duplicates become false in-batch negatives; a split by query keeps eval honest | `sources`, optional `eval_sources`, `corpus_sources` | `train`, `corpus`, `stats`; with an eval split also `eval_pairs`, `eval_set` | `data/train_pairs` min |
| `embeddings.pretrain` | Contrastive pre-training on pairs, in-batch negatives, `cached_mnrl` (gradient caching) by default, no-duplicates batches | Large effective batches are what make weak pairs useful | `init_from` (Hub id or path), `train_data` | `model`, `checkpoints` | `eval/ndcg@10` min (a floor) |
| `embeddings.mine_negatives` | The given model ranks the corpus for every training query and keeps `num_negatives` documents from a rank window, after false-negative filters | Hard negatives carry most of the fine-tuning signal; unfiltered ones are often unlabeled positives | `model`, `train_data`, `corpus`, optional `cross_encoder` teacher | `train` (query, positive, negative_1..n[, scores]), `stats` | `mine/kept_fraction` min |
| `embeddings.finetune` | Supervised training on the mined n-tuples from an earlier model; optional Matryoshka dims; optional margin distillation of teacher scores | The quality stage | `init_from: ${pretrain.model}`, `train_data: ${mine.train}` | `model`, `checkpoints` | `eval/ndcg@10` vs `pretrain.eval/ndcg@10`, `max_drop: 0` |
| `embeddings.eval` | nDCG@10, MRR@10, MAP@100, Accuracy@1, Recall@k on the eval set, at full dimension and per truncated dimension | One comparable number per model; also usable on a published model as a baseline | `model`, `eval_set` | `metrics` | relative gates against earlier stages |
| `embeddings.export` | Saves the final model (optionally truncated, normalised, fp16), reloads it and encodes once, optional ONNX copy, optional Hub push | The shipped directory is checked to load and to have the stated dimension | `model` | `model`, `summary`, `onnx` when written | - |

`pretrain` and `finetune` share one implementation (`train.py`) and the same parameters; they
differ in defaults (loss, batch size, learning rate) and in what they are for.

### How the eval split works

`eval_fraction` assigns a query to eval by a hash of its normalised text and the seed, not by row
order: every pair of a query lands on one side, and the split does not move when rows are added.
The eval corpus is the eval documents plus up to `max_eval_distractors` training documents. The
mining corpus (`corpus`) holds training documents only. With `eval_sources` (a ready dev split),
training pairs whose query is an eval query are removed and counted as `data/dropped_leaked`.

### False-negative filtering

The filters are the library's (`sentence_transformers.mine_hard_negatives`); the stage exposes
them and sets conservative defaults:

- `range_min: 10`, `range_max: 100`: skip the nearest ranks, look no further than rank 100
- `relative_margin: 0.05`: drop candidates scoring above `positive - 5%`
- `absolute_margin`, `max_score`, `min_score`: off by default
- `cross_encoder`: a teacher that rescores candidates; the filters then apply to its scores

Rows that end with fewer than `num_negatives` negatives are dropped (`mine/kept_fraction` reports
how many survived). `output_scores: true` keeps `[positive, negative_1..n]` scores per row.

### Losses

| `loss.name` | Use |
|---|---|
| `mnrl` | InfoNCE over in-batch negatives and the row's hard negatives |
| `cached_mnrl` | The same with gradient caching: batch size is the contrastive batch, `mini_batch_size` what is encoded at once |
| `margin_mse` | Only distil teacher score margins (needs a `scores` column) |

`matryoshka_dims` wraps any of them (include the full dimension). `distill_weight > 0` adds
`weight x MarginMSE` to `mnrl`; the student score is `distill_scale x (dot | cosine)`, so a model
ending in `Normalize` needs a scale that lets its margins reach the teacher's.

### Prompts

`encoder.prompts: {query: "query: ", document: "passage: "}` is stored in the model under the
names `query` and `document`. Training applies the query prompt to the `query` column and the
document prompt to every other text column; evaluation and mining use the same named prompts.
A later stage that loads the saved model inherits them without repeating the setting.

## Metrics and gates

Training stages report `train/loss`, and with an `eval_set` the same `eval/...` metrics as the
`eval` stage, so a gate can compare stages directly:

```yaml
gate: {metric: eval/ndcg@10, baseline: pretrain.eval/ndcg@10, max_drop: 0.0}
```

Per-dimension metrics are `eval/dim<d>/ndcg@10` etc. In-training metrics (for
`metric_for_best_model`) keep the library's names: `eval_ir_cosine_ndcg@10`, `eval_loss`.

## Running

```bash
pip install -e ".[embeddings]"
stagetrain kinds embeddings --params
stagetrain plan configs/embeddings/retrieval.yaml
stagetrain run  configs/embeddings/retrieval.yaml
stagetrain run  configs/embeddings/retrieval.yaml --from mine --set mine.num_negatives=8
```

- `configs/embeddings/retrieval.yaml`: the full recipe from a plain encoder.
- `configs/embeddings/distill.yaml`: no pre-training; a published embedding model is adapted with
  a cross-encoder teacher, and a `baseline` eval stage gives the number to beat.

A training stage that is interrupted resumes from its last checkpoint on the next `run`. Its
checkpoints are discarded instead when the stage is rerun after it finished or with other
parameters (`checkpoints/run.json` records which run they belong to).

## Tests

`tests/test_embeddings_data.py` (no torch needed): cleaning, the query-hash split, eval-set
construction, column ordering, parameter validation, that the track imports without torch, and
that every YAML in `configs/embeddings/` validates.

`tests/test_embeddings_pipeline.py` builds a 2-layer, 32-dim random BERT and a word-level
tokenizer in a temp directory and runs the real stages on CPU through `Pipeline`:
prepare_data → pretrain (cached InfoNCE, prompts) → mine_negatives → finetune (Matryoshka +
distillation) → eval (relative gate) → export (truncated, normalised), then resume, a failing
gate, the mining filters, a tiny cross-encoder teacher, `margin_mse` alone, triplet input with
best-checkpoint selection, and checkpoint resume. Nothing is downloaded.

Not covered by the tests: model quality (the tiny model only proves the plumbing), the ONNX copy
(`optimum` / `onnxruntime` are not installed here, so only the skip path runs), `hub_repo` push,
FAISS search (`use_faiss`), multi-GPU training, and the two example configs beyond parameter
validation (their datasets and models are never fetched).
