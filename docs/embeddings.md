# Embeddings track

Multi-stage training of a text embedding model (bi-encoder) with sentence-transformers, with the
goal of the highest retrieval accuracy on a domain. Which stages exist, their order and their
defaults follow [docs/research/embeddings.md](research/embeddings.md): start from a published
embedder, mine with a stronger one, distil a cross-encoder's ranking, measure against baselines
on queries nothing was chosen on, and mix some of the starting weights back in.

```
prepare_data ─┬─> bm25 ──> eval (untuned model)                      the two numbers to beat
              └─> mine_negatives ──> finetune x N (matrix) ──> select ──> merge ──> eval ──> export
```

The track is independent of the other three: its pipelines hold only `embeddings.*` stages.
Example pipelines in `configs/embeddings/`:

| Config | What it is |
|---|---|
| `retrieval.yaml` | The research-driven recipe: baselines, a strong miner with teacher scores, a temperature x learning-rate sweep with listwise distillation, a merge with the starting model, a paired test evaluation |
| `distill.yaml` | The shortest useful recipe: one small multilingual embedder, one teacher, one run |
| `pretrain.yaml` | Pre-training a plain encoder first. Only for a domain or language no published embedder covers, with tens of millions of weak pairs |

## Stages

| Kind | What it does | Why it is there |
|---|---|---|
| `embeddings.prepare_data` | Cleans and deduplicates (query, positive[, negative]) pairs; splits queries into train, validation (`eval`) and test; builds a retrieval eval set for each held-out split, with training documents as distractors | A query is in one split only. Validation chooses checkpoints and settings; test is only reported |
| `embeddings.bm25` | BM25 over an eval set, with the same metrics and per-query scores as `embeddings.eval` | Keyword search needs no model and is still best on some corpora: the first number to beat |
| `embeddings.eval` | nDCG@10, MRR@10, MAP@100, Accuracy@1, Recall@k; the 95 % interval of nDCG@10; each query's own score; with `compare_to`, the paired difference to another evaluation of the same queries | Two means a point apart say little on a few hundred queries; a paired difference does |
| `embeddings.mine_negatives` | A miner ranks the corpus for every training query and keeps `num_negatives` documents that score clearly below the positive; an optional cross-encoder teacher rescores them and its scores are kept | Hard negatives carry the fine-tuning signal; unfiltered ones are often unlabelled positives |
| `embeddings.finetune` | Trains on the mined rows: contrastive (`mnrl`, `cached_mnrl`, `gist`, `cached_gist`), distilled from the teacher's scores (`kl_div`, `margin_mse`), or both; optional Matryoshka dimensions | The quality stage. Run it as a `matrix` over the temperature and the learning rate |
| `embeddings.select` | The best run of a matrix by one metric; hands on that run's artifacts | The temperature has no published sweep to lean on |
| `embeddings.merge` | A weighted average of models' weights: the fine-tuned model with the model it started from, or several runs | Domain tuning lowers general retrieval ability; interpolating returns most of it for about a point in-domain |
| `embeddings.export` | Saves the final model (optionally truncated, normalised, fp16), reloads it and encodes once; optional ONNX copy; optional Hub push | The shipped directory is checked to load and to have the stated dimension |
| `embeddings.pretrain` | Contrastive pre-training of a plain encoder on weak pairs with in-batch negatives (cached InfoNCE) | Not in the default recipe; see "When to pre-train" |

`finetune` and `pretrain` share one implementation (`train.py`) and the same parameters; they
differ in defaults and in what they are for.

### Mining

Three choices decide what mining is worth, in the order of their measured effect:

- **Who mines.** `model` is the miner. A stronger embedder than the model being trained finds
  better negatives than the model itself (about +3 nDCG@10 in the one controlled comparison);
  BM25 negatives alone are worse than random ones. Use a strong published embedder. To
  self-mine, point `model` at the student or at an earlier stage's model.
- **The false-negative filter.** `relative_margin: 0.05` keeps a candidate only if it scores
  under 95 % of its positive. This single filter beat skipping ranks, an absolute score cap and
  an absolute margin. `absolute_margin`, `max_score` and `min_score` are available and off.
- **No rank skipping on top.** The library applies `range_min` after the score filters, to the
  candidates that already passed them, so `range_min: 10` with a margin throws away the ten
  hardest approved negatives. The default is `range_min: 0`. `range_max` is how deep the miner
  retrieves.

With `cross_encoder`, a teacher rescores every candidate and the filters and the written scores
are the teacher's. **`teacher_scores` says what those scores are.** A reranker with one output
is read through a sigmoid by the library, so by default (`probability`) scores lie in (0, 1) and
margins and caps are in those units; `logit` reads the raw output instead. A margin that no
candidate could meet (`absolute_margin: 1.0` on probabilities) is rejected before anything runs.

Rows that end with fewer than `num_negatives` negatives are dropped; `mine/kept_fraction`
reports how many survived. `output_scores: true` keeps `[positive, negative_1..n]` scores per
row. Use 4 to 7 negatives for contrastive training (more is flat) and up to about 19 for
listwise distillation.

### Losses

| `loss.name` | Use |
|---|---|
| `mnrl` | InfoNCE over in-batch negatives and every row's hard negatives |
| `cached_mnrl` | The same with gradient caching: the batch size is the contrastive batch, `mini_batch_size` what is encoded at once |
| `gist` | InfoNCE in which a `guide` model masks in-batch candidates it scores as high as the positive, so likely false negatives are not pushed away. Reads at most one hard negative per row |
| `cached_gist` | `gist` with gradient caching; reads every hard negative of a row |
| `kl_div` | Only distil: the student's ranking of `[positive, negatives]` matches the teacher's (needs a `scores` column) |
| `margin_mse` | Only distil: the student's positive-negative margins match the teacher's |

`distill_weight > 0` adds a distillation term (`distill_loss`, `kl_div` by default) to `mnrl` or
`gist`, and `contrastive_weight` scales the contrastive term. The combination that gained in the
controlled comparison:

```yaml
loss:
  name: mnrl
  distill_weight: 1.0        # listwise KL on the teacher's ranking
  contrastive_weight: 0.1    # + 0.1 x InfoNCE
  student_temperature: 0.05
  teacher_temperature: 0.3
```

Contrastive fine-tuning alone often lowers an embedder that is already supervised; this is why
the default recipe distils.

Details that matter:

- **Listwise KL normalises each query's teacher scores to [0, 1]** (`normalize_scores`), so the
  teacher's scale, logits or probabilities, does not matter and one `teacher_temperature` fits
  both. The library multiplies its KL by the student temperature squared, which at 0.05 makes it
  400 times smaller than a contrastive term beside it; the track divides that out, so the two
  weights mean what they say.
- **`margin_mse` matches raw margins**, so `distill_scale` must let the student's margins reach
  the teacher's: a cosine is at most 2 apart, probabilities at most 1, logits several units.
- **`scale` is 1 / temperature of the contrastive term.** Recent embedding models train at a
  temperature of 0.01 to 0.02 (scale 50 to 100), the library defaults to 0.05 (scale 20), and no
  published sweep settles it. The default here is 50; sweep it.
- **`symmetric: true`** adds the document-to-query direction as a second softmax. Evidence for
  it is contested.
- `matryoshka_dims` wraps any of the losses (include the full dimension). The cost at full
  dimension is about zero; how well a truncated dimension holds up is model-dependent, so
  evaluate the dimensions you ship (`dims` on `embeddings.eval`).

### Evaluation and gates

`embeddings.eval` and `embeddings.bm25` both write `scores`: each query's nDCG@10. Pass one
stage's `scores` as another's `compare_to` and the second reports `eval/delta_ndcg@10` with a
95 % paired-bootstrap interval:

```yaml
- name: test
  kind: embeddings.eval
  params: {model: "${merge.model}", eval_set: "${data.test_set}", compare_to: "${untuned.scores}"}
  gate:
    - {metric: eval/delta_ci_low, min: 0.0}                                # better beyond chance
    - {metric: eval/ndcg@10, baseline: bm25.eval/ndcg@10, max_drop: 0.0}   # and not below BM25
```

Which split is used where: `train` trains and is mined; `eval_set` (validation) picks the best
checkpoint, the best run of a matrix and the merge; `test_set` is only read by the baselines and
the final evaluation. A gate with zero tolerance on the validation queries treats sub-noise
differences as real; the paired gate on the test queries does not.

How the split works: `eval_fraction` and `test_fraction` assign a query by a hash of its
normalised text and the seed, not by row order, so every pair of a query lands on one side and
the split does not move when rows are added. With `eval_sources` or `test_sources` (ready
splits), training pairs whose query is held out are removed and counted as `data/dropped_leaked`.
Each held-out corpus is that split's documents plus up to `max_eval_distractors` training
documents; the mining corpus holds training documents only.

`nanobeir: true` on `embeddings.eval` adds `eval/nanobeir_ndcg@10`, a regression alarm for
general retrieval ability. It downloads the NanoBEIR datasets on first use and is not exercised
by the tests.

Training stages report `train/loss` and, with an `eval_set`, the same `eval/...` metrics, so
stages compare directly. Per-dimension metrics are `eval/dim<d>/ndcg@10` etc. In-training metric
names (for `metric_for_best_model`) keep the library's form: `eval_ir_cosine_ndcg@10`.

### Merging

```yaml
- name: merge
  kind: embeddings.merge
  params: {models: ["${best.model}"], base: "${vars.base_model}", base_weight: 0.25, eval_set: "${data.eval_set}"}
  gate: {metric: eval/ndcg@10, baseline: best.eval/ndcg@10, max_drop: 0.02}
```

`base_weight` is the starting model's share of the result. In-domain validation cannot choose
it: it always prefers no base at all. Choose it by how much in-domain score you will give up
(the gate above) for general ability, and check the latter with `nanobeir: true`. `group: <matrix
name>` averages every run of a matrix instead of listing `models`; averaging helps when the runs
differ in data or task, not when they are checkpoints or seeds of one run.

### Prompts

`encoder.prompts: {query: "query: ", document: "passage: "}` is stored in the model under the
names `query` and `document`. Training applies the query prompt to the `query` column and the
document prompt to every other text column; evaluation and mining use the same named prompts. A
later stage that loads the saved model inherits them. Keep the prompts and the pooling a
published checkpoint was trained with; a miner loaded from the Hub keeps its own.

### When to pre-train

`embeddings.pretrain` turns a plain encoder into an embedding model. Skipping it costs a plain
encoder 5 to 8 points, but that gap is closed by pre-training at a scale of 10^8+ pairs and
batches of 16k to 32k, and a published embedder already contains such a run. Pre-train yourself
only when no public model covers the language or domain (or may be used), and you have tens of
millions of weak pairs and can run contrastive batches in the thousands. `configs/embeddings/
pretrain.yaml` is that recipe.

## Running

```bash
pip install -e ".[embeddings]"
stagetrain kinds embeddings --params
stagetrain plan   configs/embeddings/retrieval.yaml
stagetrain run    configs/embeddings/retrieval.yaml --set vars.pairs=/data/my_pairs.jsonl
stagetrain report configs/embeddings/retrieval.yaml --open
stagetrain run    configs/embeddings/retrieval.yaml --from merge --set merge.base_weight=0.4
```

A training stage that is interrupted resumes from its last checkpoint on the next `run`. Its
checkpoints are discarded instead when the stage is rerun after it finished or with other
parameters.

## Not implemented

- **Synthetic queries for unlabelled documents** (generated by an LLM, filtered by round trip,
  labelled by the teacher). It is the next step by expected gain, largest when real pairs are
  few, and it needs a generation backend. The rows it produces are ordinary `query` / `positive`
  pairs, so they would enter through `prepare_data`.
- Relabelling false negatives as positives (rows with several positives), a second mining
  round, batches from one source at a time, hardness weighting, embedding-space distillation,
  LoRA for LLM-based embedders, multi-GPU training.

## Tests

`tests/test_embeddings_data.py` (no torch needed): cleaning, the query-hash split, eval-set
construction, per-query metrics, the bootstrap interval and the paired difference, BM25, score
normalisation, merge weights, parameter validation (including the impossible teacher margin),
that the track imports without torch, and that every YAML in `configs/embeddings/` validates.

`tests/test_embeddings_pipeline.py` builds a 2-layer, 32-dimension random BERT and a word-level
tokenizer in a temp directory and runs the real stages on CPU through `Pipeline`:
prepare_data (three splits) → bm25 and the untuned model on the test queries → pretrain →
mine_negatives → finetune as a two-run matrix (listwise distillation + contrastive, Matryoshka) →
select → merge with the starting model → eval → paired test evaluation → export. It also checks
that per-query scores average to the evaluator's nDCG@10, that the merged weights are the
weighted average, that a teacher's scores are probabilities unless logits are asked for, that the
listwise KL is on the scale of a plain KL divergence, every other loss path (`kl_div`,
`margin_mse`, `gist`, `cached_gist`, cached symmetric), a matrix-wide merge, resume and a failing
gate. Nothing is downloaded.

Not covered by the tests: model quality (the tiny model only proves the plumbing; on its
synthetic data BM25 is perfect and a random encoder is nearly so), the three example configs
beyond parameter validation (their models and datasets are never fetched; the ids and column
names were checked against Hub metadata), `nanobeir`, the ONNX copy (`optimum` / `onnxruntime`
are not installed here, so only the skip path runs), `hub_repo` push, FAISS search
(`use_faiss`), and multi-GPU training. No real cross-encoder was run: that
`BAAI/bge-reranker-v2-m3` yields probabilities is read from the library's source and the
model's published config.
