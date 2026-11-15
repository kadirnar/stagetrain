# Training research: text embedding models for the highest retrieval accuracy

Date: 2026-11-15. Scope: bi-encoders fine-tuned with sentence-transformers on thousands to
hundreds of thousands of (query, document) pairs, on one or a few GPUs.

**The question.** Which training steps raise the retrieval accuracy of a fine-tuned embedding
model, in what order, and which popular steps do not? The answer decides what the `embeddings`
track's pipeline contains.

**Short answer.** Start from a published embedding model, not from a plain encoder. Mine hard
negatives with a stronger embedder than the one being trained, and keep only candidates that
score clearly below the positive. Train by distilling a cross-encoder's ranking, with a small
contrastive term, because contrastive fine-tuning alone often lowers a model that is already
good. Measure against BM25 and the untuned model on queries nothing was chosen on, and compare
query by query. Then mix some of the starting weights back in.

## How to read this

Nothing here was run: this is a reading of papers, documentation and the installed library's
source. Three limits apply.

- Numbers marked **re-checked** were read on the primary page (paper table or model card). The
  others come from research passes that state they read the cited page; they are marked
  **not re-checked**. Check any figure against its source before quoting it elsewhere.
- Most controlled results are English, on BEIR / MTEB / MS MARCO. Transfer to another domain
  or language is an assumption.
- Several findings rest on one paper. They are marked.

Evidence labels: **[C]** a controlled comparison in a paper, **[P]** practitioner consensus
(several recipes agree, no clean ablation), **[A]** one anecdote.

## Findings, by expected gain per unit of cost

| # | Intervention | Evidence | What the pipeline does |
|---|---|---|---|
| 1 | **Start from a published embedding model.** Pre-train a plain encoder yourself only with tens of millions of weak pairs and contrastive batches in the thousands | [C] Skipping weak pre-training costs a plain encoder 5 to 8 points (E5: 41.5 vs 48.7 nDCG@10; GTE: 57.8 vs 62.4, not re-checked), and that gap is closed only at scale: batch 4,096 to 16,384 gives 45.36 to 46.97, and recent models pre-train on 10^8+ pairs (Arctic-embed, DenseOn, re-checked). No experiment shows a 1M-pair, batch-512 run beating a public embedder: this is an inference, not a measured head-to-head | `finetune.init_from` is a Hub embedder in the default recipe. `embeddings.pretrain` stays, in its own config, with its preconditions |
| 2 | **Mine with a stronger embedder than the student.** Never with BM25 alone | [C] NV-Retriever, mean nDCG@10 over three sets (re-checked): BM25 negatives 0.5002, random 0.5248, the student itself 0.5494, e5-mistral 0.5810. Ensembles of miners add nothing | `mine_negatives.model` is the miner and is documented as a strong published embedder; self-mining is one setting away |
| 3 | **The positive-aware filter alone.** Keep a candidate only if it scores under 95 % of its positive; do not also skip ranks | [C] Same paper (re-checked): naive top-k 0.5407, skip the top 10 ranks 0.5695, margin below the positive 0.5835, 95 % of the positive 0.5856. "Filter, then skip ranks" was tested nowhere, and the library applies `range_min` after the filter, so it removes the hardest negatives that passed | Defaults `relative_margin: 0.05`, `range_min: 0` |
| 4 | **Listwise distillation from a cross-encoder plus a small contrastive term** | [C] Tamber et al. 2025 (re-checked): base / InfoNCE only / listwise KL + 0.1 x InfoNCE, BGE-base FiQA 40.6 / 39.5 / 44.3, SciFact 74.1 / 72.1 / 76.2. Contrastive-only lowered BGE, GTE and Arctic on most sets. It is sometimes small: +0.36 in DenseOn (re-checked) | `loss: {distill_weight: 1.0, contrastive_weight: 0.1}` uses listwise KL on the teacher's scores, min-max normalised per query |
| 5 | **Evaluation that can tell a gain from noise.** BM25 and untuned baselines, separate validation and test queries, a paired interval on per-query nDCG | [C] Seed standard deviation of nDCG@10 is 0.1 to 0.5 on large sets and up to 2.0 on a few dozen queries (arXiv 2602.09229, re-checked). BM25 remains best on some corpora | `embeddings.bm25`; `test_fraction`; `embeddings.eval` writes per-query scores and, with `compare_to`, the paired bootstrap difference |
| 6 | **Sweep the temperature and the learning rate** | [P] Recent models train at a temperature of 0.01 to 0.02; the library default is 0.05. No retrieval sweep of the InfoNCE temperature was found. Published fine-tuning rates for a contrastive checkpoint are 3e-6 to 4e-5 | `matrix:` over `loss.scale` and `train.learning_rate`, then `embeddings.select`. The default scale is 50 |
| 7 | **Synthetic queries for unlabelled documents,** filtered by round trip and labelled by the teacher | [C] GPL: zero-shot 45.2, synthetic queries 48.8, with cross-encoder margins 51.5 (2021, older); confirmed in 2025, where synthetic queries matched human ones at equal count (re-checked). Saturates near 50k passages | Not implemented: it needs a generation backend. The rows it would produce are ordinary `query` / `positive` pairs |
| 8 | **Interpolate the fine-tuned weights with the starting model's** | [C] LM-Cocktail (2023, re-checked): HotpotQA 71.81 base / 75.96 tuned / 74.78 merged, while the mean of the other tasks went 49.81 / 47.49 / 49.98 | `embeddings.merge` with `base` and `base_weight` |
| 9 | **Mask in-batch false negatives with a guide model** | [C] GISTEmbed: +0.4 to +0.6 on full MTEB (not re-checked). Relabelling false negatives as positives gains more (+0.7 to +1.4 BEIR) but needs rows with several positives | `loss.name: gist` or `cached_gist` with a `guide`. Relabelling is not implemented |
| 10 | **Batches from one source at a time,** when a stage trains on several sources | [C] +3.2 nDCG@10 in pre-training (Arctic-embed, re-checked); small in fine-tuning | Not implemented: a stage trains on one file |

Already in the pipeline and supported by the evidence: duplicate-free batches, gradient caching
for pre-training, other rows' hard negatives as negatives (the installed loss does this),
Matryoshka dimensions (about zero cost at full dimension), the split by query, distractors in the
eval corpus, the best checkpoint by a retrieval metric, mixed precision, one epoch.

## What the evidence does not support

- **Self-mining as the reason for the staging** ("a model can only find good negatives once it
  retrieves reasonably"). A stronger external miner is about 3 points better.
- **Skipping the top ranks on top of a score filter.** Weaker than the filter alone, and flat in
  the one rank-range comparison found.
- **Weak pre-training at small scale,** and longer pre-training: more steps were worse in GTE and
  DenseOn (not re-checked).
- **Contrastive fine-tuning on mined negatives as a safe default** for an already supervised
  embedder. It often lowers it (finding 4).
- **More than about seven hard negatives, a large fine-tuning batch, or gradient accumulation as
  a substitute for a batch.** One to 31 negatives gave 68.96 / 69.02 / 68.96 / 68.95 (SFR);
  accumulation to 128 gave 77.2 against a true batch of 128 at 79.3 (GradCache, older). Neither
  was re-checked.
- **Hard negatives written by an LLM.** Worse than mined ones: 0.292 against 0.366 nDCG@10 over
  ten sets (arXiv 2504.21015, not re-checked).
- **A curriculum from easy to hard negatives, layer-wise learning-rate decay, LoRA for small
  encoders, averaging the checkpoints of one run** (0.6823 against 0.6818, one blog post). No
  supporting result.
- **MarginMSE at a small weight on raw teacher scores** for a dense student. The 0.05 weight comes
  from a sparse model's recipe; the dense results use listwise KL on normalised scores.
- **A symmetric loss as one softmax over both directions.** The variant with evidence is two
  softmaxes, one per direction, and even same-tower terms are contested (GTE +0.5, Gemini
  Embedding removed them). `symmetric: true` now builds the two-softmax form.
- **A zero-tolerance gate on the queries that chose the checkpoint.** It treats differences
  smaller than seed noise as real and reports an optimistically selected number.
- **Truncating to 64 dimensions, or binarising, a model not trained for it.** Retention is
  model-dependent and can be poor.

## Two defects this research found in the pipeline

Both follow from reading the installed library's source, not from a run.

1. **A teacher's scores are probabilities, not logits.** `CrossEncoder` applies a sigmoid when a
   model has one output and its config stores no other activation, which is the case for
   `BAAI/bge-reranker-v2-m3`. The old `distill.yaml` filtered with `absolute_margin: 1.0`
   ("teacher logits"): with scores in (0, 1) no candidate can be 1.0 below its positive, so
   mining kept nothing. The stage now has `teacher_scores: probability | logit`, rejects an
   impossible margin before anything runs, and listwise distillation normalises each query's
   scores, which makes the teacher's scale irrelevant to the loss.
2. **`range_min` runs after the margin filters.** The old default `range_min: 10` with
   `relative_margin: 0.05` discarded the ten hardest negatives that had already passed the
   false-negative filter. The default is now `range_min: 0`.

## Open questions

No controlled evidence was found for any of these, so the pipeline takes no position beyond a
default:

- a fine-tuning comparison across the 2025-2026 model families, or of a public embedder against
  a small own pre-training run
- an InfoNCE temperature sweep with retrieval numbers
- the gain of a second mining round; MarginMSE against KL for dense students; z-score against
  min-max normalisation; `bge-reranker-v2-m3` as a teacher
- mixing general pairs into domain data; averaging seeds
- quantisation-aware training of embedding outputs (the installed library has no such loss)
- how well NanoBEIR tracks full BEIR; which metric to select on when a reranker follows
- the effect of paraphrased queries on both sides of an exact-match query split

## Sources

Papers, by arXiv identifier:

- 2407.15831 NV-Retriever: improving text embedding models with effective hard-negative mining
- 2505.19274 and 2502.19712 Tamber et al.: cross-encoder listwise distillation and synthetic queries
- 2405.05374 Arctic-embed; 2607.27178 DenseOn / LateOn
- 2212.03533 E5; 2308.03281 GTE; 2309.07597 BGE; 2402.03216 BGE-M3
- 2506.05176 Qwen3-Embedding; 2509.20354 EmbeddingGemma; 2503.07891 Gemini Embedding
- 2507.11412 Ettin (encoders against decoders of equal size)
- 2311.13534 LM-Cocktail; 2602.05787 BOOM
- 2402.16829 GISTEmbed; 2505.16967 RLHN (relabelling hard negatives)
- 2112.07577 GPL; 2209.11755 Promptagator
- 2101.06983 GradCache; 2504.21015 (LLM-written negatives)
- 2602.09229 (seed variance; cosine against dot product)
- 2406.16048 D-MERIT (incomplete relevance labels); 1905.11096 Urbano et al. (significance tests)
- 2010.02666 MarginMSE

Other:

- Salesforce SFR-Embedding blog: https://www.salesforce.com/blog/sfr-embedding/
- Jina, model soups for embeddings: https://jina.ai/news/model-soups-recipe-for-embeddings
- sentence-transformers documentation, prompts: https://sbert.net/examples/sentence_transformer/training/prompts/README.html
- The source of `sentence-transformers 6.3.0` (losses, `mine_hard_negatives`, `CrossEncoder`, evaluators)
- The published config of `BAAI/bge-reranker-v2-m3`
