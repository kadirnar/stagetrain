# Training research: encoder text classifiers for the highest accuracy

Date: 2026-11-15. Scope: single-label and multi-label text classification with a fine-tuned
encoder (ModernBERT, DeBERTa-v3, XLM-R and their relatives).

**The question.** Which training steps raise the accuracy of a fine-tuned encoder classifier, in
what order, and which popular steps do not? The answer decides what the `classifiers` track's
pipeline contains.

**Short answer.** The evidence supports spending effort on four things: the learning rate and
the number of epochs, the encoder and its size, several seeds per setting, and the data (label
errors, more labelled text). It does not support the regularisation tricks. No controlled result
from 2025 or 2026 was found for R-Drop, adversarial training, mixout, EMA/SWA or layer-wise
learning-rate decay on a ModernBERT-family encoder, and the older support for them shrinks
against a well-trained baseline.

## How to read this

Nothing here was run: this is a reading of papers, model cards and the installed source code.
Three limits apply to every number below.

- Many papers were read at abstract level only, and some older ones are cited from memory. The
  list is at the end; their figures are limited to what the abstracts state.
- Papers were read through a summarising fetch tool. Check a figure against the paper before
  quoting it elsewhere.
- Several results rest on one study, one language or the authors' own report. They are marked.

Evidence labels: **[C]** a controlled comparison in a paper, **[P]** practitioner consensus or
the recipe a model's authors used, **[A]** one uncontrolled result. "older" means before 2024.

## Findings, by expected gain per unit of cost

A prerequisite comes before the list: without intervals and baselines, none of the gains below
can be told from noise. With a 2,000-row eval split and 90 % accuracy the standard error is
0.0067, so a gate of 0.005 macro-F1 decides nothing.

| # | Intervention | Evidence | What the pipeline does |
|---|---|---|---|
| 0 | **Intervals and baselines.** Bootstrap intervals, paired comparisons on the same rows, a TF-IDF + linear baseline | [C, older] Typical test sets cannot detect the differences usually claimed (Card 2020). TF-IDF + a linear SVM was competitive with BERT on 4 of 6 LexGLUE datasets (Lin 2023) | `classifiers.eval` reports 95 % intervals and, with `compare_to`, the paired difference to another evaluation. `classifiers.baseline` is TF-IDF + logistic regression measured the same way |
| 1 | **Sweep the learning rate and the epochs, over at least three seeds, and judge a setting by its seed mean.** Evaluate several times per epoch with early stopping | [C] ModernBERT's authors selected 5e-5 or 8e-5 for the base model on every GLUE task and 1e-5 to 8e-5 for large; the model is "particularly sensitive to learning rate choices" (Antoun 2025). Five-seed standard deviations of 0.2 to 0.7 points are the size of most claimed regulariser gains | `matrix` over `train.learning_rate` and `seed`; `classifiers.select` with `average_over: [seed]`; `early_stopping_patience`. The default learning rate is 5e-5 |
| 2 | **The encoder and its size as a search dimension** | [C] A large model is worth about 2 GLUE points over its base sibling (88.4 to 90.4, 88.1 to 91.4), more than any single regulariser. On identical French pretraining data DeBERTa-v3 beat ModernBERT on every non-retrieval task (Antoun 2025, one language, base size) | `matrix` over `init_from` |
| 3 | **Set the input length from the data.** Head+tail truncation for models that stop at 512 tokens; mean pooling for long ModernBERT inputs | [C, older] Head+tail truncation was best for BERT on two long-document sets (Sun 2019); complex long-document models often fail to beat simple baselines (Park 2022) | `max_length: auto` fits the 99th percentile of the training rows; `truncation: head_tail`; `pooling` |
| 4 | **Distil a seed ensemble.** Average the seeds' probabilities over unlabeled text and train one student on gold plus those soft labels | [C, older] Ensembles and distillation; [C] seeds with the same accuracy differ widely in which rows they get right (Bui 2025) | `classifiers.pseudo_label` with `group` and `keep_all` |
| 5 | **Label the unlabeled pool with an LLM,** filter by confidence or agreement, keep gold at full weight | [C], mixed: comparable to human labels on 14 tasks (Pangakis 2024); lower accuracy, more variance and inherited systematic errors (Lu & Smith 2025) | Not a stage. LLM-labelled rows enter through `extra_train_data` with a `weight` |
| 6 | **Audit the labels.** Rank training rows by out-of-fold loss, relabel the top slice by hand | [C, older] Loss ranking with a fine-tuned model finds label errors 15 to 28 points more precisely than earlier methods; confident learning on top adds little (Chong 2022) | `classifiers.finetune` with `fold` as a matrix, then `classifiers.audit_labels` |
| 7 | **Masked-LM adaptation, judged by the downstream metric** | [C, older] Helps most when the domain is far from the pretraining text (Gururangan 2020); more robust than self-training (Shi 2023). No controlled result on ModernBERT with a no-adaptation control | `classifiers.mlm_adapt`; the recipe trains with and without it and lets the selection decide |
| 8 | **Temperature scaling and per-label thresholds, on data that did not pick the checkpoint** | [C, older] Temperature scaling fixes most in-domain miscalibration (Desai & Durrett 2020). For rare labels the F1-optimal threshold is unstable (Lipton 2014) | `classifiers.prepare_data` has a `calibration` split; rare labels share one threshold (`min_positives`) |
| 9 | **Layer-wise learning-rate decay, a faster head and head warm-up as things to search, not defaults** | [C, older] 0.95 was best on a 12-layer BERT (Sun 2019); such gains shrink once the baseline is trained properly (Zhang 2021). No current encoder recipe uses it | All three are off by default |
| 10 | **Plain cross-entropy by default.** For imbalance, logit adjustment; class weights and focal loss as options | [C], mixed 2025 results: plain CE matched or beat class weights and focal loss on macro-F1 in several comparisons. Logit adjustment: Menon 2021 (vision) | `loss: ce` is the default; `loss: logit_adjusted` |

## Model choice

| Situation | First choice | Evidence |
|---|---|---|
| English, inputs up to 512 tokens | DeBERTa-v3, base or large | [C] At least as accurate as ModernBERT in every controlled comparison found, at two to three times the compute. DeBERTa-v3-large 91.4 against ModernBERT-large 90.4 on GLUE |
| Long inputs, or throughput matters | ModernBERT, Ettin or UltraBERT | [C] 8k to 32k tokens. UltraBERT (October 2026) reports GLUE 89.5 and 91.0; self-reported and not replicated |
| Multilingual | mmBERT-base, with XLM-R-large as the alternative to try | [C] mmBERT-base beats XLM-R-base on XTREME (72.8 against 70.4). It was not compared with XLM-R-large or mDeBERTa-v3 on classification |
| A decoder LLM with a classification head | Not a default | [C] At equal size and data, encoders win MNLI at every size (Ettin) |
| Fewer than about 100 to 300 labels | Embeddings + logistic regression, SetFit, or a zero-shot LLM | [C] A fine-tuned encoder overtakes zero-shot LLMs from roughly that many labels (Pecher 2025) |
| 100k labels or more | A base encoder, one to two epochs, little tuning | [P] |

Facts about ModernBERT that change how it is trained:

- **Learning rate.** Base: 5e-5 to 8e-5. Large: 1e-5 to 5e-5. For DeBERTa-v3 and XLM-R use a
  lower grid (1e-5 to 3e-5).
- **Weight decay.** The authors' grid was 1e-6 to 1e-5, effectively none. It is not worth trials.
- **Dropout is zero** in the published configurations. R-Drop and multi-sample dropout do
  nothing unless dropout is switched on first.
- **Pooling.** The published checkpoints use mean pooling; the class default in transformers is
  CLS. On long inputs the CLS token does not attend to every token in the local-attention layers.
- **Precision.** NaN losses have been reported with fp16 and with flash attention. Use bf16.

## What the evidence does not support

- **`llrd: 0.9` as a default.** On a 22-layer model the embeddings then train at 9 % of the
  learning rate. The only controlled support is 0.95 on 12 layers.
- **Head warm-up (LP-FT) for in-distribution accuracy.** The evidence is for out-of-distribution
  robustness, mostly in vision.
- **Label smoothing on a teacher whose confidence is thresholded or distilled.** With pretrained
  language models it raised in-domain calibration error in a way temperature scaling did not
  repair (arXiv 2203.07559), and it degrades the ranking of correct above incorrect predictions
  (Xia 2025).
- **Self-training from one same-architecture teacher at a 0.95 confidence threshold.** It adds
  the rows the model already gets right. Shi 2023 finds masked-LM adaptation more robust.
- **Class weights or focal loss as the answer to imbalance.**
- **A drop in masked-LM loss as proof that adaptation helped.** It is nearly always positive and
  is not a downstream measure. Losses at different masking rates are not comparable either.
- **Masking at 30 % for adaptation because ModernBERT was pretrained with it.** UltraBERT reports
  GLUE falling from 89.5 to 88.3 as masking rises from 6 % to 30 %, and mmBERT anneals the rate
  down. Both are pretraining results, not ablations of continued pretraining.
- **Tuning weight decay; mixout; re-initialising top layers; noise-robust losses.**
- **Selecting or reporting the best single seed** (Dodge 2020).
- **A zero-shot LLM once a few hundred labels exist** (Bucher 2024, Pecher 2025).

## What is not implemented

- R-Drop, adversarial training (FGM, AWP), EMA / SWA, mixout: no evidence on current encoders.
- An LLM labelling stage and a zero-shot LLM baseline: both need an LLM endpoint.
- Asymmetric and distribution-balanced losses for long-tailed multi-label sets.
- Chunk-and-pool for documents longer than the model's window.
- Splits grouped by author, thread or date. `prepare_data` removes exact duplicates and texts
  labelled two ways, not near-duplicates.

## Open questions

No controlled evidence was found for any of these:

- continued masked-LM on ModernBERT against no adaptation, and a masking-rate ablation for it
- an NLI-tuned checkpoint as the starting point against masked-LM adaptation
- a long-context encoder at full length against head+tail truncation against chunk-and-pool
- soft against hard LLM labels for an encoder student; relabelling against removing label errors
- how many positives a per-label threshold needs
- mmBERT against XLM-R-large; an English, same-data ModernBERT against DeBERTa-v3; an
  independent replication of UltraBERT

## Sources

Read in this session (full text or the cited tables):

- ModernBERT, arXiv 2412.13663
- Antoun, Sagot, Seddah, "ModernBERT or DeBERTaV3?", arXiv 2504.08716
- Ettin, "Seq vs Seq", arXiv 2507.11412
- mmBERT, arXiv 2509.06888
- UltraBERT, arXiv 2610.15028 (abstract and part of the text)
- The published configuration of `answerdotai/ModernBERT-base`
- The installed source of transformers 5.23.0

Read at abstract level, or from a search summary:

- Card et al. 2020, arXiv 2010.06595; Lin et al. 2023, arXiv 2306.07111
- Pecher et al. 2025, arXiv 2402.12819; Bucher & Martini 2024, arXiv 2406.08660
- Pangakis & Wolken 2024, arXiv 2406.17633; Lu & Smith 2025, arXiv 2504.15432
- Chong, Hong, Manning 2022, arXiv 2205.12702; Zhu et al. 2022, arXiv 2204.09371
- Gururangan et al. 2020, arXiv 2004.10964; Shi et al. 2023, arXiv 2305.13002
- Wettig et al. 2023, arXiv 2202.08005
- Sun et al. 2019, arXiv 1905.05583; Park et al. 2022, arXiv 2203.11258
- Desai & Durrett 2020, arXiv 2003.07892; Xia et al. 2025, arXiv 2403.14715
- Menon et al. 2021, arXiv 2007.07314; Lipton et al. 2014, arXiv 1402.1892
- Mosbach et al. 2021, arXiv 2006.04884; Zhang et al. 2021, arXiv 2006.05987
- Bui et al. 2025, arXiv 2503.07329; Dodge et al. 2020, arXiv 2002.06305
- arXiv 2203.07559 (label smoothing and calibration of pretrained language models)
- EuroBERT, arXiv 2503.05500; NeoBERT, arXiv 2502.19587

Cited from memory, not reopened: Guo et al. 2017 (arXiv 1706.04599), SMART, FreeLB, mixout, deep
ensembles, Hinton et al. 2015. Claims about competition practice rest on recollection; the search
found no recent write-up.
