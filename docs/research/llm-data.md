# Training research: data quality for LLM fine-tuning

Date: 2026-11-15. Scope: chat and instruction data, raw text for continued pretraining,
preference pairs, and prompts with checkable answers for reinforcement learning.

**The question.** Which defects in fine-tuning data cost model quality, which only look bad, and
what should a data-quality stage drop, flag or merely report?

**Short answer.** Two defects have a measured cost: wrong answers, which damage the model, and
overlap between training and evaluation data, which inflates the reported result. Most of the
other usual checks (near-duplicates inside the training set, refusal boilerplate, leftover
markup, missing end punctuation) rest on plausible mechanism and on what released pipelines do,
not on a measurement. So `llm.check_data` drops only rows that are certainly useless, flags the
rest for a person, and never filters on a quality score by default.

## How to read this

Nothing here was run. Papers were read through a summarising fetch tool: check a figure against
its source before quoting it. Several key numbers come from single studies, some of them
single-seed.

Evidence labels: **[C]** a controlled comparison in a paper, **[P]** practitioner consensus or
what released pipelines do, **[A]** one uncontrolled or unreplicated result.

## What has a measured effect

| Finding | Evidence | What `llm.check_data` does |
|---|---|---|
| **Wrong answers hurt; surface noise does not** | [C] With 20 to 40 % of labels wrong, a small model's accuracy fell from 94.4 to 90.7, 81.6 and 57.7. Typos and grammar errors at the same rates stayed within about 1 to 2 points (arXiv 2604.12469; nothing was tested below 20 %) | Flags the same question with different answers (`conflict`). With a `scorer`, drops rows whose own reference answer the scorer rejects (`unverifiable`). Repairs broken encodings instead of dropping the row. A model can grade answers (`judge`); its grades flag |
| **Overlap between training and held-out data inflates the result** | [C] With near-copies of benchmark items at 5 % of a fine-tuning mix, accuracy on the unseen items rose 10 to 11 points. String matching finds few of them: the best single method removed 6.5 % (arXiv 2602.12413) | Drops a training row whose question repeats a held-out one: exactly, by word 8-grams covering more than half of the held-out question (the rule Tulu 3 used), or by overall similarity. The split by question in `llm.prepare_data` is the first defence |
| **Selecting rows by a quality score does not beat random selection on large pools** | [C] On pools of 0.4 to 1 million rows no scoring method was significantly better than random; selection that kept the data diverse did best (arXiv 2410.09335) | No score-based filtering by default. The report gives diversity measures instead |
| **Published preference labels are often wrong** | [C] A reward model agreed with the label in 70 to 80 % of pairs across four popular DPO datasets; a curated mix 30 % smaller scored 1.1 to 2.5 points higher (arXiv 2511.10985) | Drops pairs whose two answers are the same, flags near-identical pairs, reports how often the chosen answer is the longer one. Checking the labels with a reward model is not implemented |
| **Filtering on a confidence estimate can remove good rows** | [C] Removing the least confident half helped in five of six settings and hurt in one; correcting rows instead of removing them helped in all six (arXiv 2403.12776) | Flag by default, drop on request |

## What is checked without a measured effect

These checks are in the stage because the mechanism is clear and released pipelines do them. No
study was found that measures what they are worth, so most only flag.

| Check | Default | Why |
|---|---|---|
| Empty message | drop | Nothing to learn from |
| Chat-template markers or the tokenizer's control tokens inside a message | drop | The model learns to write them. Leftovers of an earlier rendering |
| An answer that repeats the prompt | drop | Teaches copying |
| An answer stuck in a loop (more than half of its word 10-grams repeat) | drop | Teaches looping |
| Near-duplicate of an earlier row (Jaccard 0.7 over word 5-grams) | flag | No result shows that removing them raises accuracy; in templated data they are legitimate. One pretraining result found global deduplication made the data worse |
| Refusals and "As an AI language model" boilerplate | flag | Widely removed for behaviour, not accuracy |
| Placeholders (`TODO`, `lorem ipsum`, `[insert ...]`) | flag | Usually an unfinished synthetic row |
| A long answer with no closing punctuation, or an open code block | flag | A sign of a cut-off generation. As a drop rule it removed 30 % of a web corpus in one case |
| Broken encoding, control characters, HTML debris | flag (encoding repaired with `ftfy` when installed) | Surface noise costs little |
| E-mail addresses, phone and card numbers, IP addresses, credentials | flag; `redact` on request | Regular expressions find formats, not names in prose. They are a floor, not a guarantee |
| Longer than `max_seq_length` | flag | The training stages drop these rows themselves |
| Reference answer copyable from the prompt (with a `scorer`) | flag | A reward can be collected by copying |

Reported and never acted on: token-length percentiles, distinct word 2-grams of prompts and
answers, the most common answer opening and its share, rows per source.

## What the evidence does not support

- **Removing near-duplicates inside the training set to raise accuracy.** The measured benefits
  are less memorisation and less wasted compute.
- **Quality scores or an LLM judge as a general accuracy lever.** Reported gains are on injected
  noise, small noisy sets, or vendor posts.
- **N-gram decontamination as sufficient.** It finds a small part of the semantic duplicates.
  Nor does an embedding-similarity threshold work on its own: items that were similar but not
  duplicates had no effect.
- **Dropping rows for typos and grammar.**
- **Perplexity filtering of fine-tuning data.** The evidence is from large-scale pretraining.
- **Regular-expression scrubbing as a privacy guarantee.** A widely used PII tool scored F1 of
  0.14 to 0.20 on two 2026 benchmarks (read in search summaries only).
- **Language identification on short or code-mixed rows.** Detectors are near zero there. The
  stage only checks the writing system, and only when asked.

## Gates

Thresholds for a gate on this stage. Almost all are engineering conventions, not measurements.

```yaml
gate:
  - {metric: quality/chat_drop_rate, max: 0.3}    # above this the source or the normaliser is broken, not the rows
  - {metric: quality/chat_conflict, max: 0.05}    # the same question with different answers
  - {metric: quality/chat_leak, max: 0.02}        # before cleaning: above this the split does not group related rows
```

Worth a look, not a stop: near-duplicates above 10 %, suspected truncation above 2 %, refusals
above 1 %, and for preference data a chosen-is-longer share outside 30 to 70 %.

## Tools

The stage needs no extra dependency: hashing, MinHash with locality-sensitive hashing, regular
expressions and counts are implemented in the track. Tools worth adding around it:

| Tool | For | Note |
|---|---|---|
| `ftfy` | repairing broken encodings | used automatically when installed |
| SemHash | semantic near-duplicates on CPU | optional next step; flags, never proves a leak |
| GlotLID, fastText | language identification | unreliable on short and code-mixed text |
| Data-Juicer | a large operator library | the best reference for operators and thresholds; heavy as a dependency |
| datatrove | web-scale raw text | for continued-pretraining corpora |

Not recommended as dependencies at this date: distilabel and Argilla (maintenance only), Lilac
(archived), NeMo Curator (its deduplication needs a GPU), cleanlab (built for classification
labels).

## Open questions

No evidence was found for any of these:

- the effect of template markers in content, truncated answers, placeholders, mixed-language
  rows or prompt-echo answers on a fine-tuned model
- the cost of label noise below 20 %, and of conflicting answers specifically
- a near-duplicate threshold derived from model quality instead of convention
- how to deduplicate template-generated data
- the effect of identical or near-identical preference pairs, and an acceptable length ratio
- whether an offline difficulty measure of RL prompts predicts the gain from RL

## Sources

- Noise in LLM fine-tuning: arXiv 2604.12469
- Soft contamination of benchmarks: arXiv 2602.12413
- Rethinking Data Selection at Scale: arXiv 2410.09335
- Preference dataset study: arXiv 2511.10985
- CLEAR (confidence-based filtering and correction): arXiv 2403.12776
- Tulu 3 (decontamination rule): arXiv 2411.15124
- FineWeb (deduplication and filter ablations): arXiv 2406.17557
- AlpaGasus: arXiv 2307.08701; LIMA: arXiv 2305.11206 (older, cited from memory)
- Data-Juicer operator list: https://github.com/datajuicer/data-juicer
- SemHash: https://github.com/MinishLab/semhash; text-dedup: https://github.com/ChenghaoMou/text-dedup
- datatrove: https://github.com/huggingface/datatrove
