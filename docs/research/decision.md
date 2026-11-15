# Training research: decision models, for accuracy and calibration

Date: 2026-11-15. Scope: a causal LLM of 0.5B to 4B parameters that answers a typed question
(yes/no, one of N options, an ordered score) about a state in one forward pass, by a softmax over
the options' tokens at one position. The reference model is
[LiquidAI/d1-3B](https://huggingface.co/LiquidAI/d1-3B).

**The question.** What makes such a model more accurate and better calibrated on questions it was
not trained on, and what in the pipeline should change?

**Short answer.** The data and the measurement matter more than the training method. The
largest levers are how many different sources the model is trained on, and whether its
temperature is fitted, and its accuracy measured, on sources it has never seen. Loss tricks
(label smoothing, focal, Brier terms), a trained head, full fine-tuning and weight soups add
little or nothing.

## How to read this

Nothing here was run. Three limits apply.

- **The field is weeks old.** Single-pass decision models became a category in autumn 2026. d1's
  own recipe is unpublished beyond four sentences in its release post.
- **The usable evidence is thin.** One paper with ablations (StartLux-Decision, arXiv
  2610.17309); one public research log with matched controls and bootstrap intervals but one
  seed per run (Kev, github.com/jaredpalmer/kev); the source of Unsloth's decision-model code;
  and older papers on calibration, instruction-tuned classifiers and weight averaging, several
  cited from memory.
- **Numbers were read through a summarising fetch tool.** Check them before quoting.

Evidence labels: **[C]** a controlled comparison in a paper, **[P]** practitioner consensus with
no controlled test found, **[A]** one uncontrolled or unreplicated result.

## Findings, by expected gain per unit of cost

| # | Finding | Evidence | What the pipeline does |
|---|---|---|---|
| 1 | **Fit temperatures on sources the model was not trained on,** per question type and option-count band, on hard labels, inside bounds, on rows not used to select the soup | [A] A temperature fitted on a split of the training sources left ECE at 0.059 to 0.065 on new tasks; fitted on a pool of held-out datasets, 0.0085 (Kev). [C] Calibration fitted in-distribution left up to 6 points of over-confidence on unseen datasets (StartLux; Ovadia 2019). [A] A temperature of 0.05 chosen on an accuracy-like score turned 92 % of answers into 0 or 1 | Sources can be `role: heldout`. `decision.calibrate` fits per type and per option-count band (2, 3-5, 6-10, 11+), on hard labels, within [0.5, 5], and reports out-of-fold numbers. The soup is selected on other rows |
| 2 | **Judge the model on whole held-out sources,** with intervals and paired comparisons, per-source proper scores, and the share of confident errors | [C] Seed-to-seed spread is 0.2 to 0.7 points, larger than most effects below (StartLux). [A] An unpaired bar of +0.01 Brier failed an equally good model 36 to 55 % of the time (Kev) | `decision.eval` reports accuracy with a bootstrap interval, a chance-corrected macro average over sources, NLL and Brier per source, wrong answers given at 90 % or more, and a paired difference against another evaluation of the same records |
| 3 | **Breadth of sources.** Many task families with few rows each | [C] Accuracy on unseen tasks grows log-linearly with the number of training tasks and saturates near 64 rows per task (Super-NaturalInstructions). [C] Families present in training gained 8.6 points, others 0.3 (StartLux) | The example config has 14 training sources capped at 5,000 rows. That is still few; others train on 50 to 300 |
| 4 | **No label smoothing; real label distributions as soft targets** | [A] Label smoothing, focal loss and a Brier term did not beat a matched cross-entropy run (Kev). [A] Softening only ambiguous rows: Brier -0.011, accuracy +1.1 | Label smoothing is off. A binary source whose label is a share of annotators can keep it as a soft target (`threshold_soft`) |
| 5 | **Long states with the evidence buried** | [A] Burying the state among unrelated records cost 22 to 43 points; training on such rows recovered 17 to 20 (Kev). d1's post names long inputs first | `buried_fraction` places a share of a source's states among other states of that source, and the question names the record it is about |
| 6 | **Vary the question itself:** fewer options, a none-of-the-above option, a yes/no question derived from a choice | [P] Used by Unsloth, Kev, Clef and StartLux. Nobody has measured them | `subsample_options`, `none_of_the_above`, `derive_noul`, off by default |
| 7 | **Averaging over option orders at inference** | [A] Order still flips answers after shuffled training; averaging over rotations cut flips from 2.8 % to 0 at twice the cost or more (Kev) | `DecisionModel(orders=k)`, off by default. A per-position prior correction is not implemented: it is untested on fine-tuned models |
| 8 | **Choose checkpoints on held-out NLL, not accuracy** | [C] NLL starts overfitting before accuracy does (Guo 2017) | The example config keeps the checkpoint with the lowest held-out NLL |
| 9 | **A soup of three or four seeds** | [C] StartLux averages 3 to 4 seeds and reports no number for the gain; model soups in general give about 0 to 1 point and no calibration gain | Kept as it is, not expanded. `decision.soup` can take the runs of a seed matrix |
| 10 | **Bidirectional attention in part of the network** | [C] Up to +2.4 index points (StartLux) | Not implemented. It changes the model, loses prefix caching, and is the most expensive item here |

Already in the pipeline and supported: option shuffling at every step, several instructions per
source, description dropout, soft ordinal targets for scores, a content-hash split, LoRA, and the
blind-state and permutation probes.

## What the evidence does not support

- **Label smoothing, focal or Brier terms as calibration fixes.** See finding 4. Models trained
  that way are also harder to calibrate afterwards.
- **An "align" stage on short inputs before the main training.** Warm-up helps a newly
  initialised head; this readout has none. Nothing supports short-then-long ordering. What is
  supported is having long inputs in the data. The example config no longer has the stage.
- **One global ECE gate over mixed sources,** and ECE on a few hundred rows. The same model can
  be under-confident on one kind of question and over-confident on another.
- **Soups as a major lever.** Three training runs for at most a point.
- **Full fine-tuning at this data size.** No accuracy gain and slightly worse calibration than
  LoRA on the same data (Kev).
- **A permutation-consistency loss.** Negative in Kev's log. Shuffling is the supported fix.
- **A large trained decision head.** Reading an adapted backbone through the LM head scored the
  same as a pointer head (Kev), and the one system with a large trained head had the worst
  calibration of four measured (StartLux). This is the reason to keep the logit readout.
- **Reinforcement learning with proper scoring rules** for a single-pass readout. The gradient of
  the log score is the cross-entropy gradient.
- **Comparing with d1 on BoolQ, Civil Comments, XNLI or PAWS-X while training on them.** Those
  are d1's reported benchmarks. In the example config PAWS-X is held out; BoolQ and Civil
  Comments are trained on, so a comparison there says nothing.

## The Decision Index

The public benchmark these models are compared on (github.com/apolinario/decision-index): 37
benchmarks in five areas, about 110,000 requests, `choice` and yes/no questions only.

- Score per benchmark: `clip((raw - chance) / (1 - chance), 0, 1)`, averaged with area weights.
  `decision.eval` reports the same chance correction as `eval/macro_skill`.
- A request that does not fit the model's context counts as wrong. The pipeline cuts long states
  from the middle instead; for this benchmark `max_length` must cover the longest request.
- It does not measure calibration.

Running it is not part of this repository.

## Open questions

No evidence was found for any of these:

- d1's loss, data, calibration and attention layout
- how much a soup adds in a decision model (two builders do it, neither reports the difference)
- the vocabulary-mass term in this pipeline's loss, two-character option codes past 26 options,
  and cutting a state from the middle rather than the end
- several questions per row against one row per question, for accuracy
- a learning-rate or LoRA-rank sweep for this objective on models of this size
- whether an option-position prior stays stable across tasks after shuffled fine-tuning

## Sources

- StartLux-Decision, arXiv 2610.17309 (the one paper with ablations)
- Kev research log: https://github.com/jaredpalmer/kev (`PLAN.md`)
- Decision Index: https://github.com/apolinario/decision-index
- d1 release post: https://www.liquid.ai/blog/open-d1
- The decision-model source of `unsloth 2026.11.4` (`unsloth/models/decision*.py`)
- Super-NaturalInstructions, arXiv 2204.07705; FLAN, arXiv 2210.11416
- Guo et al., On Calibration of Modern Neural Networks, arXiv 1706.04599
- Ovadia et al., Can You Trust Your Model's Uncertainty?, arXiv 1906.02530
- Zheng et al., Large Language Models Are Not Robust Multiple Choice Selectors, arXiv 2309.03882
- Xue et al., Strengthened Symbol Binding, arXiv 2406.01026
- Model soups, arXiv 2203.05482
- Miller, Adding Error Bars to Evals, arXiv 2411.00640
