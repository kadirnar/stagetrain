# Decision track

Trains a **decision model** in the style of [LiquidAI/d1-3B](https://huggingface.co/LiquidAI/d1-3B): a
causal language model that takes a *state* (text or JSON) and typed *questions*, and returns a
probability distribution over each question's options in one forward pass. It generates no tokens,
so its output cannot break a schema and its cost is one read of the input.

```python
from stagetrain.tracks.decision.inference import DecisionModel

model = DecisionModel.load("runs/decision/d1-style-qwen3-0.6b/export/model")
model.decide("I was charged twice this month, please refund one of them.", {
    "refund":  {"type": "noul", "instructions": "Is the customer asking for a refund?"},
    "team":    {"type": "choice", "instructions": "Which team handles this?",
                "criteria": {"billing": "payments and invoices", "tech": "bugs and outages"}},
    "urgency": {"type": "score", "instructions": "How urgent is it?", "criteria": ["low", "medium", "high"]},
})
# {"answers": {"refund": {"noul": 0.97},
#              "team": {"choice": "billing", "confidence": 0.97, "probabilities": {...}},
#              "urgency": {"score": 1, "confidence": 0.61, "probabilities": [...], "expected": 1.2, ...}},
#  "usage": {"input_tokens": 212, "output_tokens": 0}}       (numbers are illustrative)
```

## How a decision is read

| Type | Options | Tokens that stand for an option |
|---|---|---|
| `noul` | yes, no | `yes` / `Yes` / `YES` and `no` / `No` / `NO` |
| `choice` | the named options | a positional code: `A`, `B`, ... (two characters past 26 options) |
| `score` | 2 to 10 ordered levels | the digits `0` ... `k-1` |

The state and the question are written into the tokenizer's own chat template, and the prompt stops
right after the assistant header: the **answer slot**. Each option owns the single tokens that write
its code, with and without a leading space. The option's logit is the log-sum-exp of those tokens'
logits at the slot, and the answer is the softmax over the options only
(`prompt.py`, `modeling.py`).

Rows are left-padded, so the slot is the last position of every row and the LM head runs at that
one position (`logits_to_keep=1`).

## What is trained

`decision.train` minimises, per record,

```
loss = CE(target, softmax over the options) - w * log P(any option token | whole vocabulary)
```

- The first term trains exactly the distribution the model is read with. Targets can be hard
  answers or soft distributions (an annotator share, a teacher's probabilities). `ordinal_sigma`
  blurs a score's target over neighbouring levels, so that being off by one level costs less
  than being off by three. `label_smoothing` exists and is off: it did not beat plain
  cross-entropy in the one matched comparison found, and it makes a model harder to calibrate.
- The second term (`vocab_mass_weight`, default 0.1) keeps the vocabulary's mass on option tokens,
  so the same model still writes a valid code when sampled from.

Train-time prompt variation, each one closing a shortcut:

| Setting | What it varies | Shortcut it removes |
|---|---|---|
| `shuffle_options` | a choice's option order, every time a record is seen | "the answer is usually B" |
| `description_dropout` | option names with or without descriptions | relying on descriptions being present |
| `mix_option_format`, `mix_state_format` | how options and states are laid out | one fixed layout |
| several `instructions` per source | the question's wording | wording as a cue for the answer |
| `subsample_options` | the answer and a random subset of the other options | one fixed option count per source |
| `none_of_the_above` | an added "none of the other options" option, right half the time | always having to pick something |
| `derive_noul` | "is this option the right answer?" as a yes/no question | a source that only ever asks one kind of question |

The last three change the question and are off by default. Several builders of such models use
them; none has published a measurement.

When a prompt is longer than `max_length`, only the state is cut, from its middle. The question,
the options and the slot are always whole.

## Stages

Which stages there are and what they default to follows
[docs/research/decision.md](research/decision.md).

| Stage | What it does | Why | Main outputs |
|---|---|---|---|
| `decision.build_data` | Converts labelled datasets (`classification`, `binary`, `rating`, `multiple_choice`, or ready `records`) into decision records; splits each training source by content hash into train / eval / calibration; keeps `role: heldout` sources out of training entirely; can bury states among other states; audits each source | One record format for every task. A model is over-confident on sources it has not seen, so some sources must stay unseen. The audit reports what can be answered without reading the state | `train`, `eval`, `calibration`, `heldout_eval`, `heldout_calibration`, `audit` |
| `decision.train` | Fine-tunes with the answer-slot loss, LoRA or full. As a `matrix` over `seed` it gives the soup its members | The decision training itself | `model`, `adapter` |
| `decision.soup` | Averages checkpoints, uniformly or greedily (a run joins only if the soup's held-out metric does not get worse). Takes a list of models or the runs of a matrix (`group`) | Variance reduction. Expect at most a point | `model` |
| `decision.select` | The best run of a matrix by one metric | For sweeps over the learning rate or other settings | the winner's artifacts |
| `decision.calibrate` | Fits a temperature per question type and per option-count band (2, 3-5, 6-10, 11+) on hard labels, inside bounds | The probabilities are the product, so they have to mean what they say on new tasks | `calibration` |
| `decision.eval` | Accuracy with a bootstrap interval; chance-corrected macro average over sources; NLL, ECE, Brier; confident errors; a paired comparison with another evaluation; blind and permutation probes | The gate before export | `report`, `predictions` |
| `decision.export` | Final directory: weights, tokenizer, `decision.json` (prompt style, max length, calibration) | `DecisionModel.load` reads the model the way it was trained | `model` |

Every `decision.train` stage writes a full merged checkpoint (plus the LoRA adapter when one was
used), so the next stage always starts from plain weights and a soup can average any of them.

### Which rows are used for what

| Rows | Used for |
|---|---|
| `train` | training |
| `eval` (training sources) | choosing the checkpoint during training; the "seen sources" report |
| `calibration` (training sources) | choosing the soup's members |
| `heldout_calibration` (unseen sources) | fitting the temperatures |
| `heldout_eval` (unseen sources) | the reported result and the gates |

The soup and the temperatures are fitted on different rows, and neither touches the rows the
result is reported on.

### Calibration

`decision.calibrate` writes `{"temperature": {type: T}, "bands": {type: {band: T}}}`. A question
uses its band's temperature when the band had at least `min_examples` held-out rows, otherwise
its type's. Temperatures are fitted against hard labels and kept inside
`[min_temperature, max_temperature]` (0.5 to 5): without bounds, a set the model answers without
error drives the temperature toward zero, which turns every probability into 0 or 1.
`calibration/nll_out_of_fold` is the NLL when each half of the rows is scored with temperatures
fitted on the other half.

Fit on your own data when you have it: a temperature is a statement about a kind of input.

### What `decision.eval` reports

- `eval/accuracy` with `eval/accuracy_ci_low` and `_ci_high` (bootstrap, 95 %).
- `eval/macro_skill`: accuracy corrected for chance, `(accuracy - chance) / (1 - chance)`, averaged
  over sources so that each source counts once. It is the Decision Index's correction.
- `eval/nll`, `eval/brier`, `eval/ece`, and per source in `report.json`.
- `eval/confident_errors`: the share of answers that are wrong and were given 90 % or more.
- With `compare_to` (another evaluation's `predictions` over the same records): `eval/delta_accuracy`
  and `eval/delta_nll` with paired-bootstrap intervals. `eval/delta_accuracy_ci_low > 0` means
  better beyond chance.
- **Blind probe** (`eval/blind_accuracy`, `eval/blind_gap`): the same questions with the state
  hidden. A model that reads the state falls back to the answer prior; a small gap means the
  questions give the answers away.
- **Permutation probe** (`eval/permutation_consistency`, `eval/permutation_accuracy`): every choice
  question is asked again with its options reshuffled. Consistency is the share of questions whose
  chosen option does not depend on the order. A model trained without `shuffle_options` scores low
  here (0.10 in this repository's tiny test model, trained with a fixed option order on purpose).

`DecisionModel.load(path, orders=3)` asks every choice question in three option orders and
averages, which removes what order dependence is left at three times the cost.

## Run

```bash
pip install -e ".[decision]"
stagetrain plan   configs/decision/d1-style.yaml
stagetrain run    configs/decision/d1-style.yaml
stagetrain report configs/decision/d1-style.yaml --open
stagetrain run    configs/decision/d1-style.yaml --only eval_heldout --set eval_heldout.permutations=5
```

`configs/decision/d1-style.yaml` trains Qwen3-0.6B on 14 public datasets, three seeds, and holds
three more datasets out for calibration and evaluation. The dataset ids and column names were
checked against the Hub's metadata; the config itself has not been run, and its gate thresholds
are starting points.

## Relation to d1

Liquid AI has not published d1's training recipe. Its [release post](https://www.liquid.ai/blog/open-d1)
says that d1-3B's base was a weight average of two checkpoints, that several fine-tunes with
different seeds and data mixtures were merged, and that training on long inputs, shuffling answer
options and fixing data shortcuts mattered most. The readout here (softmax over option tokens at
the answer slot) follows the mechanism in d1's published inference code, reimplemented on the
tokenizer's chat template so it works with any causal LM. The loss, the calibration and the prompt
wording are this repository's own design.

Unsloth also ships decision-model training (`FastDecisionModel`, `DecisionTrainer`). It is a
different design: a separate decision head on top of the backbone, not the LM's own logits at an
answer slot. This track does not use it; the available evidence does not favour a trained head.

Not implemented:

- **Shared-trunk inference.** d1 reads the state once and branches per question with a custom
  attention layout. Here every (state, question) pair is its own row.
- **Bidirectional attention** in part of the network, the one architectural change with a
  measured gain.
- **Images.** States are text or JSON.
- **Running the Decision Index.** `eval/macro_skill` uses its chance correction; the benchmark
  itself is not wired in.

## What the tests cover

`tests/test_decision_units.py` (no torch): question and record parsing, row conversion for every
task, soft labels, held-out sources, buried states, the audit, metrics, the temperature fit with
its bounds and option-count bands, bootstrap intervals, paired differences, chance correction,
the mixture, config validation.

`tests/test_decision_model.py` (tiny random Llama and word-level tokenizer built in the test, CPU,
about 10 s, nothing downloaded): verbalizer, prompt and truncation, left-padded batches, option
shuffling and the question-changing augmentations, the loss against a manual computation, readout
independence from padding, and the whole pipeline: data with held-out sources, training, a seed
matrix, greedy soup, calibration on the held-out sources, a paired evaluation with both probes
and gates, export, `DecisionModel` with and without order averaging. The tiny model reaches 100 %
on separable synthetic reviews: that proves the plumbing, not the quality of any real model.

Not run: any real checkpoint, multi-GPU training, `push_to_hub`, inputs long enough to need
truncation, and the example config.
