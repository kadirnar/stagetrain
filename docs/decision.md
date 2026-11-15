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
  answers or soft distributions; `label_smoothing` applies to yes/no and choice, and
  `ordinal_sigma` blurs a score's target over neighbouring levels, so that being off by one level
  costs less than being off by three.
- The second term (`vocab_mass_weight`, default 0.1) keeps the vocabulary's mass on option tokens,
  so the same model still writes a valid code when sampled from.

Train-time prompt variation, each one closing a shortcut:

| Setting | What it varies | Shortcut it removes |
|---|---|---|
| `shuffle_options` | a choice's option order, every time a record is seen | "the answer is usually B" |
| `description_dropout` | option names with or without descriptions | relying on descriptions being present |
| `mix_option_format`, `mix_state_format` | how options and states are laid out | one fixed layout |
| several `instructions` per source | the question's wording | wording as a cue for the answer |

When a prompt is longer than `max_length`, only the state is cut, from its middle. The question,
the options and the slot are always whole.

## Stages

| # | Stage | What it does | Why | Main outputs |
|---|---|---|---|---|
| 1 | `decision.build_data` | Converts labelled datasets (`classification`, `binary`, `rating`, `multiple_choice`, or ready `records`) into decision records; splits train / eval / calibration by content hash; audits each source | One record format for every task. A duplicate row always lands in one split, so nothing leaks. The audit reports what can be answered without reading the state | `train`, `eval`, `calibration`, `audit` |
| 2 | `decision.train` (align) | Short inputs, large batches, higher learning rate | The model first learns to answer at the slot | `model`, `adapter` |
| 3 | `decision.train` (full, several) | Long inputs, every prompt variation on, one run per seed and data `mixture` | The actual decision training; the runs differ so that averaging them helps | `model`, `adapter` |
| 4 | `decision.soup` | Averages the runs' weights, uniformly or greedily (a run joins only if the soup's held-out metric does not get worse) | One model that is more robust than any single run | `model` |
| 5 | `decision.calibrate` | Fits one temperature per question type on held-out records | The probabilities are the product, so they have to mean what they say | `calibration` |
| 6 | `decision.eval` | Accuracy, NLL, ECE, Brier, per source and per type; blind probe; permutation probe | The gate before export | `report`, `predictions` |
| 7 | `decision.export` | Final directory: weights, tokenizer, `decision.json` (prompt style, max length, temperatures) | `DecisionModel.load` reads the model the way it was trained | `model` |

Every `decision.train` stage writes a full merged checkpoint (plus the LoRA adapter when one was
used), so the next stage always starts from plain weights and a soup can average any of them.

### The two probes in `decision.eval`

- **Blind probe** (`eval/blind_accuracy`, `eval/blind_gap`): the same questions with the state
  hidden. A model that reads the state falls back to the answer prior; a small gap means the
  questions give the answers away.
- **Permutation probe** (`eval/permutation_consistency`, `eval/permutation_accuracy`): every choice
  question is asked again with its options reshuffled. Consistency is the share of questions whose
  chosen option does not depend on the order. A model trained without `shuffle_options` scores low
  here (0.10 in this repo's tiny test model, which was trained with a fixed option order on purpose).

## Run

```bash
pip install -e ".[decision]"
stagetrain plan configs/decision/d1-style.yaml
stagetrain run  configs/decision/d1-style.yaml
stagetrain run  configs/decision/d1-style.yaml --only eval --set eval.permutations=5
```

`configs/decision/d1-style.yaml` trains Qwen3-0.6B on eight public datasets (BoolQ, Civil Comments,
PAWS-X, AG News, XNLI, Banking77, CommonsenseQA, Yelp). The dataset ids and column names were
checked against the Hub's metadata; the config itself has not been run, and its gate thresholds are
starting points.

## Relation to d1

Liquid AI has not published d1's training recipe. Its [release post](https://www.liquid.ai/blog/open-d1)
says that d1-3B's base was a weight average of two checkpoints, that several fine-tunes with
different seeds and data mixtures were merged, and that training on long inputs, shuffling answer
options and fixing data shortcuts mattered most. This track is built around those four points
(stages 3 and 4, `max_length`, `shuffle_options`, the audit and the probes). The loss, the
calibration and the prompt wording are this repo's own design; the readout (softmax over option
tokens at the answer slot) follows the mechanism in d1's published inference code, reimplemented
here on the tokenizer's chat template so it works with any causal LM.

Unsloth also ships decision-model training (`FastDecisionModel`, `DecisionTrainer`). It is a
different design: a separate decision head on top of the backbone, not the LM's own logits at an
answer slot. This track does not use it, because the target here is the d1 mechanism.

Not implemented:

- **Shared-trunk inference.** d1 reads the state once and branches per question with a custom
  attention layout. Here every (state, question) pair is its own row.
- **Images.** States are text or JSON.
- **Averaging two different base checkpoints** before training. `decision.soup` can average any
  checkpoints of one architecture, so it can be put first in a pipeline for this.

## What the tests cover

`tests/test_decision_units.py` (no torch): question and record parsing, row conversion for every
task, the content split, the audit, metrics, the temperature fit, the mixture, config validation.

`tests/test_decision_model.py` (tiny random Llama and word-level tokenizer built in the test, CPU,
about 10 s, nothing downloaded): verbalizer, prompt and truncation, left-padded batches, option
shuffling, the loss against a manual computation, readout independence from padding, and the whole
pipeline (data, align, two soup members, greedy soup, calibrate, eval with both probes and relative
gates, export, `DecisionModel`). The tiny model reaches 100 % on separable synthetic reviews: that
proves the plumbing, not the quality of any real model.

Not run: any real checkpoint, multi-GPU training, `push_to_hub`, and inputs long enough to need
truncation at 8k tokens.
