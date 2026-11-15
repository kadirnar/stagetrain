# llm track: multi-stage fine-tuning with Unsloth

A causal language model is adapted in stages, each one a registered stage kind that starts from a
base model or from the `model` artifact of an earlier stage:

```
prepare_data ─┬─> cpt ──> sft (broad) ──> sft (domain) ──> dpo ──> eval ──> export
              └─> eval (base model: the bar the gates compare against)
```

Any stage can be left out, and `llm.sft` can appear as many times as needed. Two example
pipelines ship in `configs/llm/`: `qwen3-multistage.yaml` (every stage, with gates) and
`sft-only.yaml` (the shortest useful one).

## Stages

| Kind | What it does | Why it is a stage of its own | Reads | Writes (artifacts) | Typical gate |
|---|---|---|---|---|---|
| `llm.prepare_data` | Normalises chat sources (`messages`, `sharegpt`, `alpaca`, `prompt_completion`), raw text and preference pairs; drops invalid rows and duplicates; filters by length; splits train/eval by content hash. | Every later stage reads the same clean files, and the held-out rows stay held out for all of them. No model is loaded. | `DataSource`s | `chat_train`, `chat_eval`, `text_train`, `text_eval`, `preference_train`, `preference_eval`, `stats` | `chat/train_rows` min |
| `llm.cpt` | Continued pretraining on raw text. LoRA (rank 128, rsLoRA) plus fully trained `embed_tokens` and `lm_head`, which get their own, 10x lower learning rate. | Domain vocabulary and facts come from unlabelled text, before any instruction data is spent. | `text_train` | `model` (adapter), `log` | `eval/loss` max |
| `llm.sft` | Chat fine-tuning. Conversations are rendered with the tokenizer's chat template; the loss covers assistant turns only (`train_on_responses_only`). | Broad instruction data first, then the small curated domain set at a lower learning rate, so the second cannot be drowned by the first. | `chat_train`, an earlier `model` | `model`, `log`, `base` (when merged) | `eval/loss` not above the base model's |
| `llm.dpo` | Direct preference optimisation on prompt / chosen / rejected (TRL `DPOTrainer`). | Preferences (tone, refusals, format) are learnt after the model already answers correctly. | `preference_train`, an earlier `model` | `model`, `log`, `base` | `eval/loss` max |
| `llm.eval` | Held-out loss and perplexity with the same masking as `llm.sft`, plus greedy generations compared with the references (`eval/exact_match`, `eval/contains`). Run it on the base model too. | Gates need a number measured the same way for every model in the chain. | `chat_eval`, a `model` | `samples`, `report` | relative to `eval_base` and to the previous stage |
| `llm.export` | Saves `lora`, `merged_16bit`, `merged_4bit` and/or `gguf` (any llama.cpp quantisation). Pushes to the Hub only when `push_to_hub` names a repository; repositories are private unless `private: false`. | Export is slow and large; it should run once, after the gates passed. | a `model` | one artifact per format, `model` | - |

Metrics follow one naming scheme (`train/loss`, `eval/loss`, `eval/perplexity`, `eval/exact_match`,
`chat/train_rows`, ...), so a gate can name any of them, alone (`min`, `max`) or against an earlier
stage (`baseline: eval_base.eval/loss`, `max_rise: 0.0`).

## How stages chain

`init_from` is a Hub id or `${stage.model}`. When it is a LoRA adapter, `on_adapter` decides:

- `continue` (default for `cpt` and `sft`): the adapter is loaded on its base model and trained
  further. This is what Unsloth documents for resuming from a saved adapter. The adapter keeps
  the shape the earlier stage gave it, so this stage's `lora` block is not used. If the adapter
  carries trained embeddings (it came from `llm.cpt`), they keep training, at a tenth of the
  learning rate unless `embedding_learning_rate` says otherwise.
- `merge` (default for `dpo`): the adapter is folded into a 16-bit copy of the base weights
  (`<stage>/merged_base`, several GB) and a fresh LoRA is attached. Use it to change rank or
  target modules, to freeze what the earlier stage learned (the example does this between `cpt`
  and `sft_broad`), and for DPO, where the policy the loss is measured against must be exactly the
  model the stage started from.

`full_finetuning: true` (with `load_in: 16bit`) trains every weight instead of a LoRA; an adapter
it starts from is always merged first.

A training stage that is interrupted continues from its last checkpoint when the pipeline is run
again (`resume: true`); a finished stage is skipped; a stage whose parameters changed reruns
together with everything after it.

## Data formats

| `format` | Columns | Notes |
|---|---|---|
| `messages` | `messages`: `[{role, content}]` | roles `system`, `user`, `assistant`, `tool` |
| `sharegpt` | `conversations`: `[{from, value}]` | `human`/`gpt`/`system`/`observation` are mapped |
| `alpaca` | `instruction`, `input`, `output` | `input` is appended to the instruction |
| `prompt_completion` | `prompt`, `completion` | |
| text sources | `text` | `chunk_chars` cuts long documents at paragraph breaks |
| preference sources | `prompt`, `chosen`, `rejected` | strings or message lists; with no `prompt`, `chosen` and `rejected` are whole conversations |

Map other column names with `columns` on the source. A conversation is kept only if it starts with
a user turn (after an optional system message), alternates, and ends with the assistant; the
reasons for every dropped row are counted in `stats.json`.

## Running

Train in an environment of its own: `unsloth 2026.11.4` pins `transformers<=5.17`, `trl<=1.15`,
`datasets<5` and `torch<2.15`, which is older than what the other tracks are developed on.

```bash
pip install -e ".[llm]"                                   # on the GPU machine
stagetrain kinds llm --params                              # every stage and parameter
stagetrain plan configs/llm/qwen3-multistage.yaml          # validate, show what would run
stagetrain run  configs/llm/qwen3-multistage.yaml          # run; rerun to resume
stagetrain run  configs/llm/qwen3-multistage.yaml --from sft_domain \
    --set sft_domain.train.learning_rate=5e-5              # change one stage, rerun from it
stagetrain status configs/llm/qwen3-multistage.yaml        # states and metrics
```

## Code layout

- `data.py`: row normalisation, chunking, the content-hash split. No model or tokenizer.
- `backend.py`: every Unsloth / TRL call, in `UnslothBackend`, and the plain functions that turn
  stage parameters into library keyword arguments. Stages only talk to `get_backend()`, so tests
  swap in a fake with `set_backend`.
- `prepare.py`, `train.py`, `evaluate.py`, `export.py`: the stages.

## What was verified, and how

Nothing in this track was run against a model: it was written on a machine without a CUDA device,
and no weights were downloaded.

Run here (47 tests, `pytest tests/test_llm_*.py`):

- data normalisation, rejection reasons, chunking, the split's stability;
- parameters -> keyword arguments, and that `sft_config_kwargs` / `dpo_config_kwargs` build real
  `trl.SFTConfig` / `trl.DPOConfig` objects on the installed trl 1.18 / transformers 5.23;
- the stages through `Pipeline` with a fake backend: artifacts, metrics, `continue` vs `merge`,
  full fine-tuning, resume/skip, gates, export order, and that both shipped configs validate.

Verified by reading the source of `unsloth 2026.11.4` and `unsloth_zoo 2026.11.4` only, never by
running it:

- `FastLanguageModel.from_pretrained` parameters, and that an adapter directory loads its base
  model with the adapter trainable (`PeftModel.from_pretrained(..., is_trainable=True)`);
- `FastLanguageModel.get_peft_model` parameters; a string `target_modules` is passed through;
  `embed_tokens` / `lm_head` belong in `modules_to_save`;
- `UnslothTrainingArguments(embedding_learning_rate=...)` and `UnslothTrainer.create_optimizer`;
- `train_on_responses_only(trainer, instruction_part=None, response_part=None, ...)` detects the
  markers from the chat template when both are `None`;
- `save_pretrained_merged`, `save_pretrained_gguf` (GGUF files land in `<dir>_gguf`; the call
  returns `gguf_directory` and `gguf_files`), `push_to_hub_merged`, `push_to_hub_gguf`;
- `PatchDPOTrainer()` is a no-op in this release; it is still called for older ones.

Not verified at all, to check on the first real run:

- `fit_config_kwargs` renames arguments for the older transformers / trl that Unsloth pins
  (`warmup_ratio`, `group_by_length`, `max_seq_length`); it was tested against stand-in
  dataclasses, not against those releases.
- TRL's conversational preference format (`prompt` / `chosen` / `rejected` as message lists) was
  read in trl 1.18; Unsloth pins trl <= 1.15.
- `llm.eval` measures loss through an `UnslothTrainer` that is never trained (TRL requires a train
  set, so it is handed the eval rows); on a 4-bit base model this path is untested.
- Exporting several formats from one loaded model (`lora`, `merged_16bit`, `gguf`, then
  `merged_4bit` last) assumes the 16-bit merge and the GGUF conversion leave the loaded weights
  untouched.
- `on_adapter: continue` for `llm.dpo` relies on TRL keeping a frozen copy of the starting adapter
  as the reference; `merge` (the default) does not depend on it.
