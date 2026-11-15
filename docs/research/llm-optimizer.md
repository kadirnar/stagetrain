# Training research: optimizer and optimisation hyperparameters for LLM fine-tuning

Date: 2026-11-15. Scope: open-weight models of roughly 0.5B to 30B parameters, LoRA or QLoRA
first and full fine-tuning second, one to eight GPUs.

**The question.** Which optimizer and which optimisation settings raise the final task accuracy
over AdamW with a tuned learning rate, which popular ones do not, and which settings should a
pipeline sweep?

**Short answer.** No optimizer and no Adam setting was found that reliably beats AdamW with a
tuned learning rate. Three things have support from more than one group: the learning rate, its
coupling with the effective batch size, and a decaying schedule. Almost every reported gain of a
"better LoRA optimiser" is measured against a baseline whose learning rate was ten times too low.

## How to read this

Nothing here was run. Evidence labels: **[R]** the passage or table was read in the paper's
text, **[A]** the abstract only, **[S]** a search summary or a page read through a summarising
tool, **[D]** library documentation or source.

Limits: numbers that exist only in figures could not be read. Almost every controlled result is
MetaMathQA to GSM8K, CodeFeedback to HumanEval, or commonsense suites, on Llama-2/3, Gemma and
Qwen. The search for Adafactor, learning-rate-free methods, Schedule-Free and warmup studies was
partly done by web search only, so "nothing found" is weaker for those.

## What has support

| # | Finding | Evidence | What the pipeline does |
|---|---|---|---|
| 1 | **Sweep the learning rate together with the effective batch size.** The best batch size moves with the size of the dataset, and the best learning rate is different for each batch size | [R] Batch size alone moved accuracy by more than 10 points and explains why papers on LoRA variants contradict each other (2602.09492: Llama-2-7B, rank 128, 3 seeds, the learning rate swept again for every batch size). [R] With the learning rate tuned, any batch from 16 to 128 reached the same accuracy and the ceiling fell at 256 and above (2602.04998). The papers disagree on the rule that links the two, so there is none to apply. [R] Gradient accumulation and a true batch of the same size give the same loss curve and score (2412.13337) | `matrix:` over `train.learning_rate` and `train.gradient_accumulation_steps` runs the grid; `llm.select` keeps the best. The default effective batch stays 16 |
| 2 | **Keep a decaying schedule for SFT.** | [R] A constant rate at the tuned learning rate was worse than cosine in 6 of 6 cells (2609.01244) and cost about 5 accuracy points in a second study (2602.09492; the number is in a figure). A constant rate only ties when the learning rate is below its optimum, or for full fine-tuning at very large batches (2412.13337) | `lr_scheduler_type: cosine` is the default for SFT. RL keeps its constant rate |
| 3 | **Three seeds before trusting a small difference.** | [R] Runs that differ only in their seed: standard deviation 0.2 to 0.8 points near the best learning rate (Gemma-3-1B), 0.4 to 2.4 (Llama-2-7B and 13B, 2605.10468), up to 5.2 on single tasks over ten seeds (Llama-3.2-3B, 2503.07329). Single full fine-tuning runs differed by -5.2, +7.2 and -0.8 points between two precisions with no consistent sign | `seed` is a `matrix` key; `llm.select` with `average_over: [seed]` judges a configuration by its mean and reports the spread |
| 4 | **The learning-rate range.** | [R] Tuned optima: 1e-3 from 0.6B to 32B at alpha = r/2, unstable at 3e-3 (2609.01244); 7e-4 at alpha = 2r on Llama-2-13B (2605.10468); 2e-4 at alpha = r on Gemma-3-1B (2602.04998). The best LoRA rate did not move with model size at a fixed recipe | The suggested sweep stays 5e-5 to 1e-3. Two studies found the best value at the top of their range: extend upwards when the highest value wins |

## What made no measurable difference

Once the learning rate is tuned. These stay at their defaults and are not worth sweeping.

| Setting | Evidence |
|---|---|
| **The optimizer.** Muon | [R] Three groups, one direction. Llama-2-7B, 3 seeds, the learning rate swept for each: full fine-tuning Adam 61.66 against Muon 57.37 on GSM8K; LoRA is a tie (2605.10468). On Qwen3-8B Muon was better by at most 0.012 nats, at a third of the learning rate and with a narrower usable range (2609.01244). Muon-style LoRA won only on the one model that had been pretrained with Muon (2608.14492) |
| Lion, Sophia, AdEMAMix, Schedule-Free, Prodigy, D-Adaptation, SOAP | No controlled comparison on LLM fine-tuning was found for any of them. One row for Shampoo: no better than Adam [R] |
| SGD in RL | [R] Parity with AdamW in full fine-tuning, with AdamW at one untuned learning rate and no seeds (2602.07729). Not tested with LoRA |
| 8-bit against 32-bit AdamW | No comparison on LoRA found. For an adapter the optimizer state is a fraction of a percent of the memory, so the 8-bit variant saves nothing there; it does for full fine-tuning |
| **Warmup** | [R] Removing it "maintains robust performance across all batch sizes" (2602.09492); 0, 25 and 100 steps gave the same result in full fine-tuning (2412.13337) |
| **Weight decay** on LoRA weights | [R] 0.01 "barely helps" (2310.00035); the controlled studies ran with 0 |
| **Adam's beta2, epsilon, amsgrad; the clipping threshold** | No fine-tuning ablation found. 1.0 is the clipping value the studies ran with |
| **LoRA variants and initialisations** (DoRA, PiSSA, MiLoRA, OLoRA, LoRA-GA) | [R] Ten methods within 0.84 % of each other with the learning rate tuned (2602.04998). LoRA-GA, LoRA-Pro and AGGC report gains of 11 to 15 points over a LoRA baseline run at 2e-5, where plain LoRA scores 42 on GSM8K against 57 to 60 at a higher rate in three other papers |
| **rsLoRA** | [R] At a fixed rank it is a change of alpha, which the learning-rate sweep absorbs: +1.5, +1.1 and -0.1 on three models at a fixed rate |
| **NEFTune** | [R] Benchmark accuracy unchanged by the authors' own figure, confirmed by a second group (2310.05914, 2605.23171) |
| **Large effective batches** | [R] The best reachable accuracy falls at a batch of 256 and above (2602.04998) |

## Optional, with mixed or single-group evidence

| Technique | Evidence | In the pipeline |
|---|---|---|
| **LoRA+**: the B matrices at a multiple of the learning rate | Four groups, from no gain to +4 points. No gain on Gemma-2B and 7B with a tuned learning rate (2410.20625 [R]); +1.9, +4.0 and 0 on three 7B to 8B models (2609.05885 [R]); +1.4 and +0.2 in RL, single seed (2512.23165 [R]). The authors: the best ratio "is model and task sensitive" | `loraplus_lr_ratio` on `llm.sft` and `llm.cpt` with `backend: hf`, off by default. 16 is the authors' value; lower the learning rate by about half with it |
| **LoRA dropout** | "Useful for small models (7B, 13B), but not for larger" (QLoRA, no number [R]); no gain for the baselines in another study | `lora.dropout`, 0 by default |
| **A much larger beta1 in RL** | [R] 44.3 to 46.1 at 32B over 3 seeds, apparently full fine-tuning (2610.20413); a second paper finds momentum hurts in RL | `train.adam_beta1` can be set; not a default |
| **Flat-LoRA** (random weight perturbation) | [R] +1.2 on GSM8K and +1.9 on HumanEval over LoRA at a strong learning rate, 3 runs with standard errors (2409.14396). Both papers are from one group | Not implemented: not in PEFT or transformers |
| **LoRA-RITE** | [R] +4.5 and +8 points averaged, the one optimizer paper whose baseline was tuned (2410.20625). No seeds, attention-only LoRA, no independent reproduction | Not implemented: not in PEFT |
| **Bias-corrected weight averaging** (BEMA) for full fine-tuning | [R] Gains in figures only (2508.00180). No LoRA result | Not implemented |

## What the pipeline's defaults are, and why they stay

| Default | Verdict |
|---|---|
| Learning rate 2e-4, swept over 5e-5 to 1e-3 | Keep. See finding 4 |
| Cosine decay | Keep. Finding 2 |
| Effective batch 16 (2 x 8) | Keep as the start; make it the second sweep axis |
| Warmup 5 %, weight decay 0.01 | No evidence either way. 0 for both is what the controlled studies used |
| `max_grad_norm: 1.0`, betas 0.9 and 0.999 | Keep. No ablation found |
| `optim: adamw_8bit` | Kept because it is what Unsloth's recipes use and no accuracy difference is known. For an adapter `adamw_torch` is the same optimizer without bitsandbytes; the `hf` backend already falls back to it where bitsandbytes or a GPU is missing |
| LoRA rank 32, alpha 32, all linear layers, dropout 0 | Keep. If alpha changes, the learning-rate sweep has to move with it |
| bf16 | Keep for LoRA: PEFT keeps the adapter weights in fp32 by default [D]. For full fine-tuning with weights stored in bf16, rounding biases small updates (2507.07101 [R]) |
| RL: constant 1e-5, no warmup, weight decay 0, 64 completions per update | Keep. The one controlled LoRA study of RL used this setting and saw no difference at 5e-6 or between 32 and 128 completions, single seed (2512.23165 [R]) |
| DPO | Its `beta` and learning rate are entangled, and the loss is not comparable across `beta` (2608.27032 [R]): sweep them together and select by the task metric |

New named settings on every `train:` block, for when they are needed: `adam_beta1`, `adam_beta2`,
`adam_epsilon`, `lr_scheduler_kwargs` (for example `{min_lr_rate: 0.1}` with
`cosine_with_min_lr`). `optim` passes any optimizer the installed transformers knows.

## How to search

Ranked by the size and reproducibility of the effect: (1) learning rate; (2) effective batch
size, with the learning rate chosen again for each; (3) epochs, through the checkpoint chosen by
the task metric; (4) rank and alpha, moving the learning-rate range with them; (5) optionally
the LoRA+ ratio and dropout. Warmup, weight decay, betas, epsilon and clipping: do not search.

- **How many trials.** No study settles it. The controlled papers use 4 to 9 learning rates over
  two to three decades and three seeds at the best few.
- **Random or Bayesian search.** Random search was not beaten by TPE or plain Bayesian
  optimisation over 30 trials (2602.11171 [R], seeds not stated). Nothing was found on
  successive halving for LLM fine-tuning.
- **What transfers.** The best learning rate and batch size transfer across model size within a
  family and across rank [R]. The best batch size does not transfer across dataset size. A tenth
  of the data ranks configurations well enough for a search (2602.11171 [R]). Nothing transfers
  across methods: PiSSA needs a learning rate about 10 times lower than LoRA, Muon about 3 times.

## Open questions

- How the best learning rate moves with the batch size: proportional (2602.04998), rising then
  falling (2602.09492), not at all (2412.13337, full fine-tuning).
- Whether a small batch is best or an optimum exists; the studies differ in whether they fix the
  number of samples or of steps.
- Alpha: 32 at rank 64 best in one study, 16 to 32 times the rank in another. Agreed: alpha and
  the learning rate are coupled.
- Momentum in RL: harmful (2602.07729) or helpful when far above 0.9 (2610.20413).
- 4-bit QLoRA against 16-bit LoRA: no controlled comparison from 2025 or 2026 was found. QLoRA's
  own 2023 claim of parity stands unreplicated.

## Sources

Read in the paper's text: 2602.09492 Beware of the Batch Size; 2602.04998 Learning Rate Matters
(v2); 2609.01244 Post-Training Science for Supervised Fine-Tuning; 2605.10468 and 2608.14492
(Muon in fine-tuning); 2412.13337 (full fine-tuning of small LLMs); 2410.20625 LoRA-RITE;
2402.12354 LoRA+; 2609.05885; 2512.23165 PEFT in RL; 2407.05000 LoRA-GA; 2407.18242 LoRA-Pro;
2409.14396 and 2508.19564 Flat-LoRA; 2310.00035; 2310.05914 NEFTune; 2605.23171; 2503.07329;
2602.11171; 2507.07101; 2508.00180 BEMA; 2610.02199; 2602.07729; 2610.20413; 2608.27032;
2305.14314 QLoRA; 2110.02861 8-bit optimizers.

Documentation and source: transformers `training_args.py`; PEFT `optimizers/` and
`tuners/lora/config.py`; TRL `dpo_config.py`, `grpo_config.py`; Axolotl, LLaMA-Factory and
ms-swift parameter references; Unsloth's hyperparameter guide (through a summarising fetch).

Not checked: the best batch sizes in 2602.09492 and BEMA's gains (figures only); whether
Unsloth keeps adapter weights in fp32.
