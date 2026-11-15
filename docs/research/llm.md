# Training research: LLM fine-tuning for the highest task accuracy

Date: 2026-11-15. Scope: open-weight models of roughly 0.5B to 30B parameters, fine-tuned with
LoRA / QLoRA or fully, on one or a few GPUs, with Unsloth.

**The question.** Which training steps raise the task accuracy of a fine-tuned LLM, in what
order, and which popular steps do not? The answer decides what the `llm` track's pipeline
contains.

**Short answer.** Most of the gain comes from four cheap things, in this order: measuring
correctly, sweeping the learning rate, choosing the checkpoint by the task metric, and adding
verified training examples. Post-training (preference optimisation, reinforcement learning) adds
a few points on top when its preconditions hold. Choosing a fancier adapter adds nothing once the
learning rate is tuned.

## How to read this

Nothing here was run: this is a reading of papers, documentation and source code. Three limits
apply to every number below.

- Nearly all controlled results come from Llama and Qwen models on maths, code and instruction
  benchmarks. Transfer to another task is an assumption.
- Papers were read through a summarising fetch tool. Check a figure against the paper before
  quoting it elsewhere.
- Several magnitudes rest on a single study. They are marked.

Evidence labels: **[C]** a controlled comparison in a paper, **[P]** practitioner consensus or
vendor documentation with no controlled test found, **[A]** one uncontrolled or unreplicated
result.

## Findings, by expected gain per unit of cost

| # | Intervention | Evidence | What the pipeline does |
|---|---|---|---|
| 1 | **Trustworthy evaluation first.** Splits by question, a task metric, the same harness for the base model and a prompt-only baseline, paired statistics, 500 to 1,000 items | [C] Format and extraction artefacts of 4 to 50 points are documented; a 3B base model scored 62.9 % with flexible extraction and 10.7 % with strict matching (2603.19335) | `llm.prepare_data` splits by question into train / validation / test. `llm.eval` reports a scorer's mean with its standard error, compares item by item against another evaluation (`compare_to`), and runs few-shot and system-prompt baselines |
| 2 | **Sweep the learning rate.** 4 to 6 values across 5e-5 to 1e-3 for LoRA; full fine-tuning 10 to 30 times lower | [C] The most sensitive setting. A wrong value costs about 10 points; the optimum moves with model, rank and batch size (2602.04998, 2609.01244) | `matrix:` on a stage runs it once per value; `llm.select` keeps the best run |
| 3 | **Free configuration choices.** All linear layers, rank 32 to 64, loss on assistant tokens only, cosine decay, no truncated targets, the exact chat template | [C] Attention-only adapters and ranks 8 to 16 are measurably worse; cosine beats a constant rate (2609.01244, Thinking Machines "LoRA Without Regret") | These are the defaults. Rows longer than `max_seq_length` are dropped instead of cut |
| 4 | **Data quality, then more verified examples.** Consistent targets first; then fresh examples, including teacher-generated ones that pass a check | [C] Tripling fresh examples from 5k to 15k lowered validation loss by a median 14 % (2609.01244). Synthetic task data from a stronger teacher is among the most reliable sources (2410.18588, 2508.04063) | `llm.generate_data` samples answers from any model and keeps the ones the scorer accepts |
| 5 | **Choose the checkpoint by the task metric,** over 2 to 3 epochs or more | [C] Validation loss bottoms out near epoch 2 and rises afterwards while judged quality holds or improves (2609.01244, 2602.11149) | `llm.sft` keeps a checkpoint per epoch; `llm.select_checkpoint` answers the validation questions with each and keeps the best |
| 6 | **A larger or better base model,** when inference cost allows | [C] In the one controlled comparison, scale mattered about five times more than the training method (2603.19335) | Not a pipeline step: `vars.base_model` |
| 7 | **Reasoning traces that were generated blind and checked.** For students of 3B or less, mix long and short traces about 1:4 | [C] Traces written toward a shown answer pass a correctness filter and still cost about 16 points on MATH-500 (2607.14552); the 1:4 mix gave Qwen2.5-3B 7 to 8 points (2502.12143) | `llm.generate_data` never shows the reference to the model; `require_reasoning` keeps only answers with a valid reasoning block |
| 8 | **Mix in 1 to 10 % general data** when general ability must survive | [C] 1 % prevented forgetting, measured for full fine-tuning only (2502.06042) | `replay_ratio` on a chat source |
| 9 | **Post-training after SFT.** GRPO-family RL when answers can be checked and the model already solves part of the prompts; DPO at 7B and above | [C], single study: at 1.5B, GRPO-style +3.6 points over SFT while DPO was 5 points below it; at 7B, DPO +7.5 and SimPO +9.4 over SFT (2603.19335) | `llm.grpo`, `llm.dpo`; `llm.generate_data` reports whether the precondition holds and builds the DPO pairs |
| 10 | **Several seeds,** then averaging or ensembling | [C]/[A] Typically 1 to 2 points for N times the cost. Seeds are needed anyway to trust a difference under about 2 points (SFT at 1.5B varied by 0.6 points, DPO at 3B by 2.0) | `seed` is a `matrix` key. Adapter averaging is not implemented |

## What the evidence does not support

These are common and the pipeline leaves them out or off by default.

- **DoRA, PiSSA, MiLoRA instead of LoRA, for accuracy.** Within 1 to 2 % once each method's
  learning rate is tuned; three independent 2026 studies (2602.04998, 2602.09492, 2606.10428).
  For rsLoRA no independent accuracy evidence was found at all.
- **A raw-text continued-pretraining stage before SFT, as a default.** It adds knowledge and
  damages prompting ability (AdaptLLM, 2309.09530); diverse synthetic rewrites of the corpus
  learn far more (2409.07431, 2508.06178). `llm.cpt` stays available and is not in the accuracy
  recipe. No direct "CPT then SFT" against "SFT only" comparison was found.
- **Early stopping on validation loss.** See finding 5.
- **NEFTune for task accuracy.** Its gains are on judged chat quality; benchmark accuracy is
  unchanged by the authors' own account (2310.05914).
- **DPO variants.** None of 20 beat vanilla DPO after correction for multiple comparisons
  (2603.19335). And at 1.5B every preference method scored below plain SFT.
- **TIES / DARE merging to raise single-task accuracy.** Only plain task arithmetic reliably
  improved anything across four model families (2511.21437).
- **One universal alpha rule.** `alpha = 2r`, `alpha = 32` fixed and very large alphas each win
  in one controlled study. What is consistent: alpha and learning rate are coupled, so fix alpha
  and sweep the learning rate.
- **Label smoothing, LoRA dropout, weight decay and warmup as accuracy levers.** No controlled
  evidence of an effect after 2023.
- **Starting RL from the SFT checkpoint with the best pass@1.** Deeper SFT raised pass@1 and
  lowered what GRPO later reached, from 0.806 to 0.481 pass@10, because the policy stopped
  exploring (2606.18487). `llm.select_checkpoint` has a `tolerance` for this.
- **Training a reasoning model on answer-only targets and judging it by answer accuracy.** The
  share of valid reasoning fell to 0 % while accuracy rose (2605.21127). `llm.eval` reports
  `eval/valid_reasoning`.

## Post-training in more detail

**Preference optimisation (DPO).** Needs pairs of a better and a worse answer to the same prompt.
Self-generated pairs (a right and a wrong answer of the model itself) are what the controlled
study used. Known failure: the loss falls while the probability of the preferred answer also
falls (2410.08847). Use it at 7B and above, when SFT on gold answers has stopped helping.

**Reinforcement learning with verifiable rewards.** GRPO samples several completions per prompt,
scores them, and raises the probability of the better ones. Variants are settings of one trainer:

| Setting | What it is |
|---|---|
| `loss_type: dapo` | Token-level normalisation that removes GRPO's bias toward short completions. Default in TRL and Unsloth |
| `loss_type: dr_grpo` | Removes the length bias another way; also turns reward scaling off |
| `loss_type: bnpo`, `grpo` | Unsloth's previous default, and the original objective |
| `loss_type: cispo` | Clipped importance sampling; reported best at large scale in one study, collapsed in another |
| `importance_sampling_level: sequence` | GSPO: one importance ratio per sequence. More stable, especially for mixture-of-experts models |

What it needs and where it fails:

- An automatic check of the answer, and a model that solves a prompt sometimes but not always.
  When every completion of a group gets the same reward there is no signal. RL mostly makes the
  model's existing correct paths more likely; it rarely adds new ones (2504.13837). So distil
  first, then apply RL.
- Rewards: keep correctness dominant, format terms small. A model can collect format reward
  without solving anything.
- Compute: every step generates `num_generations` completions per prompt. No measurement of the
  cost relative to SFT was found; it is many times higher per example.
- Reported gains are often measurement artefacts (baselines evaluated with a different template
  or extractor). Hence the paired evaluation against the model RL started from.

## What Unsloth 2026.11.4 provides

Read from the release's source and documentation, not from a run.

- **RL.** `import unsloth` patches TRL on import; `trl.GRPOTrainer` then works unchanged.
  Generation with vLLM (`fast_inference=True`) is optional and LoRA-only. `max_prompt_length` no
  longer exists, so over-long prompts must be filtered beforehand (`llm.grpo` does).
- **Defaults are rewritten.** Under Unsloth an unspecified TRL config has learning rate 5e-5,
  batch size 4, and other values that differ from TRL's own. The pipeline passes every value
  explicitly.
- **Modes.** 4-bit QLoRA, 16-bit LoRA, 8-bit, full fine-tuning, FP8, QAT, DoRA, rsLoRA, LoftQ.
  Unsloth's documentation recommends 16-bit LoRA when maximum accuracy is the goal.
- **Multi-GPU.** DDP works under `torchrun` or `accelerate launch`. This repository's pipeline
  is a single process and does not launch it.
- **Version pins.** `transformers<=5.17`, `trl<=1.15`, `datasets<5`, `torch<2.15`. The fields of
  `GRPOConfig` and `DPOConfig` are the same in trl 1.15 and 1.18, against which this
  repository's keyword arguments were checked.

## Open questions

No controlled evidence was found for any of these, so the pipeline takes no position beyond a
default:

- the effect of maximum sequence length, and of sequence packing, on accuracy
- 4-bit QLoRA against 16-bit LoRA in 2025 or 2026
- the right replay ratio for LoRA, and the right mix of reasoning and non-reasoning rows
- how many sweep trials are enough, and how large seed variance is for LoRA
- EMA, SWA and soups compared for LoRA fine-tunes
- post-training against SFT outside maths, in a second model family, or above 7B

Two controlled results disagree: 2609.01244 finds fresh examples better than repeated epochs,
2602.11149 finds 128 epochs on 400 samples better than one epoch on 51,200 for long reasoning
traces. They measure different things (validation loss, sampled accuracy). Both agree that the
number of epochs must be chosen on the task metric.

## Sources

Papers, by arXiv identifier:

- 2602.04998 Learning Rate Matters: Vanilla LoRA May Suffice for LLM Fine-tuning
- 2602.09492 Beware of the Batch Size: Hyperparameter Bias in Evaluating LoRA
- 2606.10428 Which LoRA?
- 2609.01244 Post-Training Science for Supervised Fine-Tuning
- 2602.06204 Learning Rate Scaling across LoRA Ranks and Transfer to Full Finetuning
- 2405.09673 LoRA Learns Less and Forgets Less
- 2603.19335 Do Post-Training Algorithms Actually Differ?
- 2602.11149 Data Repetition Beats Data Scaling in Long-CoT Supervised Fine-Tuning
- 2607.14552 Answer-Conditioned Chains of Thought Degrade Verifiable-Reasoning Distillation
- 2502.12143 Small Models Struggle to Learn from Strong Reasoners
- 2605.21127 Reasoning-Trace Collapse
- 2606.18487 (SFT overtraining and entropy collapse before RL)
- 2504.13837 (what RL with verifiable rewards adds over the base model)
- 2503.14476 DAPO; 2503.20783 Dr. GRPO; 2507.18071 GSPO
- 2410.08847 (likelihood displacement in DPO)
- 2309.09530 AdaptLLM; 2409.07431 Synthetic continued pretraining; 2508.06178
- 2502.06042 (replay of pretraining data); 2509.04259 RL's Razor
- 2511.21437 A Systematic Study of In-the-Wild Model Merging for LLMs
- 2411.00640 Adding Error Bars to Evals
- 2410.18588, 2508.04063 (synthetic data from a teacher)
- 2310.05914 NEFTune

Other:

- Thinking Machines, "LoRA Without Regret": https://thinkingmachines.ai/blog/lora/
- Unsloth documentation: https://unsloth.ai/docs (LoRA hyperparameters guide, reinforcement
  learning guide, multi-GPU training)
- TRL GRPO trainer documentation: https://huggingface.co/docs/trl/main/en/grpo_trainer
- The source of `unsloth 2026.11.4`, `unsloth_zoo 2026.11.4` and `trl 1.18.0`
