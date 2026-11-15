# Training research: reinforcement learning and post-training for LLMs

Date: 2026-11-15. Scope: open-weight models of roughly 0.5B to 30B parameters, LoRA or full
fine-tuning, one to eight GPUs, tasks whose answers a program can check.

**The question.** Which reinforcement-learning techniques raise the accuracy of a fine-tuned
model, which popular ones do not, and what should `llm.grpo` and the stages around it do?

**Short answer.** The objective is the choice that matters least: at this scale DAPO, Dr. GRPO,
BNPO, GSPO and CISPO differ by less than the noise between seeds, and no 2026 objective has
displaced them. What moves the result is which prompts RL is run on, how many prompts an update
sees, the learning rate, how plain the reward is, and what is measured afterwards. Two cheaper
routes reach most of what RL reaches: rounds of fine-tuning on the model's own verified answers,
and on-policy distillation from a teacher. The supported size of the gain from RL after SFT is
a few points, and zero is a common outcome.

## How to read this

Nothing here was run. Two surveys were made for this report. The one on objectives and settings
read passages of the papers' arXiv HTML directly; the one on methods next to RL read papers
through a summarising fetch tool. Check a figure against its source before quoting it.

Evidence labels: **[R]** the passage or table was read in the paper, **[A]** the abstract only,
**[S]** a search summary only. "Reproduced" means a second group reports the same direction in
its own setup.

Limits that apply to everything below:

- Almost all controlled results are maths (GSM8K, MATH, AIME) and some code, on Qwen, Llama,
  DeepSeek distills and OLMo. Transfer to another task is an assumption.
- Most 2026 method papers come from one group and were not reproduced.
- Results on Qwen2.5-Math deserve less weight: that model gained 21.4 points on MATH-500 from
  random rewards against 29.1 from true ones (2506.10947 [A]), and AIME24 has 30 questions.
  Gains reported for distilled 1.5B models "fall entirely within the variance range" of seeds
  and decoding settings (2504.07086 [R]).
- The literature is large: about 2,900 papers matched the search terms for the last twelve
  months, about 165 abstracts and passages of 32 papers were read. A relevant paper can have been
  missed.

## What was added to the pipeline

Ordered by how well the evidence and the gain line up.

| # | Finding | Evidence | What the pipeline does |
|---|---|---|---|
| 1 | **Run RL on prompts the model solves sometimes.** A group of completions that all score the same carries no gradient | [R] A band filter gave 28.4 to 30.1 against 26.3 for plain GRPO on Qwen2.5-3B, and filtering only one end gave 23.8 to 25.9, below plain GRPO (2504.03380, no seeds). [R] Dropping prompts with a pass rate of 0.9 or more raised the asymptote at 8B (ScaleRL, 2510.13786). [A] 2 to 6 times fewer rollouts (SPEED, 2506.09016); +0.8 to +1.7 points at equal compute at 0.6B to 1.5B (2611.13560). [R] MiMo-7B keeps solved prompts in a pool sampled 10 % of the time because removing them was unstable (2505.07608) | `llm.generate_data` writes `rl`: prompts with a pass rate above `rl_min_pass_rate` and below `rl_max_pass_rate` (0 and 0.9), plus `rl_keep_solved` (a tenth) of the solved ones; `prompts` holds every prompt with its pass rate. Caveat below |
| 2 | **Eight or more prompts per update.** | [R] Every recipe read uses 8 to 512 prompts per update: 8 x 8 completions (2610.25812, LoRA, 4B), 16 x 8 (PEFT-RLVR, 2512.23165), 32 x 16 (BNPO), 48 x 16 (ScaleRL). With one prompt, a degenerate group makes the whole step empty. Not measured directly | `llm.grpo` defaults to 64 completions per update (batch 8, accumulation 8, 8 generations), where it used 8 |
| 3 | **A constant learning rate near 1e-5 for LoRA, swept.** | [R] The most sensitive setting: at 1B, 3e-5 peaked at step 30 and diverged while 3e-6 was still climbing at step 80 (2610.04928). LoRA runs used 1e-5 (PEFT-RLVR), 4e-5 (2510.26788), 5e-5 (2610.25812); full fine-tuning 5e-7 to 2e-6. Most recipes use a constant rate | Default 1e-5, `constant`, no warmup. Sweep 5e-6 to 5e-5 with `matrix` |
| 4 | **A binary, strict reward; other terms paid only on correct answers.** | [R] A length penalty lowered AIME24 from 55 % to 50 % at 1.5B (JustRL, 2512.16649). [A] An added brevity reward gave 0.220 pass@4 against 0.581 for a gated one (2610.21404). [R] GDPO's authors recommend conditioning the easy reward on the important one (2601.05242). MiMo, rStar2-Agent and Phi-4-mini use answer-only rewards [R] | `gated: true` on a reward term pays it only where the `correct` term accepts. `multi_objective_aggregation: normalize_then_sum` is exposed for ungated terms |
| 5 | **Measure pass@k and the majority vote, not only one greedy answer.** | [A] One LoRA adapter raised single-sample accuracy on every model from 1.5B to 8B and left majority voting below the untrained base on three of four (2610.00991). [A] 82 % of a reported out-of-domain gain disappeared under a fixed answer extractor (2610.20639) | `llm.eval` with `samples_per_prompt` reports `eval/pass_at_k`, `eval/majority_vote` and `eval/sampled_pass_at_1` next to the greedy score and the paired comparison |
| 6 | **RL on prompts no earlier stage trained on.** | [R] After distillation, GRPO on a held-out half reached 78.5 on MATH against 75.7 when replaying seen prompts (2605.12483, one paper). [R] Tulu 3: prompts not used in SFT were slightly better for the preference stage | `rl_fraction` on `llm.prepare_data` sets questions aside as `chat_rl` |
| 7 | **Rounds of fine-tuning on the model's own verified answers.** | [R] At an equal sample budget, filtered SFT recovered 88 % of GRPO's gain on Qwen2.5-Math-7B and 84 % on Llama-3.2-3B (2504.11343; the percentages are derived). [R] GSM8K on Gemma2-2B: 41.9, 52.5, 56.0, 56.9, 57.6 over four rounds (2504.18116). [R] It saturates after two or three rounds, and pass@k falls with more (2412.02674, 2403.04642). Reproduced in direction | Repeat `llm.generate_data` and `llm.sft`; gate each round on `generate/pass_at_k` against the round before. `keep_per_prompt` caps the answers kept per prompt |
| 8 | **On-policy distillation from a teacher's logits.** | [R] Qwen3-8B on AIME'24: 55.0 after off-policy distillation, 67.6 with RL at 17,920 GPU hours, 74.4 with on-policy distillation at 1,800 (2505.09388). [R] KDRL at 1.5B: GRPO 54.6, distillation 56.1, both jointly 56.8 (2506.02208). Reproduced in direction by a third group. [R] A larger teacher that was not trained on the task was worse than direct GRPO in two model families (2605.12483) | `llm.distill`, on TRL's `DistillationTrainer`. The teacher must share the tokenizer |
| 9 | **A second, lenient check of the scorer before RL.** | [A] Wrong rejections hurt fastest at 0.5B to 1.5B and more rollouts do not buy the loss back (2605.25252). [A] Systematic wrong acceptances give anything from a plateau to collapse (2605.02909). [S] One rule-based checker rejected over 38 % of correct answers (2510.00915) | `audit_scorer` on `llm.generate_data` reports `generate/scorer_disagreement`: of the answers the scorer rejected, the share a lenient scorer accepts |
| 10 | **Length-normalised DPO; near-miss negatives.** | [R] The only preference loss that beat the SFT model in Tulu 3's ablation: 57.3 against 55.2 for DPO and 55.7 for the base (2411.15124). [R] At 1.5B after SFT, random pairs lowered MATH from 65.2 to 64.1 and mined pairs raised it to 67.1 (2512.19728; mining and weighting not separated) | `loss_type: sigmoid_norm` on `llm.dpo`; `negatives: nearest` on `llm.generate_data` pairs each right answer with the wrong one closest in wording |
| 11 | **Shorter correct answers as training rows, for reasoning models.** | [A] SFT on own samples that are correct and short cut reasoning length by up to 80 % at unchanged accuracy (2602.13407) | `prefer: shortest` on `llm.generate_data` |

Already in the pipeline and confirmed by this reading: `loss_type: dapo` with no KL penalty;
sampling at temperature 1.0 without top-p during RL; starting RL from a checkpoint that still
explores (`tolerance` on `llm.select_checkpoint`); the same scorer for training and evaluation;
the paired comparison against the model RL started from. RL forgets less than SFT at matched
accuracy (2509.04259 [R]), so the replay mix (`replay_ratio`) stays on the SFT stages.

**The caveat on finding 1.** The measured gain belongs to filtering during training. A filter
computed once with the starting model gave no gain in the one study that tested it (26.3 against
26.3, 2504.03380). The recipes that filter beforehand, MiMo and rStar2-Agent, measure again
between RL stages. So `rl` is a way to stop spending rollouts on prompts that cannot teach, and
the measurement goes stale: `configs/llm/post-training.yaml` runs RL in two stages and measures
before each.

## The objectives

With one update per batch of rollouts, which is the default in TRL and here, the importance ratio
is exactly 1, so the clipped objectives differ only in how they normalise.

| `llm.grpo` setting | What it changes | Evidence | Verdict |
|---|---|---|---|
| `loss_type: dapo` | Loss normalised over all tokens of the batch; no KL | [R] The cumulative ablation is one run each on 30 questions (2503.14476). Token-level loss confirmed for base models by Lite PPO (2508.08221) | The default |
| `loss_type: dr_grpo` | No division by length or by the reward's spread | [R] "Similar performance" for three normalisations at 8B (ScaleRL); within seed noise at 1B (2610.04928). [A] Per-prompt scaling ahead at 1.5B LoRA, equal at 7B (2601.23135) | Equivalent |
| `loss_type: bnpo` | TRL's version: the same normalisation over the local batch | [R] Five estimators within 1.4 points at 1.5B and 7B, by the method's own authors (2506.02864) | Equivalent |
| `loss_type: grpo` + `importance_sampling_level: sequence` (GSPO) | One importance ratio per sequence | [R] Ahead of DAPO at 8B with a clip range ten times the paper's (ScaleRL). Behind token-level GRPO at 0.5B and 1.5B, 39.6 against 58.0 (2603.19335, an early checkpoint). Shortens answers (2602.05261) | For reused rollouts and mixture-of-experts models |
| `loss_type: cispo` | Clips the weight, keeps every token's gradient | [R] ScaleRL's choice; inside its full recipe DAPO reached the same asymptote | For reused rollouts |
| `loss_type: sapo`, `vespo`, `luspo` | Soft gate; reshaped weights for stale rollouts; GSPO without the length bias | [A] Authors only. Each targets off-policy or mixture-of-experts training | Available, no reason to switch |
| `beta` (KL to the starting policy) | | [R] Stable entropy and worse accuracy (2505.22617); did not rescue a collapsed run (2606.18487); removed by DAPO, JustRL, ScaleRL, MiMo. For it: with reference resets over 2,000 steps (ProRL, 2505.24864) | 0 |
| `epsilon`, `epsilon_high` | Clip range; a wider upper bound | [R] 0.2 / 0.28 in DAPO, JustRL, PEFT-RLVR; 0.32 best at 4B (Lite PPO). Inert with one update per batch | Set them together with reuse |
| `steps_per_generation`, `num_iterations` | Several updates per batch of rollouts | [R] 4 in JustRL and ProRL, 16 in DAPO. With rollouts 32 updates old a clipped ratio lost nothing (2610.04928). [A] Speedups of 2 to 3 times, authors only | Optional |
| `scale_rewards` | Spread taken per group, per batch, or not at all | See `dr_grpo` | The library's default |
| `cast_lm_head_to_fp32` | Output layer in full precision | [R] Asymptote 0.52 to 0.61 at 8B on a DAPO baseline; little difference inside the full recipe (ScaleRL) | Off; for long runs |
| `mask_truncated_completions` | Drops cut-off completions from the loss | [R] +6 points in DAPO's table; restricts small models according to Lite PPO | Only when truncation is rare |

## What the evidence does not support

- **Switching the objective** for an on-policy LoRA run. See the table above.
- **A KL penalty or a fixed entropy bonus against collapse.** The entropy bonus is "highly
  sensitive to coefficients" and never beat the baselines (2505.22617 [R]).
- **Length penalties and soft punishment of long completions** (JustRL [R], 2610.21404 [A]).
- **Resampling until a batch is full of mixed groups.** "The best overall results are achieved
  with DAPO when DS is disabled" (2512.07611 [R]). Dropping degenerate groups is the part that
  helps.
- **Dense, step-level or graded rewards as a default.** At equal compute none of four dense
  variants beat plain GRPO across 1,668 runs (2611.13560 [A]); pass-rate rewards for code "did
  not reliably outperform binary rewards" (2605.02944 [A]). One exception at 0.5B (2607.02869 [A]).
- **An LLM judge as the reward where a checker exists.** Judges accept outputs such as a lone
  ":" (2507.08794 [A]); with judges as the reward, acceptance rose while held-out accuracy did
  not (2610.15778 [A]).
- **Updating only the highest-entropy tokens** below 14B: parity at 8B by the authors' own
  numbers (2506.01939 [A]).
- **RLOO, ReMax or REINFORCE++ as separate trainers.** The same estimator up to a constant, or
  no measured gain (BNPO's table [R]).
- **A learned critic** at this scale: unreproduced, and a second network to train.
- **Entropy-minimisation and confidence rewards in place of labels.** They match RL early and
  then fall below the starting model (2506.17219 [A]).
- **More than three rounds of self-training,** or self-training filtered by the model's own
  judgement instead of an external check (2412.02674 [R], 2606.28438 [A]).
- **SimPO, KTO, IPO or ORPO as an upgrade over DPO.** SimPO was the worst method at 3B and
  below (2603.19335 [R]) and below the SFT model in Tulu 3 [R].
- **Distilling from a larger model that was never trained on the task** when direct RL is
  possible (2605.12483 [R]).
- **RL after a well-tuned SFT below about 1B.** Distilled SFT reached 74.0 on GSM8K where RL
  alone reached 57.0, and RL on top added nothing (MobileLLM-R1, 2509.24945 [R]).
- **Expecting new capability from a few hundred steps.** Pass@1 rose from 58.0 to 61.3 while
  pass@256 was already 94.5 (2611.02574 [A]): RL of this length mostly moves probability toward
  answers the model could already sample. After SFT on gold answers, 60 GRPO steps added 0.22
  points, with an interval that includes zero (2611.07639 [A]).

**A correction to [llm.md](llm.md).** Its finding 9, "DPO at 7B and above", rests on one study
(2603.19335). On a closer reading that study ran RL only at 0.5B and 1.5B, its 7B tier is single
LoRA runs, its 3B tier scores 10.7 % or 62.9 % depending on the answer extractor, and the
preference methods appear to start from the instruct model, so it compares DPO with SFT and not
DPO after SFT. Its usable result is GRPO 58.0 against SFT 54.4 at 1.5B over three seeds. Treat
the 7B rule as unverified. Independent support for a preference stage at that size comes from
Tulu 3 (+4.1 points average at 8B) and Olmo 3 (AIME up, MATH and IFEval down at the same stage).

## Considered and not implemented

| Technique | Why not now |
|---|---|
| Filtering prompts during training (the variant with the measured gain), and leaving groups with identical rewards out of the loss | Needs a subclass of TRL's trainer, which Unsloth also patches. The staged measurement is the substitute |
| Majority-vote reward for prompts without references | [A] It tracks the true reward when the model's majority is usually right (2610.20898) and collapses onto the majority answer otherwise (2611.01307). Needs a reward that sees the whole group |
| Replay of earlier correct answers during RL | [A] +1.7 to +7.6 points, each from its authors only. TRL removed its replay trainer |
| A baseline for small groups (fixed reference, normalising by the number of successes) | [A] Large at 4 generations, small at 8 (2605.07689). Use 8 or more generations instead |
| Half precision as FP16 instead of BF16 | [R] BF16 LoRA collapsed near step 600 in one study (2510.26788). Not reproduced; needs loss scaling |
| Distillation and RL in one loss | [R] +0.7 over distillation alone (KDRL). Two stages do nearly the same |
| Averaging an early and a late SFT checkpoint before RL | [A] Restores pass@k (2504.10478). Thin evidence |

## Settings the papers used

| Paper | Model | Adapter | Learning rate | Generations x prompts | Other |
|---|---|---|---|---|---|
| DAPO 2503.14476 | Qwen2.5-32B | full | 1e-6 constant | 16 x 512 | clip 0.2 / 0.28, no KL |
| ScaleRL 2510.13786 | 8B | full | 5e-7 constant | 16 x 48 | CISPO, reward +1 / -1 |
| JustRL 2512.16649 | 1.5B | full | 1e-6 constant | 8, batch 256 in 4 updates | no KL, no entropy term, temperature 1.0 |
| PEFT-RLVR 2512.23165 | 1.5B, 7B | LoRA r 32, alpha 64, all linear | 1e-5 constant, no warmup | 8, 128 or 32 completions | TRL, DAPO, no KL, binary reward |
| 2610.25812 | Qwen3-4B | LoRA r 64, all linear | 5e-5 cosine | 8 x 8, 300 steps | TRL, `dapo`, beta 0 |
| 2510.26788 | Qwen2.5-Math-1.5B | LoRA r 32, alpha 64 | 4e-5 | | FP16 stable, BF16 not |
| MiMo 2505.07608 | 7B | full | 1e-6 | batch 512 | no KL, rule-based reward |
| Phi-4-mini-reasoning 2504.21233 | 3.8B | full | 5e-7 | | reward +1 / -1, temperature 1.0 falling to 0.6 |

For LoRA in RL the evidence disagrees on the cost: equal to full fine-tuning even at rank 1
(Thinking Machines [S]), 2.4 points below it (PEFT-RLVR, one seed [R]). Common ground: all linear
layers, rank 16 to 64, no SVD-based initialisation (PiSSA and MiLoRA collapsed in RL).

## Recipes of published models

| Model | Order | Per-stage effect |
|---|---|---|
| Phi-4-mini-reasoning, 3.8B [R] | teacher rollouts by SFT twice, DPO on right against wrong rollouts, RL | AIME24 10.0, 30.0, 43.3, 50.0, 57.5 |
| Tulu 3, 8B [R] | SFT, length-normalised DPO, RL | average 60.6, 64.7, 65.1 |
| Olmo 3 Think, 7B [R] | SFT, DPO on strong-against-weak pairs, RL | AIME 2024 69.6, 74.6, 71.6; MATH 94.4, 92.4, 95.1 |
| MiMo-7B [R] | SFT, RL on problems with the easy ones removed | AIME24 58.7, 68.2 |
| Qwen3 small models [R] | off-policy distillation, on-policy distillation | a tenth of the GPU hours of the RL route, and better |
| MobileLLM-R1, under 1B [R] | SFT twice, no RL | RL added nothing |

The pattern: teacher data by SFT first, on-policy distillation where a teacher's logits are
available, a preference stage at about 4B and above, RL last with an answer-only reward on
filtered prompts. In every recipe with a table the largest step comes from distillation and the
later stages add single digits. `configs/llm/post-training.yaml` follows this order.

## Open questions

Papers disagree, and the pipeline only offers a setting:

- **Filtered SFT against GRPO.** GRPO ahead by 8 to 11 points on one task with LoRA at 4B
  (2610.25812 [R]); within 4 points at equal samples (2504.11343 [R]); equal once the rollout
  budget is matched (2611.14912 [A]).
- **Iterated DPO against RL.** Equal at a fraction of the cost (2503.12854 [R]); 7.5 points
  behind (2504.11343 [R]).
- **A DPO stage before RL.** Gains in Phi-4-mini and Olmo 3; Tulu 3 reaches the same reward from
  either start.
- **Token-level or sequence-level loss for models that were already fine-tuned.** Lite PPO finds
  sequence-level better for aligned models, which is this pipeline's case; DAPO and Dr. GRPO call
  it length-biased. Both are settings (`dapo`, `grpo`).
- **Medium difficulty or harder prompts.** Most studies favour a pass rate near one half; two
  find a slightly harder band better or the hard prompts the ones that add capability
  (2610.13669, 2611.04061 [A]).
- **Is one update per batch of rollouts stable?** It collapses without a trust region in one
  study (2611.15404 [R]) and is routine in others (2610.25812 [R]).
- **Which scorer error is worse,** wrong rejections (2605.25252) or wrong acceptances
  (2605.02909), and how much random noise is tolerable: 20 % cost under a point at 8B
  (2604.07666 [A]).
- **Does RL add capability or sharpen?** Short runs sharpen (2611.02574 [A]); with 2,000 and
  more steps on mixed tasks one group reports problems solved that the base model never solves
  at any number of samples (ProRL [A]).

Not found at all: an equal wall-clock comparison of filtered SFT, iterated DPO and GRPO; the cost
of RL relative to SFT; an ablation of one against several kept answers per prompt.

## Sources

Read in the paper's text, by arXiv identifier:

- Objectives: 2503.14476 DAPO; 2503.20783 Dr. GRPO; 2507.18071 GSPO; 2506.02864 BNPO; 2506.13585
  CISPO; 2511.20347 SAPO; 2602.05261 LUSPO; 2601.05242 GDPO
- Comparisons and recipes for RL: 2510.13786 ScaleRL; 2508.08221 Lite PPO; 2512.16649 JustRL;
  2505.24864 ProRL; 2512.23165 PEFT-RLVR; 2603.19335; 2512.07611; 2610.04928; 2610.25812;
  2611.15404; 2505.22617; 2510.26788; 2504.07086
- Prompt selection: 2504.03380; 2506.09016 SPEED; 2505.07608 MiMo; 2508.20722 rStar2-Agent
- Self-training and preference methods: 2504.11343; 2403.04642; 2504.18116; 2412.02674;
  2503.12854; 2411.15124 Tulu 3; 2512.19728
- Distillation: 2505.09388 Qwen3; 2605.12483; 2506.02208 KDRL; 2504.21233 Phi-4-mini-reasoning;
  2509.24945 MobileLLM-R1
- Forgetting: 2509.04259

Abstract only: 2610.00991; 2610.20639; 2611.13560; 2610.21404; 2605.25252; 2605.02909;
2604.07666; 2602.13407; 2606.18487; 2611.02574; 2611.07639; 2506.10947; 2610.20898; 2611.01307;
2605.07689; 2504.10478; 2506.17219; 2507.08794; 2610.15778; 2506.01939; 2605.02944; 2607.02869;
2601.23135; 2606.28438; 2610.13669; 2611.04061; 2611.14912

Other:

- Thinking Machines, "LoRA Without Regret": https://thinkingmachines.ai/blog/lora/ and
  "On-Policy Distillation": https://thinkingmachines.ai/blog/on-policy-distillation/
- Olmo 3 model card: https://huggingface.co/allenai/Olmo-3-7B-Think
- TRL documentation: https://huggingface.co/docs/trl/main/en/grpo_trainer, and the source of
  `trl 1.18.0`
