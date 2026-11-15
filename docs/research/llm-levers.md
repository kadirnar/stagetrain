# Training research: what else raises the accuracy of a fine-tuned LLM

Date: 2026-11-15. Scope: open-weight models of roughly 0.5B to 30B parameters, LoRA or full
fine-tuning, tasks with a measurable accuracy.

**The question.** Beyond what the `llm` track already does (data checks, a learning-rate sweep,
checkpoint selection, verified synthetic data, distillation, DPO, RL), what else measurably
raises final task accuracy? Optimizer settings are in [llm-optimizer.md](llm-optimizer.md).

**Short answer.** One thing is nearly free and was added: averaging the runs of one recipe that
differ only in their seed. Three more have support and are not implemented: a data loop aimed at
the model's errors, an optimised prompt as the baseline, and constraining only the final answer
when decoding. Several popular levers have none.

## How to read this

Nothing here was run. Evidence labels: **[R]** the passage or table was read in the paper's
text, **[A]** the abstract only, **[S]** a search summary only, **[D]** library documentation.

Most positive results below come from the method's own authors with one run per cell. Runs that
differ only in their seed are 0.3 to 1.1 points apart, now and then 3, so a single-run gain
under about 2 points is not evidence.

## Worth doing

| # | Technique | Evidence | In the pipeline |
|---|---|---|---|
| 1 | **Average the updates of several seeds of one recipe.** | [R] Three seeds of LoRA on MetaMathQA, plain average of the updates against the single runs: GSM8K 64.0 to 66.6 on Llama-2-7B and 75.8 to 79.1 on Mistral-7B; MATH +0.5 and +3.1 (SeedLoRA, ICML 2025). The authors' refined fusion reaches +4.9 on both, above full fine-tuning. Same direction for random soups of checkpoints (Souper-Model [R]). No independent rerun of the seed result was found. A plain average of models that differ in data or recipe was worse than the best member [R] | `llm.soup`: the exact average of the runs' updates, written as one adapter. With `selected_by`, the seeds of the configuration `llm.select` chose |
| 2 | **A data loop aimed at the errors**: collect what the model gets wrong on training-side data, generate new examples of that kind, retrain, repeat | [R] From 100 examples on GSM8K: 23.7, against 18.1 for the same amount of untargeted paraphrase and 19.9 for the same amount of extra real data; several rounds beat one (LLM2LLM, Llama-2-7B). [R] +2 to +3 points over plain SFT at 1,000 examples (PaDA-Agent, Llama-3.2-1B). [R] +3.1 and +4.1 over untargeted synthetic data for two of four models (CRAFT). Three groups, all single runs, mostly little data. Counter-case: on one task more real data was clearly better | Partly: `llm.generate_data` writes every prompt with its pass rate (`prompts`), which is the list of what the model fails on. Generating new examples from those is not implemented |
| 3 | **An optimised prompt as the baseline, and before weight training.** | [R] Qwen3-8B, six tasks: a prompt optimiser (GEPA) +9.6 on aggregate, against +3.7 for GRPO with 24,000 rollouts; GRPO still won on the hardest maths set. [R] Three seeds: untuned 66.3, optimised prompt 70.0, GRPO 71.2, prompt then GRPO 73.4, with standard deviations as large as some of the gaps (mmGRPO). All of it is on compound programs with a few hundred examples; no comparison against SFT on thousands of gold labels was found | Not implemented. `llm.eval` takes a `system_prompt` and few-shot rows, so a prompt found elsewhere can be the baseline and can be kept through every stage |
| 4 | **Reason first, constrain last** when the output has a schema | [R] Constraining only the final answer after free reasoning helped: GSM8K 80.1 to 81.6-83.8 on Llama-3.1-8B. Forcing an answer-only schema on small models hurt: accuracy 19.7 to 11.0 while every output became valid; a tool call fell from 91.5 to 48.0. Each direction seen by two groups, none on fine-tuned models | Not implemented (no constrained decoding). The `json` scorer and `eval/no_answer` separate a wrong answer from an invalid one |
| 5 | **Score the artefact that ships.** | [R] 4-bit GGUF cost 0 to 1 point on general benchmarks for an 8B model, and 10 points for one fine-tuned 1B classifier (0.99 to 0.89 at Q4_K_M, none lost at Q5_K_M) | `llm.eval` runs on a merged checkpoint; it cannot load a GGUF file. Evaluate the 4-bit merge, or the GGUF outside the pipeline |
| 6 | **Rewriting the training questions** (paraphrases and reversed questions) on top of more answers | [R] Llama-2-7B on GSM8K: 41.6 plain, 59.6 with augmented answers, 64.4 with rephrased and backward questions added (MetaMath, 2023). The effect is large when data is scarce | Not implemented |
| 7 | **Process rules.** Three seeds for any decision under about 2 points; a validation set sized for the difference to detect; a test set that is read once | Arithmetic, not a paper: seeing a 3-point paired difference needs about 900 to 1,700 items, a 2-point one 2,000 to 4,000. Picking the best of ten near-equal candidates on 500 items inflates the winner by 1.5 to 2 points | `test_fraction` keeps a split nothing is selected on; `compare_to` gives the paired interval; `select/std` the seed spread |

Also read, single-group: a second SFT stage with the loss on the final answer only (+2 to +10 %
relative, judged by a model, 2025); training on hundreds of prompt wordings (+2 to +5 points
over unseen wordings, contradicted on robustness by a second study); scaling the adapter down at
merge time under distribution shift; training a model to abstain, which raises accuracy on the
answered share only and must be reported with the coverage.

## Popular, not supported

- **Curriculum or easy-to-hard ordering in SFT.** No ordering wins across models (2510.19099
  [R]); no robust advantage over random order (2603.27226 [A]).
- **Choosing training rows by influence or gradient similarity.** Random selection is
  competitive, several methods fall below it on larger pools, and selection can cost more than
  training on everything [R].
- **TIES, DARE or SLERP for runs of one task.** Erratic where a plain average helps [R].
- **A moving average of the weights (EMA) as an accuracy lever.** No gain on small models by
  the method's own authors [R].
- **min-p sampling, and tuning the temperature for accuracy.** No advantage at equal tuning
  effort [R]; no significant effect between 0 and 1 on untuned models [A]. For a reasoning
  model, use the vendor's documented sampling settings.
- **More votes on hard items with a small model.** Majority voting lowered per-problem accuracy
  on 57 to 66 % of hard problems for 7B to 8B models [R]. Voting saturates at the share of items
  whose most common answer is already right, which is why `llm.eval` reports the majority vote
  next to pass@k instead of assuming it helps.
- **Up-sampling the rare class** for a generative classifier: little effect [A].
- **Multi-task SFT with loosely related tasks.** Negative transfer on average [S].

No evidence found either way: maximum sequence length, truncation and packing; dropping the
system prompt during SFT; few-shot examples at inference after fine-tuning (2025-2026);
distilling an ensemble of fine-tunes into one model; a controlled comparison of QLoRA, LoRA and
full fine-tuning from 2025 or 2026.

## Open questions

- **Uniform or weighted averaging.** Uniform over seeds helps; uniform over different models
  hurts and weights have to be tuned. Whether averaging learning-rate variants or the
  checkpoints of one run behaves like seeds is untested.
- **Prompt optimisation against fine-tuning** on thousands of gold labels: not compared.
- **Constrained decoding on a model that was fine-tuned to emit the schema:** not measured.
- **Base or instruct checkpoint as the start.** Equal in one study, the instruct model clearly
  ahead on code in another; Unsloth advises the base model above about 1,000 rows.
- **Targeted synthetic data against more real data:** each wins on some tasks.

## Sources

Read in the paper's text: SeedLoRA (ICML 2025, PMLR); Souper-Model (Meta, 2025); LLM2LLM;
PaDA-Agent; CRAFT; GEPA; mmGRPO; BetterTogether; JSONSchemaBench; Constraint Tax; Let Me Speak
Freely; MetaMath; PAFT; SFTKey; RLSR (ICML 2026); 2601.14277 and 2510.21970 (quantised export);
2510.19099 (curriculum); Ivison et al. 2025 and "Chasing Random" (data selection); BEMA;
Schaeffer et al. 2025 (min-p); 2608.11403 and 2606.28661 (voting).

Abstract or summary only: 2603.27226; Renze and Guven 2024; 2506.10687; 2511.13368; SelectLLM.

Not checked: whether SeedLoRA tuned the learning rate per arm; numbers that exist only in
figures. The validation-size and selection-inflation figures are this report's own arithmetic.
