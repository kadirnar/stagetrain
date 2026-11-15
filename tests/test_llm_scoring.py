"""Scorers, the statistics behind an evaluation, GRPO rewards and the GRPO configuration."""

import pytest

from stagetrain.tracks.llm.backend import (
    GrpoTrainArgs,
    LoadSpec,
    fit_config_kwargs,
    grpo_config_kwargs,
    load_kwargs,
)
from stagetrain.tracks.llm.rewards import RewardSpec, build_rewards, completion_text
from stagetrain.tracks.llm.scoring import (
    ScorerSpec,
    build_scorer,
    choice_letter,
    has_valid_reasoning,
    last_number,
    mean_with_error,
    paired_difference,
    parse_json,
    split_reasoning,
)
from stagetrain.tracks.llm.select import choose, list_checkpoints


def judge(name, output, reference, **spec):
    return build_scorer(ScorerSpec(name=name, **spec))(output, reference, {})


# ------------------------------------------------------------------- scorers


def test_reasoning_is_separated_from_the_answer():
    assert split_reasoning("<think>2+2</think> 4") == ("2+2", "4")
    assert split_reasoning("just 4") == (None, "just 4")
    assert split_reasoning("<think>ran out of tok") == ("ran out of tok", "")  # never closed: no answer
    assert split_reasoning("2+2</think>4") == ("2+2", "4")  # the template opened the block in the prompt
    assert has_valid_reasoning("<think>because</think> 4")
    assert not has_valid_reasoning("<think></think> 4") and not has_valid_reasoning("4")
    assert not has_valid_reasoning("<think>because</think>")  # reasoning with no answer after it


def test_exact_and_contains_ignore_case_whitespace_and_reasoning():
    assert judge("exact", "<think>hm</think>  Paris ", "paris") == 1.0
    assert judge("exact", "It is Paris", "Paris") == 0.0 and judge("contains", "It is Paris.", "paris") == 1.0
    assert judge("exact", "paris", "Paris", case_sensitive=True) == 0.0
    assert judge("contains", "anything", "") == 0.0
    assert judge("exact", "<think>Paris</think> Rome", "Paris") == 0.0  # the reasoning is not the answer
    assert judge("contains", "<think>Paris</think> Rome", "Paris", strip_thinking=False) == 1.0


def test_numeric_reads_the_last_number():
    assert last_number("so 3 apples and 1,250.50 dollars") == 1250.5 and last_number("none") is None
    assert judge("numeric", "First 12, then 30. The total is 42.", "#### 42") == 1.0
    assert judge("numeric", "about 41", "42") == 0.0 and judge("numeric", "no number", "42") == 0.0
    assert judge("numeric", "0.3333", "0.3334", tolerance=1e-3) == 1.0
    assert judge("numeric", "-7", "-7.0") == 1.0


@pytest.mark.parametrize("text, letter", [
    ("B", "B"), ("(c)", "C"), ("The answer is (B).", "B"), ("Answer: d", "D"),
    ("I think A is wrong, so the answer is C", "C"), ("It could be anything", None), ("option b", "B"),
])
def test_choice_letter(text, letter):
    assert choice_letter(text) == letter


def test_choice_regex_json_and_custom_scorers():
    assert judge("choice", "<think>A? no</think> The answer is B", "B") == 1.0
    assert judge("choice", "The answer is B", "(A)") == 0.0
    assert judge("regex", "blah FINAL: 17", "17", pattern=r"FINAL:\s*(\S+)") == 1.0
    assert judge("regex", "FINAL: 17", "FINAL: 18", pattern=r"FINAL:\s*(\S+)") == 0.0  # the reference's group is used
    assert judge("regex", "no marker", "17", pattern=r"FINAL:\s*(\S+)") == 0.0

    assert parse_json('Here: ```json\n{"a": 1}\n```') == {"a": 1} and parse_json('x {"a": [1, 2]} y') == {"a": [1, 2]}
    assert judge("json", '{"label": "spam", "why": "x"}', '{"label": "spam", "why": "y"}', fields=["label"]) == 1.0
    assert judge("json", '{"label": "ham"}', '{"label": "spam"}', fields=["label"]) == 0.0
    assert judge("json", '{"a": 1, "b": 2}', '{"b": 2, "a": 1}') == 1.0 and judge("json", "not json", "{}") == 0.0

    custom = {"function": "llm_fakes:partial_credit"}
    assert judge("custom", "42", "42", **custom) == 1.0  # the function returned 5: clamped into [0, 1]
    assert judge("custom", "it is 42", "42", **custom) == 0.5 and judge("custom", "7", "42", **custom) == 0.0
    with pytest.raises(ValueError, match="is not a callable"):
        build_scorer(ScorerSpec(name="custom", function="llm_fakes:nope"))
    for bad in ({"name": "regex"}, {"name": "custom"}, {"name": "custom", "function": "nocolon"}, {"name": "regex", "pattern": "("}):
        with pytest.raises(ValueError):
            ScorerSpec(**bad)


# ---------------------------------------------------------------- statistics


def test_mean_and_standard_error():
    stats = mean_with_error([1.0] * 70 + [0.0] * 30)
    assert stats["mean"] == 0.7 and stats["se"] == pytest.approx(0.0461, abs=1e-4)
    assert stats["ci_low"] == pytest.approx(0.6097, abs=1e-3) and stats["ci_high"] == pytest.approx(0.7903, abs=1e-3)
    assert mean_with_error([]) == {"n": 0.0} and mean_with_error([1.0])["se"] == 0.0


def test_paired_difference_is_tighter_than_two_separate_intervals():
    base = {str(i): float(i % 10 < 6) for i in range(200)}  # 60 % right
    tuned = {k: 1.0 if i % 10 == 6 else base[k] for i, k in enumerate(base)}  # the same answers plus one more in ten
    diff = paired_difference(tuned, base)
    assert diff["delta"] == pytest.approx(0.1) and diff["wins"] == 20 and diff["losses"] == 0 and diff["n"] == 200
    assert diff["ci_low"] > 0  # clearly better...
    separate = mean_with_error(list(tuned.values()))["ci_low"] - mean_with_error(list(base.values()))["ci_high"]
    assert separate < 0  # ...which two unpaired intervals of this size could not show
    assert paired_difference({"a": 1.0}, {"b": 1.0}) == {"n": 0.0}
    assert paired_difference({"a": 1.0, "x": 0.0}, {"a": 0.0, "y": 1.0})["n"] == 1  # only shared questions count


# ------------------------------------------------------------------- rewards


def test_rewards_follow_trl_calling_convention():
    functions, weights = build_rewards([
        RewardSpec(kind="correct", scorer={"name": "numeric"}, weight=1.0),
        RewardSpec(kind="format", pattern=r"^answer: ", weight=0.2),
        RewardSpec(kind="reasoning", weight=0.1),
        RewardSpec(kind="length", max_chars=20, weight=0.1),
    ])
    assert [f.__name__ for f in functions] == ["correct", "format", "reasoning", "length"] and weights == [1.0, 0.2, 0.1, 0.1]
    completions = [
        [{"role": "assistant", "content": "<think>6*7</think> answer: 42"}],
        [{"role": "assistant", "content": "41"}],
        [{"role": "assistant", "content": "answer: " + "x" * 52}],
    ]
    kwargs = {"prompts": [[{"role": "user", "content": "6*7?"}]] * 3, "completions": completions,
              "completion_ids": None, "answer": ["42", "42", "42"], "trainer_state": None}
    correct, fmt, reasoning, length = (f(**kwargs) for f in functions)
    assert correct == [1.0, 0.0, 0.0] and fmt == [1.0, 0.0, 1.0] and reasoning == [1.0, 0.0, 0.0]
    assert length[0] == -0.45 and length[1] == 0.0 and length[2] == -1.0  # 29 chars, 2 chars, 60 chars against 20
    assert completion_text("plain") == "plain" and completion_text([]) == "" and completion_text({"content": "c"}) == "c"

    with pytest.raises(ValueError, match="names must be distinct"):
        build_rewards([RewardSpec(), RewardSpec()])
    assert [f.__name__ for f in build_rewards([RewardSpec(name="a"), RewardSpec(name="b")])[0]] == ["a", "b"]
    with pytest.raises(ValueError, match="at least one reward"):
        build_rewards([])
    for bad in ({"kind": "format"}, {"kind": "length"}, {"kind": "custom"}):
        with pytest.raises(ValueError):
            RewardSpec(**bad)


# --------------------------------------------------------------- grpo config


def grpo(**over):
    args = {"output_dir": "out", "seed": 3, "num_generations": 8, "max_completion_length": 256}
    return grpo_config_kwargs(over.pop("train", GrpoTrainArgs()), **{**args, **over})


def test_grpo_kwargs_and_their_rules():
    kw = grpo(reward_weights=[1.0, 0.2], template_kwargs={"enable_thinking": False})
    # a constant 1e-5 and eight prompts (64 completions) per update
    assert kw["learning_rate"] == 1e-5 and kw["lr_scheduler_type"] == "constant" and kw["max_steps"] == 300
    assert kw["per_device_train_batch_size"] * kw["gradient_accumulation_steps"] == 64 and kw["num_generations"] == 8
    assert kw["loss_type"] == "dapo" and kw["beta"] == 0.0 and kw["temperature"] == 1.0
    assert kw["importance_sampling_level"] == "token" and kw["reward_weights"] == [1.0, 0.2]
    assert kw["chat_template_kwargs"] == {"enable_thinking": False} and kw["eval_strategy"] == "no"
    for unset in ("epsilon", "epsilon_high", "max_prompt_length", "steps_per_generation", "num_iterations",
                  "scale_rewards", "multi_objective_aggregation", "cast_lm_head_to_fp32"):
        assert unset not in kw  # the library's own defaults stand unless a stage asks otherwise
    assert grpo(loss_type="cispo")["epsilon_high"] == 5.0

    one = GrpoTrainArgs(per_device_train_batch_size=8, gradient_accumulation_steps=1)
    with pytest.raises(ValueError, match="= 8 must be a multiple of num_generations = 6"):
        grpo(train=one, num_generations=6)
    assert grpo(train=one, num_generations=4)["num_generations"] == 4  # 8 completions: two prompts
    assert grpo(train=GrpoTrainArgs(per_device_train_batch_size=4, gradient_accumulation_steps=3), num_generations=6)
    assert grpo(train=one, num_generations=4, num_processes=2)  # 16 completions over two GPUs
    with pytest.raises(ValueError, match="= 12 must be a multiple of num_generations = 8"):
        grpo(train=GrpoTrainArgs(per_device_train_batch_size=4, gradient_accumulation_steps=1), num_processes=3)
    for bad, message in (({"loss_type": "gspo"}, "loss_type must be one of"), ({"num_generations": 1}, "at least 2"),
                         ({"temperature": 0.0}, "temperature must be above 0"),
                         ({"importance_sampling_level": "word"}, "token .* or sequence"),
                         ({"scale_rewards": "std"}, "group, batch or none"),
                         ({"multi_objective_aggregation": "max"}, "sum_then_normalize or normalize_then_sum"),
                         ({"steps_per_generation": 12}, "multiple of gradient_accumulation_steps")):
        with pytest.raises(ValueError, match=message):
            grpo(**bad)


def test_gspo_is_the_grpo_loss_at_sequence_level():
    gspo = grpo(loss_type="grpo", importance_sampling_level="sequence")
    assert gspo["loss_type"] == "grpo" and gspo["importance_sampling_level"] == "sequence"
    assert grpo(loss_type="luspo", importance_sampling_level="sequence")["loss_type"] == "luspo"
    for loss in ("dapo", "dr_grpo", "bnpo", "cispo"):  # these would weight every sequence by its length
        with pytest.raises(ValueError, match="GSPO is loss_type: grpo"):
            grpo(loss_type=loss, importance_sampling_level="sequence")
    with pytest.raises(ValueError, match="luspo is a sequence-level loss"):
        grpo(loss_type="luspo")


def test_rollout_reuse_and_reward_processing_settings():
    kw = grpo(steps_per_generation=32, num_iterations=2, epsilon=0.2, epsilon_high=0.28, scale_rewards="batch",
              multi_objective_aggregation="normalize_then_sum", cast_lm_head_to_fp32=True)
    assert kw["steps_per_generation"] == 32 and kw["num_iterations"] == 2 and kw["epsilon_high"] == 0.28
    assert kw["scale_rewards"] == "batch" and kw["multi_objective_aggregation"] == "normalize_then_sum"
    assert kw["cast_lm_head_to_fp32"] is True


def test_grpo_kwargs_build_the_installed_trl_config(tmp_path):
    trl = pytest.importorskip("trl")
    kw = grpo(output_dir=str(tmp_path), reward_weights=[1.0], template_kwargs={"enable_thinking": False},
              loss_type="grpo", importance_sampling_level="sequence", mask_truncated_completions=True,
              steps_per_generation=16, num_iterations=2, epsilon=0.2, epsilon_high=0.28, scale_rewards="batch",
              multi_objective_aggregation="normalize_then_sum",
              train=GrpoTrainArgs(precision="fp32", extra={"use_cpu": True}))
    config = trl.GRPOConfig(**fit_config_kwargs(kw, trl.GRPOConfig))
    assert config.num_generations == 8 and config.loss_type == "grpo" and config.max_completion_length == 256
    assert config.learning_rate == 1e-5 and config.per_device_train_batch_size == 8 and config.reward_weights == [1.0]
    assert config.chat_template_kwargs == {"enable_thinking": False} and config.importance_sampling_level == "sequence"
    assert config.mask_truncated_completions is True and config.steps_per_generation == 16 and config.num_iterations == 2
    assert config.epsilon_high == 0.28 and config.scale_rewards == "batch"
    assert config.multi_objective_aggregation == "normalize_then_sum"
    for loss in ("sapo", "vespo"):  # every objective the stage accepts is one the library knows
        assert trl.GRPOConfig(**fit_config_kwargs(grpo(output_dir=str(tmp_path), loss_type=loss,
                                                       train=GrpoTrainArgs(precision="fp32")), trl.GRPOConfig)).loss_type == loss


def test_distillation_config(tmp_path):
    trl = pytest.importorskip("trl")
    from stagetrain.tracks.llm.backend import DpoTrainArgs, distill_config_kwargs

    kw = distill_config_kwargs(DpoTrainArgs(precision="fp32", max_steps=10), output_dir=str(tmp_path), seed=1,
                               teacher="org/teacher", max_completion_length=128, beta=0.5, temperature=0.9)
    config = trl.DistillationConfig(**fit_config_kwargs(kw, trl.DistillationConfig))
    assert config.teacher_model_name_or_path == "org/teacher" and config.beta == 0.5 and config.max_completion_length == 128
    with pytest.raises(ValueError, match="between 0"):
        distill_config_kwargs(DpoTrainArgs(), output_dir="o", seed=1, teacher="t", max_completion_length=8, beta=1.5)


def test_gated_rewards_are_paid_only_on_correct_answers():
    functions, weights = build_rewards([
        RewardSpec(kind="correct", scorer={"name": "numeric"}),
        RewardSpec(kind="format", pattern=r"^answer: ", weight=0.2, gated=True),
        RewardSpec(kind="format", name="ungated_format", pattern=r"^answer: ", weight=0.2),
    ])
    completions = [[{"role": "assistant", "content": c}] for c in ("answer: 42", "answer: 41", "42")]
    kwargs = {"prompts": [None] * 3, "completions": completions, "answer": ["42"] * 3}
    correct, gated, ungated = (f(**kwargs) for f in functions)
    assert correct == [1.0, 0.0, 1.0]
    assert ungated == [1.0, 1.0, 0.0]  # a wrong answer in the right format still collects this one
    assert gated == [1.0, 0.0, 0.0]  # and not this one
    with pytest.raises(ValueError, match="add a `correct` reward"):
        build_rewards([RewardSpec(kind="format", pattern="x", gated=True)])
    with pytest.raises(ValueError, match="cannot be gated on itself"):
        build_rewards([RewardSpec(kind="correct", gated=True)])


def test_vllm_loading_is_lora_only():
    kw = load_kwargs(LoadSpec("m", load_in="16bit", fast_inference=True, max_lora_rank=32, gpu_memory_utilization=0.9))
    assert kw["fast_inference"] is True and kw["max_lora_rank"] == 32 and kw["gpu_memory_utilization"] == 0.9
    assert "fast_inference" not in load_kwargs(LoadSpec("m"))
    for spec in (LoadSpec("m", load_in="8bit", fast_inference=True),
                 LoadSpec("m", load_in="16bit", full_finetuning=True, fast_inference=True)):
        with pytest.raises(ValueError, match="LoRA on a 4bit or 16bit base only"):
            load_kwargs(spec)


# -------------------------------------------------------- checkpoint choice


def test_checkpoint_choice(tmp_path):
    for step in (300, 100, 200):
        (tmp_path / f"checkpoint-{step}").mkdir()
    (tmp_path / "checkpoint-final.txt").write_text("not a checkpoint")
    assert [step for step, _ in list_checkpoints(tmp_path)] == [100, 200, 300]
    scores = [(100, 0.70), (200, 0.79), (300, 0.80)]
    assert choose(scores) == (300, 0.80)
    assert choose(scores, tolerance=0.02) == (200, 0.79)  # nearly as good and less trained
    assert choose([(100, 0.8), (200, 0.8)]) == (100, 0.8)  # a tie goes to the earlier one


def test_optimizer_settings_reach_the_library_config(tmp_path):
    trl = pytest.importorskip("trl")
    from stagetrain.tracks.llm.backend import LlmTrainArgs, sft_config_kwargs

    plain = sft_config_kwargs(LlmTrainArgs(precision="fp32"), output_dir=str(tmp_path), seed=1, has_eval=False, max_seq_length=64)
    assert not {"adam_beta1", "adam_beta2", "adam_epsilon", "lr_scheduler_kwargs", "loraplus_lr_ratio"} & set(plain)
    train = LlmTrainArgs(
        precision="fp32", optim="adamw_torch", adam_beta1=0.85, adam_beta2=0.95, adam_epsilon=1e-6,
        lr_scheduler_type="cosine_with_min_lr", lr_scheduler_kwargs={"min_lr_rate": 0.1}, warmup=0, weight_decay=0.0,
    )
    kw = sft_config_kwargs(train, output_dir=str(tmp_path), seed=1, has_eval=False, max_seq_length=64)
    config = trl.SFTConfig(**fit_config_kwargs(kw, trl.SFTConfig))
    assert (config.adam_beta1, config.adam_beta2, config.adam_epsilon) == (0.85, 0.95, 1e-6)
    assert config.lr_scheduler_kwargs == {"min_lr_rate": 0.1} and config.weight_decay == 0.0 and config.optim == "adamw_torch"
    with pytest.raises(ValueError, match="adam_beta2"):
        LlmTrainArgs(adam_beta2=1.0)
    ratio = sft_config_kwargs(train, output_dir=str(tmp_path), seed=1, has_eval=False, max_seq_length=64, loraplus_lr_ratio=16)
    assert ratio["loraplus_lr_ratio"] == 16
    with pytest.raises(ValueError, match="one of them"):
        sft_config_kwargs(train, output_dir=str(tmp_path), seed=1, has_eval=False, max_seq_length=64,
                          loraplus_lr_ratio=16, embedding_learning_rate=1e-5)
