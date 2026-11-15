"""The accuracy pipeline of the llm track, end to end through `Pipeline` with the fake backend:
baselines, a learning-rate sweep, checkpoint selection, self-generated data, DPO, GRPO and a
paired final evaluation."""

import json

import pytest
from llm_fakes import FakeBackend

from stagetrain.core import GateFailed, Pipeline, PipelineError, build_config, read_jsonl, write_jsonl
from stagetrain.core.history import load_history
from stagetrain.core.report import report_data
from stagetrain.tracks.llm import set_backend
from stagetrain.tracks.llm.data import question_key
from stagetrain.tracks.llm.evaluate import build_prompt, prompt_key, sampled_metrics
from stagetrain.tracks.llm.generate import nearest, rl_prompts, sort_answers
from stagetrain.tracks.llm.scoring import ScorerSpec, build_scorer
from stagetrain.tracks.llm.train import reward_trend

pytest.importorskip("datasets")


@pytest.fixture
def backend():
    fake = FakeBackend()
    fake.checkpoint_steps = [10, 20, 30]
    fake.loss_by_lr = {1e-4: 1.4, 3e-4: 1.1, 1e-3: 1.6}
    # how many of every ten questions a model answers correctly
    fake.skill = {"org/base": 5, "checkpoint-10": 6, "checkpoint-20": 8, "checkpoint-30": 8, "grpo": 9}
    set_backend(fake)
    yield fake
    set_backend(None)


@pytest.fixture
def sources(tmp_path):
    chat = [{"messages": [{"role": "user", "content": f"Question {i}?"}, {"role": "assistant", "content": f"Answer {i}"}]}
            for i in range(200)]
    chat.append({"messages": [{"role": "user", "content": "Question 7?"}, {"role": "assistant", "content": "Another answer"}]})
    write_jsonl(tmp_path / "chat.jsonl", chat)
    write_jsonl(tmp_path / "general.jsonl", [{"instruction": f"Do {i}", "input": "", "output": f"Done {i}"} for i in range(80)])
    return tmp_path


def stages(tmp_path):
    model = {"max_seq_length": 64}
    return [
        {"name": "data", "kind": "llm.prepare_data", "params": {
            "chat": [{"source": {"path": str(tmp_path / "chat.jsonl")}},
                     {"source": {"path": str(tmp_path / "general.jsonl")}, "format": "alpaca", "replay_ratio": 0.1}],
            "eval_fraction": 0.2, "test_fraction": 0.25}},
        {"name": "eval_base", "kind": "llm.eval", "params": {
            **model, "init_from": "org/base", "eval_data": "${data.chat_test}", "measure_loss": False}},
        {"name": "eval_prompted", "kind": "llm.eval", "params": {
            **model, "init_from": "org/base", "eval_data": "${data.chat_test}", "measure_loss": False,
            "few_shot": 2, "few_shot_data": "${data.chat_train}", "system_prompt": "Answer briefly.",
            "compare_to": "${eval_base.scores}"}},
        {"name": "sft", "kind": "llm.sft", "matrix": {"train.learning_rate": [1e-4, 3e-4, 1e-3]}, "params": {
            **model, "init_from": "org/base", "train_data": "${data.chat_train}", "eval_data": "${data.chat_eval}"}},
        {"name": "best_lr", "kind": "llm.select", "params": {"group": "sft", "metric": "eval/loss"}},
        {"name": "pick", "kind": "llm.select_checkpoint", "params": {
            **model, "init_from": "${best_lr.checkpoints}", "eval_data": "${data.chat_eval}"}},
        {"name": "selfplay", "kind": "llm.generate_data", "params": {
            **model, "init_from": "${pick.model}", "data": "${data.chat_train}", "samples_per_prompt": 4}},
        {"name": "dpo", "kind": "llm.dpo", "params": {
            **model, "init_from": "${pick.model}", "train_data": "${selfplay.preference}"}},
        {"name": "grpo", "kind": "llm.grpo", "params": {
            **model, "init_from": "${dpo.model}", "train_data": "${data.chat_train}", "num_generations": 4,
            "max_completion_length": 32, "chat_template_kwargs": {"enable_thinking": False},
            "rewards": [{"kind": "correct"}, {"kind": "reasoning", "weight": 0.1}],
            "train": {"per_device_train_batch_size": 4, "max_steps": 50}},
         "gate": {"metric": "train/reward_gain", "min": 0.0}},
        {"name": "eval_final", "kind": "llm.eval", "params": {
            **model, "init_from": "${grpo.model}", "eval_data": "${data.chat_test}", "measure_loss": False,
            "compare_to": "${eval_base.scores}"},
         "gate": {"metric": "eval/delta_ci_low", "min": 0.0}},
    ]


def pipeline(tmp_path, **over):
    spec = stages(tmp_path)
    for s in spec:
        s["params"].update(over.get(s["name"], {}))
    return Pipeline(build_config({"track": "llm", "name": "acc", "output_dir": str(tmp_path / "runs"), "seed": 5, "stages": spec}))


def test_data_has_three_splits_split_by_question_and_replay_only_in_train(sources, backend):
    p = pipeline(sources)
    m = p.run(only=["data"])[0]
    run = p.run_dir / "data"
    parts = {name: list(read_jsonl(run / f"chat.{name}.jsonl")) for name in ("train", "eval", "test")}
    assert m.metrics["chat/test_rows"] == len(parts["test"]) > 20 and m.metrics["chat/eval_rows"] == len(parts["eval"]) > 15
    keys = {name: {question_key(r) for r in rows} for name, rows in parts.items()}
    assert not keys["train"] & keys["eval"] and not keys["train"] & keys["test"] and not keys["eval"] & keys["test"]
    # "Question 7?" appears with two different answers: both rows land on the same side
    assert sum(any(r["messages"][0]["content"] == "Question 7?" for r in rows) for rows in parts.values()) == 1
    replay = [r for r in parts["train"] if r["messages"][0]["content"].startswith("Do ")]
    task = len(parts["train"]) - len(replay)
    assert len(replay) == round(0.1 * task) == m.metrics["chat/replay_rows"]
    assert not any(r["messages"][0]["content"].startswith("Do ") for r in parts["eval"] + parts["test"])
    assert len(parts["train"]) + len(parts["eval"]) + len(parts["test"]) == 201 + len(replay)


def test_accuracy_pipeline(sources, backend):
    p = pipeline(sources)
    m = {x.name: x for x in p.run()}

    # baselines: the same harness on the untuned model, with and without a better prompt
    base, prompted = m["eval_base"].metrics, m["eval_prompted"].metrics
    assert base["eval/score"] == pytest.approx(0.5, abs=0.06) and base["eval/score_se"] > 0
    assert base["eval/score_ci_low"] < base["eval/score"] < base["eval/score_ci_high"] and "eval/loss" not in base
    assert prompted["eval/delta"] == 0.0 and prompted["eval/delta_n"] == base["eval/generated"]  # no better, and it shows

    # the sweep: three runs, the lowest eval loss wins and hands on its artifacts
    assert [m[f"sft-{i}"].metrics["eval/loss"] for i in (1, 2, 3)] == [1.4, 1.1, 1.6]
    assert m["best_lr"].notes["winner"] == "sft-2" and m["best_lr"].notes["variant"] == {"train.learning_rate": 3e-4}
    assert m["best_lr"].artifacts["checkpoints"] == "run://sft-2/checkpoints"
    sft_call = backend.of("train_text")[1][1]
    assert sft_call["config"]["num_train_epochs"] == 3.0 and sft_call["config"]["lr_scheduler_type"] == "cosine"
    assert sft_call["config"]["save_total_limit"] is None and sft_call["resume"] is False
    assert load_history(p.run_dir / "sft-2")[0]["train/loss"] == 0.9  # curves for the report

    # checkpoint selection by task metric: 0.6, 0.8, 0.8 -> the earlier of the two best
    pick = m["pick"]
    assert [round(c["score"], 1) for c in pick.notes["checkpoints"]] == [0.6, 0.8, 0.8]
    assert pick.metrics["select/chosen_step"] == 20 and pick.metrics["select/best_step"] == 20
    assert pick.notes["chosen"].endswith("sft-2/checkpoints/checkpoint-20")
    assert [r["eval/score"] for r in load_history(p.run_dir / "pick")] == [c["score"] for c in pick.notes["checkpoints"]]

    # self-play: of every four prompts one is always solved, two sometimes, one never
    gen = m["selfplay"].metrics
    assert gen["generate/pass_at_k"] == pytest.approx(0.75, abs=0.02) and gen["generate/mixed_fraction"] == pytest.approx(0.5, abs=0.02)
    assert gen["generate/pairs"] == pytest.approx(gen["generate/prompts"] / 2, abs=1)
    pair = next(read_jsonl(p.run_dir / "selfplay" / "preference.jsonl"))
    assert pair["chosen"][0]["content"].endswith(FakeBackend.truth(pair["prompt"])) and pair["rejected"][0]["content"].startswith("wrong")
    kept = next(read_jsonl(p.run_dir / "selfplay" / "sft.jsonl"))
    assert kept["messages"][-1]["role"] == "assistant" and kept["messages"][-2]["role"] == "user"
    assert backend.of("sample")[0][2:] == (4, 0.8)

    # dpo trains on those pairs; grpo then learns from rewards
    assert len(backend.of("train_dpo")[0][1]["train"]) == gen["generate/pairs"]
    call = backend.of("train_grpo")[0][1]
    assert set(call["rows"][0]) == {"prompt", "answer"} and call["rows"][0]["prompt"][-1]["role"] == "user"
    assert call["rewards"]["correct"][:4] == [1.0, 0.0, 1.0, 0.0] and set(call["rewards"]["reasoning"]) == {0.0}
    config = call["config"]
    assert config["num_generations"] == 4 and config["reward_weights"] == [1.0, 0.1] and config["loss_type"] == "dapo"
    assert config["learning_rate"] == 1e-5 and config["max_steps"] == 50 and config["chat_template_kwargs"] == {"enable_thinking": False}
    grpo = m["grpo"].metrics
    assert grpo["train/reward_start"] == pytest.approx(0.2) and grpo["train/reward"] == pytest.approx(0.74)
    assert grpo["train/reward_gain"] == pytest.approx(0.54) and m["grpo"].notes["mode"] == "adapter:continue"
    assert [r["train/reward"] for r in load_history(p.run_dir / "grpo")][:2] == [0.2, 0.26]

    # the final model against the base model, question by question
    final = m["eval_final"].metrics
    assert final["eval/score"] == pytest.approx(0.9, abs=0.06) and final["eval/delta"] == pytest.approx(0.4, abs=0.08)
    assert final["eval/delta_ci_low"] > 0 and m["eval_final"].gates[0]["passed"]
    assert len(backend.of("release")) == len(backend.of("load"))  # every loaded model is released again

    report = report_data(p)
    scores = next(c for c in report["comparisons"] if c["metric"] == "eval/score")
    assert [r["stage"] for r in scores["rows"]] == ["eval_base", "eval_prompted", "pick", "eval_final"]
    assert next(s for s in report["stages"] if s["name"] == "grpo")["charts"][1]["title"] == "train/reward"


def test_few_shot_prompts_are_built_from_training_rows(sources, backend):
    p = pipeline(sources)
    p.run(until="eval_prompted")
    prompt = backend.prompts[0]
    assert [m["role"] for m in prompt] == ["system", "user", "assistant", "user", "assistant", "user"]
    assert prompt[0]["content"] == "Answer briefly."
    train = {r["messages"][0]["content"] for r in read_jsonl(p.run_dir / "data" / "chat.train.jsonl")}
    assert prompt[1]["content"] in train and prompt[3]["content"] in train and prompt[5]["content"] not in train
    question = [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}]
    assert build_prompt(question) == question
    assert prompt_key(question) == prompt_key(build_prompt(question, None, "another system prompt")[:])  # same question


def test_a_model_worse_than_the_base_fails_the_paired_gate(sources, backend):
    backend.skill["grpo"] = 4  # the final model answers one question in ten worse than the base model
    with pytest.raises(GateFailed, match="eval/delta_ci_low"):
        pipeline(sources).run()


def test_overlong_rows_are_dropped_not_truncated(sources, backend):
    long = {"messages": [{"role": "user", "content": "word " * 100}, {"role": "assistant", "content": "x"}]}
    write_jsonl(sources / "chat.jsonl", [*read_jsonl(sources / "chat.jsonl"), long])
    everything_trains = {"chat": [{"source": {"path": str(sources / "chat.jsonl")}}], "eval_fraction": 0.0, "test_fraction": 0.0}
    p = pipeline(sources, data=everything_trains, sft={"eval_data": None})
    p.run(only=["data", "sft-1"])
    assert p.manifest("sft-1").metrics["train/dropped_overlength"] == 1
    assert not any("word word" in t for t in backend.of("train_text")[0][1]["train"])
    kept = pipeline(sources, data=everything_trains, sft={"eval_data": None, "overlength": "truncate"})
    kept.run(only=["sft-1"])
    assert kept.manifest("sft-1").metrics["train/dropped_overlength"] == 0


@pytest.mark.parametrize("stage, params, message", [
    ("grpo", {"num_generations": 3}, "multiple of num_generations = 3"),
    ("grpo", {"eval_data": "x.jsonl"}, "takes no eval_data"),
    ("grpo", {"max_completion_length": 64}, "leave room for the prompt"),
    ("grpo", {"rewards": []}, "at least one reward"),
    ("grpo", {"fast_inference": True, "load_in": "8bit"}, "LoRA on a 4bit or 16bit base only"),
    ("eval_final", {"few_shot": 3}, "few_shot needs few_shot_data"),
    ("selfplay", {"temperature": 0.0}, "greater than 0"),
    ("pick", {"tolerance": -1}, "greater than or equal to 0"),
])
def test_bad_parameters_fail_before_anything_runs(sources, backend, stage, params, message):
    with pytest.raises(PipelineError, match=message):
        pipeline(sources, **{stage: params}).validate()
    assert backend.calls == []


def test_prompts_reserved_for_rl_are_not_trained_on_before(sources, backend):
    p = pipeline(sources, data={"rl_fraction": 0.3})
    m = p.run(only=["data"])[0]
    run = p.run_dir / "data"
    train = {question_key(r) for r in read_jsonl(run / "chat.train.jsonl")}
    rl = [r for r in read_jsonl(run / "chat.rl.jsonl")]
    assert m.metrics["chat/rl_rows"] == len(rl) > 20 and not train & {question_key(r) for r in rl}
    assert not any(r["messages"][0]["content"].startswith("Do ") for r in rl)  # replay rows are not RL prompts
    assert m.artifacts["chat_rl"] == "run://data/chat.rl.jsonl"


def test_generated_data_for_rl_and_for_pairs(sources, backend):
    # of every four prompts the fake model solves one always, two sometimes, one never
    p = pipeline(sources, selfplay={"negatives": "nearest", "prefer": "shortest", "keep_per_prompt": 2,
                                    "audit_scorer": {"name": "contains"}})
    p.run(until="selfplay")
    m = p.manifest("selfplay")
    run = p.run_dir / "selfplay"
    rated = list(read_jsonl(run / "prompts.jsonl"))
    assert {round(r["pass_rate"], 2) for r in rated} == {0.0, 0.5, 1.0} and all(r["samples"] == 4 for r in rated)
    rl = list(read_jsonl(run / "rl.jsonl"))
    always = sum(r["pass_rate"] == 1.0 for r in rated)
    sometimes = sum(0 < r["pass_rate"] < 0.9 for r in rated)
    # the prompts solved sometimes, a tenth of those always solved, none of those never solved
    assert len(rl) == sometimes + round(0.1 * always) == m.metrics["generate/rl_rows"]
    assert "pass_rate" not in rl[0] and rl[0]["messages"][-1]["role"] == "assistant"  # ordinary chat rows for llm.grpo
    assert m.metrics["generate/scorer_disagreement"] == 0.0  # nothing the task scorer rejects is accepted by the lenient one
    assert rl_prompts(rated, low=0.0, high=1.0, keep_solved=0.0) == [r for r in rated if 0 < r["pass_rate"] < 1]
    assert rl_prompts([], keep_solved=1.0) == []

    assert nearest("the answer is 12", ["I think it is banana", "the answer is 13"]) == "the answer is 13"
    judge = build_scorer()
    right, _, _ = sort_answers(["<think>a long chain of reasoning</think> A", "<think>short</think> A"], "A", {}, judge)
    assert sorted(right, key=len)[0] == "<think>short</think> A"


def test_eval_can_sample_for_pass_at_k_and_majority_vote(sources, backend):
    p = pipeline(sources, eval_base={"samples_per_prompt": 4})
    m = p.run(until="eval_base")[1].metrics
    # the fake model's samples: always right, sometimes right (twice), never right, by question
    assert m["eval/pass_at_k"] == pytest.approx(0.75, abs=0.05) and m["eval/sampled_pass_at_1"] == pytest.approx(0.5, abs=0.05)
    assert m["eval/pass_at_k"] >= m["eval/majority_vote"] >= 0.25
    assert backend.of("sample")[0][2:] == (4, 0.8)
    answers = [["<think>x</think> 4", "<think>y</think> 4", "5"], ["7", "8", "8"]]
    rows = [{"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": ref}]} for ref in ("4", "7")]
    assert sampled_metrics(answers, rows, ScorerSpec()) == {
        "eval/sampled_pass_at_1": 0.5, "eval/pass_at_k": 1.0, "eval/majority_vote": 0.5}


def test_helpers():
    judge = build_scorer()
    right, wrong, n_right = sort_answers(["A", "a ", "b", "<think>x</think> A", "b"], "A", {}, judge)
    assert right == ["A", "a ", "<think>x</think> A"] and wrong == ["b"] and n_right == 3  # distinct, order kept
    assert sort_answers(["A", "<think>x</think> A"], "A", {}, judge, require_reasoning=True)[0] == ["<think>x</think> A"]
    assert reward_trend([{"reward": r} for r in (0.0, 0.1, 0.5, 0.9)]) == {
        "train/reward_start": 0.0, "train/reward": 0.9, "train/reward_gain": 0.9}
    assert reward_trend([{"loss": 1.0}]) == {}
    assert json.dumps(sorted(reward_trend([{"reward": 1}]))) == '["train/reward", "train/reward_gain", "train/reward_start"]'
