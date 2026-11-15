"""The decision track on a tiny random model: prompt, verbalizer, batches, loss, and the whole
pipeline end to end on CPU. No weights are downloaded."""

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from decision_tiny import LEVELS, build_model_dir, build_tokenizer, reviews  # noqa: E402

from stagetrain.core import Pipeline, build_config, write_jsonl  # noqa: E402
from stagetrain.tracks.decision.inference import DecisionModel  # noqa: E402
from stagetrain.tracks.decision.modeling import (  # noqa: E402
    AugmentConfig,
    DecisionCollator,
    LossConfig,
    decision_loss,
    option_scores,
    read_meta,
    slot_logits,
    smooth_targets,
)
from stagetrain.tracks.decision.prompt import PromptStyle, Verbalizer, encode_prompt, render_question  # noqa: E402
from stagetrain.tracks.decision.schema import TYPES, Question, Record  # noqa: E402

CHOICE = {"type": "choice", "instructions": "Which sentiment does the review have?",
          "criteria": {"positive": "praise", "negative": "complaint", "neutral": None}}
NOUL = {"type": "noul", "instructions": "Is the review positive?"}
SCORE = {"type": "score", "instructions": "How good is it?", "criteria": LEVELS}


@pytest.fixture(scope="module")
def tok():
    return build_tokenizer()


# ------------------------------------------------------ prompt and verbalizer


def test_verbalizer_gives_each_option_its_own_single_tokens(tok):
    v = Verbalizer(tok)
    codes, groups = v.groups(Question.parse(CHOICE))
    assert codes == ["A", "B", "C"] and all(len(g) >= 1 for g in groups)
    assert len({i for g in groups for i in g}) == sum(len(g) for g in groups)  # no token stands for two options
    assert [tok.decode(g[0]) for g in groups] == ["A", "B", "C"]
    codes, groups = v.groups(Question.parse(NOUL))
    assert codes == ["yes", "no"] and {tok.decode(i) for i in groups[0]} == {"yes", "Yes", "YES"}
    assert not set(groups[0]) & set(groups[1])
    codes, groups = v.groups(Question.parse(SCORE))
    assert codes == ["0", "1", "2"] and [tok.decode(g[0]) for g in groups] == ["0", "1", "2"]
    many = Question.parse({"type": "choice", "instructions": "x", "criteria": {f"o{i}": None for i in range(30)}})
    codes, groups = v.groups(many)
    assert codes[:3] == ["00", "01", "02"] and len(codes) == 30  # past 26 options, two-character codes
    too_many = Question.parse({"type": "choice", "instructions": "x", "criteria": {f"o{i}": None for i in range(800)}})
    with pytest.raises(RuntimeError, match="single-token codes"):
        v.groups(too_many)


def test_prompt_ends_at_the_answer_slot(tok):
    q = Question.parse(CHOICE)
    text = render_question(q, ["A", "B", "C"])
    assert "A. positive: praise" in text and "C. neutral" in text and text.endswith("code of one option.")
    assert "yes or no" in render_question(Question.parse(NOUL), ["yes", "no"])
    assert "2: positive" in render_question(Question.parse(SCORE), ["0", "1", "2"])
    ids, cut = encode_prompt(tok, "the movie was great", q, ["A", "B", "C"], PromptStyle(system="system"))
    decoded = tok.decode(ids)
    assert not cut and decoded.startswith("<|im_start|> system") and decoded.endswith("<|im_start|> assistant")
    assert "QUESTION" in decoded
    blind, _ = encode_prompt(tok, None, q, ["A", "B", "C"], PromptStyle())
    assert "QUESTION" not in tok.decode(blind) and len(blind) < len(ids)
    sections, _ = encode_prompt(tok, {"context": "great", "question": "okay"}, q, ["A", "B", "C"],
                                PromptStyle(state_format="sections"))
    assert "[ context ]" in tok.decode(sections)


def test_only_the_state_is_truncated(tok):
    q = Question.parse(CHOICE)
    short, _ = encode_prompt(tok, "great", q, ["A", "B", "C"], PromptStyle())
    state = " ".join(["great"] * 150 + ["awful"] * 150)
    ids, cut = encode_prompt(tok, state, q, ["A", "B", "C"], PromptStyle(), max_length=len(short) + 40)
    decoded = tok.decode(ids)
    assert cut and len(ids) <= len(short) + 40
    assert "great" in decoded and "awful" in decoded and "[...]" in decoded  # head and tail kept, middle cut
    assert "A . positive : praise" in decoded and decoded.endswith("<|im_start|> assistant")
    with pytest.raises(ValueError, match="does not fit"):
        encode_prompt(tok, state, q, ["A", "B", "C"], PromptStyle(), max_length=10)


# --------------------------------------------------------------------- batches


def records():
    return [
        Record.parse({"id": "c", "source": "s", "state": "the movie was awful", "question": CHOICE, "answer": "negative"}),
        Record.parse({"id": "n", "source": "s", "state": "great", "question": NOUL, "answer": True}),
        Record.parse({"id": "r", "source": "s", "state": "the film was really okay and fine", "question": SCORE, "answer": 1}),
    ]


def test_collator_left_pads_and_lays_out_options(tok):
    batch = DecisionCollator(tok, PromptStyle(), max_length=128, with_meta=True)(records())
    assert batch["input_ids"].shape == batch["attention_mask"].shape
    assert batch["attention_mask"][:, -1].tolist() == [1, 1, 1]  # the answer slot is every row's last position
    assert batch["attention_mask"].sum(1).tolist() == [m["tokens"] for m in batch["meta"]]
    assert (batch["attention_mask"][:, 0] == 0).any()  # shorter rows are padded on the left
    assert batch["targets"].tolist() == [[0, 1, 0], [1, 0, 0], [0, 1, 0]]
    assert batch["qtype"].tolist() == [TYPES.index("choice"), TYPES.index("noul"), TYPES.index("score")]
    assert batch["option_mask"].any(-1).tolist() == [[True, True, True], [True, True, False], [True, True, True]]
    rows = DecisionCollator(tok, PromptStyle(), 128)([{"record": json.dumps(r.to_json())} for r in records()])
    assert torch.equal(rows["input_ids"], batch["input_ids"]) and "meta" not in rows


def test_shuffling_moves_the_answer_with_its_option(tok):
    record = records()[0]
    collator = DecisionCollator(tok, PromptStyle(), 128, AugmentConfig(shuffle_options=True), seed=3)
    orders = set()
    for _ in range(30):
        e = collator.encode(record)
        orders.add(tuple(e["order"]))
        assert e["order"][e["target"].index(1.0)] == 1  # the gold option is still "negative"
    assert len(orders) > 1
    fixed = DecisionCollator(tok, PromptStyle(), 128, permute_seed=5)
    assert fixed.encode(record)["order"] == fixed.encode(record)["order"]
    for other in records()[1:]:  # yes/no and score keep their order
        assert collator.encode(other)["order"] == sorted(collator.encode(other)["order"])
    dropped = DecisionCollator(tok, PromptStyle(), 128, AugmentConfig(shuffle_options=False, description_dropout=1.0))
    assert "praise" not in tok.decode(dropped.encode(record)["ids"])


# ------------------------------------------------------------------------ loss


def test_option_scores_and_loss_match_a_manual_computation():
    slot = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [5.0, 4.0, 3.0, 2.0, 1.0, 0.0]], requires_grad=True)
    batch = {
        # row 0: three options, the first written by two tokens; row 1: two options and a padding option
        "option_ids": torch.tensor([[[1, 2], [3, 0], [4, 0]], [[0, 0], [5, 0], [0, 0]]]),
        "option_mask": torch.tensor([[[True, True], [True, False], [True, False]],
                                     [[True, False], [True, False], [False, False]]]),
        "targets": torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        "qtype": torch.tensor([1, 0]),
    }
    scores = option_scores(slot, batch["option_ids"], batch["option_mask"])
    first = torch.logsumexp(torch.tensor([1.0, 2.0]), 0)
    assert scores[0].tolist() == pytest.approx([first.item(), 3.0, 4.0])
    assert scores[1, :2].tolist() == [5.0, 0.0] and scores[1, 2] == float("-inf")

    loss, _ = decision_loss(slot, batch, LossConfig(vocab_mass_weight=0.0))
    manual = (-torch.log_softmax(torch.tensor([first.item(), 3.0, 4.0]), 0)[0]
              - torch.log_softmax(torch.tensor([5.0, 0.0]), 0)[1]) / 2
    assert loss.item() == pytest.approx(manual.item(), rel=1e-5)
    loss.backward()
    assert slot.grad is not None and torch.isfinite(slot.grad).all()
    assert slot.grad[1, 1:5].abs().sum() == 0  # tokens that stand for no option get no gradient

    with_mass, _ = decision_loss(slot, batch, LossConfig(vocab_mass_weight=1.0))
    assert with_mass.item() > loss.item()  # some vocabulary mass lies outside the options


def test_target_smoothing():
    targets = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    valid = torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.0]])
    qtype = torch.tensor([TYPES.index("score"), TYPES.index("noul"), TYPES.index("choice")])
    out = smooth_targets(targets, qtype, valid, LossConfig(label_smoothing=0.1, ordinal_sigma=1.0))
    assert out.sum(-1).tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert out[0, 0] > out[0, 1] > out[0, 2] > out[0, 3] > 0  # a score's mass falls off with level distance
    assert out[1].tolist() == pytest.approx([0.05, 0.95, 0.0, 0.0])  # smoothing stays inside the real options
    assert out[2].tolist() == pytest.approx([0.9 + 0.1 / 3, 0.1 / 3, 0.1 / 3, 0.0])
    assert torch.equal(smooth_targets(targets, qtype, valid, LossConfig()), targets)


def test_readout_is_independent_of_padding(tok, tmp_path):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(str(build_model_dir(tmp_path / "m"))).eval()
    collator = DecisionCollator(tok, PromptStyle(), 128)
    together = collator(records())
    with torch.no_grad():
        batched = slot_logits(model, together["input_ids"], together["attention_mask"])
        for i, record in enumerate(records()):
            alone = collator([record])
            single = slot_logits(model, alone["input_ids"], alone["attention_mask"])
            assert torch.allclose(batched[i], single[0], atol=1e-4)
    assert batched.shape == (3, len(tok))


# ---------------------------------------------------------------- end to end


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("decision")
    base = build_model_dir(tmp / "base")
    write_jsonl(tmp / "reviews.jsonl", reviews(900, seed=0))
    data = {"path": str(tmp / "reviews.jsonl")}
    train = {"learning_rate": 3e-3, "num_train_epochs": 3, "per_device_train_batch_size": 16, "logging_steps": 50,
             "warmup": 0.05, "weight_decay": 0.0, "precision": "fp32", "extra": {"use_cpu": True}}
    member = {"init_from": "${align.model}", "train_data": "${data.train}", "eval_data": "${data.eval}",
              "max_length": 96, "lora": {"r": 4, "alpha": 8},
              "train": {**train, "learning_rate": 1e-3, "num_train_epochs": 1}}
    config = build_config({
        "track": "decision", "name": "tiny", "output_dir": str(tmp / "runs"), "seed": 11,
        "stages": [
            {"name": "data", "kind": "decision.build_data", "params": {"sources": [
                {"name": "topic", "task": "classification", "data": {**data, "columns": {"label": "name"}},
                 "instructions": "Which sentiment does the review have?",
                 "descriptions": {"positive": "praise", "negative": "complaint"},
                 "eval_fraction": 0.1, "calibration_fraction": 0.1},
                {"name": "positive", "task": "binary", "data": {**data, "columns": {"label": "is_positive"}},
                 "instructions": "Is the review positive?", "eval_fraction": 0.1, "calibration_fraction": 0.1},
                {"name": "stars", "task": "rating", "data": data, "instructions": "How good is it?",
                 "levels": LEVELS, "eval_fraction": 0.1, "calibration_fraction": 0.1},
            ]}},
            {"name": "align", "kind": "decision.train", "params": {
                "init_from": str(base), "train_data": "${data.train}", "eval_data": "${data.eval}",
                "max_length": 96, "lora": None, "train": train, "augment": {"shuffle_options": False},
                "loss": {"label_smoothing": 0.05, "ordinal_sigma": 0.5}},
             "gate": {"metric": "eval/accuracy", "min": 0.9}},
            {"name": "full-a", "kind": "decision.train",
             "params": {**member, "seed": 1, "augment": {"shuffle_options": False}}},
            {"name": "full-b", "kind": "decision.train",
             "params": {**member, "seed": 2, "mixture": {"stars": 0.5}, "augment": {"shuffle_options": False}}},
            {"name": "soup", "kind": "decision.soup", "params": {
                "models": ["${full-a.model}", "${full-b.model}"], "method": "greedy", "data": "${data.calibration}",
                "precision": "fp32"}},
            {"name": "calibrate", "kind": "decision.calibrate", "params": {
                "model": "${soup.model}", "data": "${data.calibration}", "precision": "fp32"}},
            {"name": "eval", "kind": "decision.eval", "params": {
                "model": "${soup.model}", "data": "${data.eval}", "calibration": "${calibrate.calibration}",
                "permutations": 2, "precision": "fp32"},
             "gate": [{"metric": "eval/accuracy", "baseline": "align.eval/accuracy", "max_drop": 0.05},
                      {"metric": "eval/blind_gap", "min": 0.2}]},
            {"name": "export", "kind": "decision.export", "params": {
                "model": "${soup.model}", "calibration": "${calibrate.calibration}"}},
        ],
    })
    pipeline = Pipeline(config)
    pipeline.run()
    return pipeline


def test_pipeline_trains_a_decision_model(run):
    m = {name: run.manifest(name) for name in ("data", "align", "full-a", "full-b", "soup", "calibrate", "eval", "export")}
    assert all(x.completed for x in m.values())
    assert m["data"].metrics["data/sources"] == 3 and m["data"].metrics["data/train"] > 1500
    assert m["data"].metrics["data/duplicates"] > 0  # 900 rows of a small vocabulary repeat; repeats are dropped
    assert m["align"].metrics["eval/accuracy"] >= 0.9
    assert m["align"].metrics["train/loss"] < 1.5
    assert "adapter" in m["full-a"].artifacts and "adapter" not in m["align"].artifacts
    assert m["full-b"].metrics["train/records"] < m["full-a"].metrics["train/records"]  # its mixture halves a source
    assert 1 <= m["soup"].metrics["soup/members"] <= 2
    assert m["soup"].metrics["soup/nll"] <= m["soup"].metrics["soup/best_single_nll"] + 1e-9


def test_calibration_and_eval_probes(run):
    cal, ev = run.manifest("calibrate").metrics, run.manifest("eval").metrics
    assert cal["calibration/nll_after"] <= cal["calibration/nll_before"] + 1e-9
    assert all(cal[f"calibration/temperature_{t}"] > 0 for t in TYPES)
    assert ev["eval/accuracy"] >= 0.85 and ev["eval/accuracy_noul"] >= 0.85
    assert ev["eval/blind_gap"] >= 0.2  # without the state the model falls back to the answer prior
    assert 0 <= ev["eval/permutation_consistency"] <= 1
    assert "eval/nll_uncalibrated" in ev and "eval/noul_auroc" in ev and "eval/score_mae" in ev
    report = json.loads(open(run.manifest("eval").artifact("report", run.run_dir)).read())
    assert set(report["by_source"]) == {"topic", "positive", "stars"}


def test_exported_model_answers_typed_questions(run):
    path = run.manifest("export").artifact("model", run.run_dir)
    meta = read_meta(path)
    assert meta["calibration"] and meta["max_length"] == 96 and meta["prompt"]["question_header"] == "QUESTION:"
    model = DecisionModel.load(path, precision="fp32", device="cpu")
    questions = {
        "topic": {"type": "choice", "instructions": "Which sentiment does the review have?",
                  "criteria": {"negative": "complaint", "neutral": None, "positive": "praise"}},
        "positive": NOUL,
        "stars": SCORE,
    }
    good, bad = model.decide_batch([("the movie was excellent and story", questions),
                                    ("the film was dreadful and acting", questions)])
    assert good["answers"]["topic"]["choice"] == "positive" and bad["answers"]["topic"]["choice"] == "negative"
    assert good["answers"]["positive"]["noul"] > 0.5 > bad["answers"]["positive"]["noul"]
    assert good["answers"]["stars"]["score"] == 2 and bad["answers"]["stars"]["score"] == 0
    assert sum(good["answers"]["topic"]["probabilities"].values()) == pytest.approx(1.0)
    assert good["usage"]["output_tokens"] == 0 and good["usage"]["input_tokens"] > 0
    assert model.decide("the movie was superb", {"positive": NOUL})["answers"]["positive"]["noul"] > 0.5


def test_a_second_run_is_fully_cached(run):
    again = Pipeline(run.config)
    assert {s.state for s in again.status()} == {"cached"}
