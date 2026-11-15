import math
from pathlib import Path

import pytest

from stagetrain.core import Pipeline, load_config
from stagetrain.tracks.decision.data import SourceSpec, audit_source, bury, split_of, to_record
from stagetrain.tracks.decision.evaluate import permutation_report
from stagetrain.tracks.decision.inference import format_answer
from stagetrain.tracks.decision.metrics import (
    auroc,
    band,
    bootstrap_interval,
    by_source,
    confident_errors,
    ece,
    fit_calibration,
    fit_temperature,
    fit_temperatures,
    item_scores,
    macro_skill,
    out_of_fold,
    paired_difference,
    skill,
    softmax,
    summarize,
    temperature_of,
)
from stagetrain.tracks.decision.schema import Question, Record, content_key
from stagetrain.tracks.decision.train import apply_mixture

CHOICE = {"type": "choice", "instructions": "Which sentiment does the review have?",
          "criteria": {"positive": "praise", "negative": "complaint", "neutral": None}}
NOUL = {"type": "noul", "instructions": "Is the review positive?"}
SCORE = {"type": "score", "instructions": "How good is it?", "criteria": ["negative", "neutral", "positive"]}


# ------------------------------------------------------------------ schema


def test_question_types_and_answers():
    c, n, s = Question.parse(CHOICE), Question.parse(NOUL), Question.parse(SCORE)
    assert c.names == ("positive", "negative", "neutral") and c.descriptions == ("praise", "complaint", None)
    assert n.names == ("yes", "no") and s.names == ("0", "1", "2")
    assert (c.answer_index("negative"), n.answer_index(True), n.answer_index("No"), s.answer_index(2)) == (1, 0, 1, 2)
    assert (c.answer_value(2), n.answer_value(0), n.answer_value(1), s.answer_value(1)) == ("neutral", True, False, 1)
    assert Question.parse(c.to_json()) == c and Question.parse(s.to_json()) == s
    assert Question.parse(n.to_json()) == n
    assert c.permuted([2, 0, 1]).names == ("neutral", "positive", "negative")
    assert c.without_descriptions().descriptions == (None, None, None)
    for question, bad in ((c, "other"), (n, "maybe"), (s, 3), (s, True)):
        with pytest.raises(ValueError):
            question.answer_index(bad)
    with pytest.raises(ValueError, match="only choice"):
        s.permuted([0, 1, 2])


@pytest.mark.parametrize("bad", [
    {"type": "choice", "instructions": "x", "criteria": {"only": None}},
    {"type": "choice", "instructions": "", "criteria": {"a": None, "b": None}},
    {"type": "score", "instructions": "x", "criteria": ["one"]},
    {"type": "score", "instructions": "x", "criteria": [str(i) for i in range(11)]},
    {"type": "rank", "instructions": "x"},
])
def test_invalid_questions(bad):
    with pytest.raises(ValueError):
        Question.parse(bad)


def test_record_targets():
    r = Record.parse({"state": "s", "question": CHOICE, "answer": "negative"})
    assert r.target == (0.0, 1.0, 0.0) and r.answer_index == 1 and r.source == "default" and r.id
    soft = Record.parse({"state": "s", "question": SCORE, "target": [1, 1, 2]})
    assert soft.target == (0.25, 0.25, 0.5) and soft.to_json()["target"] == [0.25, 0.25, 0.5]
    assert "target" not in r.to_json() and r.to_json()["answer"] == "negative"
    for row in ({"state": "s", "question": CHOICE}, {"state": "s", "question": SCORE, "target": [1, 1]}):
        with pytest.raises(ValueError):
            Record.parse(row)


# ------------------------------------------------------------------ metrics


def test_softmax_ece_auroc():
    assert softmax([0.0, 0.0]) == [0.5, 0.5]
    assert softmax([1000.0, 0.0])[0] == pytest.approx(1.0)
    assert softmax([2.0, 0.0], temperature=2.0)[0] == pytest.approx(1 / (1 + math.exp(-1)))
    assert ece([0.75] * 4, [True, True, True, False]) == pytest.approx(0.0)
    assert ece([1.0] * 4, [True, False, True, False]) == pytest.approx(0.5)
    assert auroc([0.9, 0.8, 0.2, 0.1], [True, True, False, False]) == 1.0
    assert auroc([0.5, 0.5, 0.5, 0.5], [True, False, True, False]) == 0.5
    assert auroc([0.1, 0.9], [True, True]) is None


def test_summarize_and_by_source():
    preds = [
        {"type": "choice", "source": "a", "scores": [3.0, 0.0, 0.0], "target": [1, 0, 0]},
        {"type": "choice", "source": "a", "scores": [0.0, 3.0, 0.0], "target": [1, 0, 0]},
        {"type": "noul", "source": "b", "scores": [2.0, 0.0], "target": [1, 0]},
        {"type": "noul", "source": "b", "scores": [0.0, 2.0], "target": [0, 1]},
        {"type": "score", "source": "c", "scores": [0.0, 0.0, 5.0], "target": [1, 0, 0]},
    ]
    m = summarize(preds)
    assert m["n"] == 5 and m["accuracy"] == pytest.approx(3 / 5)
    assert m["accuracy_choice"] == 0.5 and m["accuracy_noul"] == 1.0 and m["accuracy_score"] == 0.0
    assert m["score_mae"] == 2.0 and m["noul_auroc"] == 1.0 and m["nll"] > 0 and 0 <= m["ece"] <= 1
    assert set(by_source(preds)) == {"a", "b", "c"} and by_source(preds)["b"]["accuracy"] == 1.0
    assert summarize([]) == {"n": 0.0}


def rows_with_frequencies(dists, kind="choice", scale=3.0, repeat=1):
    """Hard-labelled predictions whose labels occur exactly as often as `dists` say, and whose
    scores are `scale` times the log of those frequencies: the temperature `scale` makes the
    model exactly right."""
    preds = []
    for d in dists:
        scores = [scale * math.log(q) for q in d]
        for label, q in enumerate(d):
            for _ in range(round(q * 20) * repeat):
                preds.append({"type": kind, "scores": scores, "target": [float(i == label) for i in range(len(d))]})
    return preds


def test_temperature_fit_recovers_the_scale():
    preds = rows_with_frequencies([[0.7, 0.2, 0.1], [0.1, 0.6, 0.3], [0.25, 0.25, 0.5], [0.9, 0.05, 0.05]])
    assert fit_temperature(preds) == pytest.approx(3.0, rel=1e-3)
    temps = fit_temperatures(preds + [{"type": "noul", "scores": [1.0, 0.0], "target": [1, 0]}])
    assert temps["choice"] == pytest.approx(3.0, rel=1e-3) and temps["noul"] == 1.0 and temps["score"] == 1.0
    assert summarize(preds, temps)["nll"] < summarize(preds)["nll"]
    # Soft targets are fitted as hard labels: the fit asks which option was right, not how sure the label was.
    soft = [{"type": "choice", "scores": [2.0, 0.0], "target": [0.6, 0.4]}] * 30
    assert fit_temperature(soft) == fit_temperature([{**p, "target": [1.0, 0.0]} for p in soft])


def test_temperatures_stay_inside_their_bounds():
    # Every answer right and confident: the unbounded optimum is T -> 0, i.e. "be certain".
    perfect = [{"type": "noul", "scores": [4.0, 0.0], "target": [1.0, 0.0]}] * 60
    assert fit_temperature(perfect) == pytest.approx(0.5, rel=1e-3)
    assert fit_temperature(perfect, lo=0.2) == pytest.approx(0.2, rel=1e-3)
    # Always wrong and confident: the optimum is T -> infinity.
    wrong = [{"type": "noul", "scores": [4.0, 0.0], "target": [0.0, 1.0]}] * 60
    assert fit_temperature(wrong) == pytest.approx(5.0, rel=1e-3)
    assert fit_temperature([]) == 1.0


def test_calibration_per_option_count_band():
    assert [band(n) for n in (2, 3, 5, 6, 10, 11, 77)] == ["2", "3-5", "3-5", "6-10", "6-10", "11+", "11+"]
    few = rows_with_frequencies([[0.7, 0.2, 0.1], [0.2, 0.7, 0.1]], scale=1.5, repeat=2)  # 3 options: T = 1.5
    many = rows_with_frequencies([[0.4] + [0.05] * 12], scale=3.0, repeat=3)  # 13 options: T = 3
    calibration = fit_calibration(few + many, min_examples=40)
    assert calibration["bands"]["choice"]["3-5"] == pytest.approx(1.5, rel=1e-2)
    assert calibration["bands"]["choice"]["11+"] == pytest.approx(3.0, rel=1e-2)
    assert 1.5 < calibration["temperature"]["choice"] < 3.0  # one temperature for both would fit neither
    assert temperature_of("choice", 4, calibration) == calibration["bands"]["choice"]["3-5"]
    assert temperature_of("choice", 8, calibration) == calibration["temperature"]["choice"]  # no band: the type's
    assert temperature_of("noul", 2, calibration) == 1.0 and temperature_of("choice", 4, None) == 1.0
    assert temperature_of("choice", 4, {"choice": 2.0}) == 2.0  # a flat {type: T} mapping is read too
    banded, flat = summarize(few + many, calibration), summarize(few + many, {"temperature": calibration["temperature"]})
    assert banded["nll"] < flat["nll"]
    assert fit_calibration(few, min_examples=40)["bands"] == {}  # one band only: the type's temperature covers it
    folded = out_of_fold(few + many, min_examples=20)
    assert folded["nll"] == pytest.approx(banded["nll"], abs=0.05) and 0 <= folded["ece"] <= 1
    assert out_of_fold(few[:1]) == {}


def test_intervals_paired_differences_and_chance_corrected_skill():
    low, high = bootstrap_interval([1.0] * 70 + [0.0] * 30)
    assert 0.58 < low < 0.66 and 0.74 < high < 0.82 and bootstrap_interval([1.0]) == (1.0, 1.0)
    assert bootstrap_interval([1.0] * 70 + [0.0] * 30) == (low, high)  # seeded: the same every time

    def answered(right):  # a 4-option question; the model puts 0.85 on its answer
        scores = [math.log(0.85), math.log(0.05), math.log(0.05), math.log(0.05)]
        return {"type": "choice", "source": "s", "scores": scores, "target": [1, 0, 0, 0] if right else [0, 1, 0, 0]}

    base = [{**answered(i % 10 < 6), "id": i} for i in range(300)]
    tuned = [{**answered(i % 10 < 7), "id": i} for i in range(300)]
    diff = paired_difference(item_scores(tuned), item_scores(base))
    assert diff["n"] == 300 and diff["delta_accuracy"] == pytest.approx(0.1)
    assert diff["delta_accuracy_ci_low"] > 0 and diff["delta_nll"] < 0 and diff["delta_nll_ci_high"] < 0
    assert paired_difference(item_scores(tuned), {"x": {"correct": 1.0, "nll": 0.0}}) == {"n": 0.0}

    assert skill(base) == pytest.approx((0.6 - 0.25) / 0.75)  # 60 % on four options is 47 % of the way from chance
    binary = [{"type": "noul", "source": "b", "scores": [1.0, 0.0], "target": [1, 0] if i % 2 else [0, 1]} for i in range(100)]
    assert skill(binary) == 0.0  # 50 % on yes/no is chance
    assert macro_skill(base + binary) == pytest.approx(((0.6 - 0.25) / 0.75 + 0.0) / 2)  # every source counts once
    assert confident_errors(base, level=0.8) == pytest.approx(0.4) and confident_errors(base, level=0.9) == 0.0
    assert skill([]) == 0.0 and confident_errors([]) == 0.0


def test_permutation_report_maps_answers_back_to_the_record_order():
    given = [{"type": "choice", "scores": [5.0, 0.0, 0.0], "target": [1, 0, 0], "order": [0, 1, 2]},
             {"type": "noul", "scores": [1.0, 0.0], "target": [1, 0], "order": [0, 1]}]
    same = [{"type": "choice", "scores": [0.0, 0.0, 5.0], "target": [0, 0, 1], "order": [2, 1, 0]}, given[1]]
    moved = [{"type": "choice", "scores": [5.0, 0.0, 0.0], "target": [0, 0, 1], "order": [2, 1, 0]}, given[1]]
    assert permutation_report([given, same]) == {"permutation_consistency": 1.0, "permutation_accuracy": 1.0}
    assert permutation_report([given, moved]) == {"permutation_consistency": 0.0, "permutation_accuracy": 0.5}
    assert permutation_report([[given[1]]]) == {}


def test_format_answer():
    assert format_answer(Question.parse(NOUL), [0.8, 0.2]) == {"noul": 0.8}
    c = format_answer(Question.parse(CHOICE), [0.1, 0.7, 0.2])
    assert c["choice"] == "negative" and c["confidence"] == 0.7 and c["probabilities"]["neutral"] == 0.2
    s = format_answer(Question.parse(SCORE), [0.0, 0.5, 0.5])
    assert s["score"] == 1 and s["expected"] == pytest.approx(1.5) and s["legend"] == ["negative", "neutral", "positive"]


# -------------------------------------------------------------------- data


def spec(task, **kw):
    base = {"name": "src", "task": task, "data": {"path": "unused"}, "instructions": "Which sentiment?"}
    return SourceSpec.model_validate({**base, **kw})


def test_rows_become_records():
    c = to_record({"text": "great film", "label": 2},
                  spec("classification", label_names=["negative", "neutral", "positive"], descriptions={"positive": "praise"}))
    assert c["question"]["criteria"] == {"negative": None, "neutral": None, "positive": "praise"}
    assert c["answer"] == "positive" and c["source"] == "src" and c["state"] == "great film"

    mapped = to_record({"text": "x", "label": "pos"}, spec("classification", label_names={"pos": "positive", "neg": "negative"}))
    assert mapped["answer"] == "positive" and list(mapped["question"]["criteria"]) == ["positive", "negative"]

    b = to_record({"q": "refund?", "doc": "policy", "label": "True"},
                  spec("binary", state_columns={"query": "q", "passage": "doc"}, descriptions={"true": "relevant", "false": "not"}))
    assert b["state"] == {"query": "refund?", "passage": "policy"} and b["answer"] is True
    assert b["question"]["criteria"] == {"true": "relevant", "false": "not"}

    r = to_record({"text": "x", "label": 4}, spec("rating", levels=["1 star", "2", "3", "4", "5 stars"], offset=1))
    assert r["question"]["type"] == "score" and r["answer"] == 3

    for answer in (1, "B", "Paris"):
        m = to_record({"question": "Capital of France?", "choices": ["Rome", "Paris"], "answer": answer, "context": "geo"},
                      spec("multiple_choice", instructions=None))
        assert m["answer"] == "opt2" and m["question"]["criteria"] == {"opt1": "Rome", "opt2": "Paris"}
        assert m["state"] == {"context": "geo", "question": "Capital of France?"}

    passthrough = to_record({"state": {"a": 1}, "question": NOUL, "answer": False}, spec("records", instructions=None))
    assert passthrough["answer"] is False and passthrough["source"] == "src"

    with pytest.raises(ValueError, match="needs instructions"):
        spec("binary", instructions=None)
    with pytest.raises(ValueError, match="needs levels"):
        spec("rating")


def test_fractional_labels_become_soft_targets():
    s = spec("binary", threshold_soft=True)
    soft = to_record({"text": "borderline", "label": 0.3}, s)
    assert soft["target"] == pytest.approx([0.3, 0.7]) and soft["answer"] is False
    assert "target" not in to_record({"text": "clear", "label": 1.0}, s) and to_record({"text": "clear", "label": 1.0}, s)["answer"] is True
    assert to_record({"text": "none", "label": 0.0}, s)["answer"] is False


def test_heldout_sources_never_reach_training():
    held = spec("binary", role="heldout")
    keys = [content_key("src", f"text {i}") for i in range(400)]
    sides = [split_of(k, held) for k in keys]
    assert set(sides) == {"heldout_eval", "heldout_calibration"} and 150 < sides.count("heldout_eval") < 250


def test_buried_states():
    q = {"type": "noul", "instructions": "Is it positive?", "criteria": None}
    records = [{"id": f"id{i}", "source": "s", "state": f"review number {i}", "question": q, "answer": i % 2 == 0} for i in range(60)]
    originals = {r["id"]: r["state"] for r in records}
    n = bury(records, fraction=0.5, n_records=8, seed=3)
    assert 15 < n < 45
    for record in records:
        if isinstance(record["state"], dict):
            entries = record["state"]["records"]
            assert len(entries) == 8 and [e["id"] for e in entries] == [f"r{k}" for k in range(1, 9)]
            target = record["question"]["instructions"].split(":")[0].rsplit(" ", 1)[-1]  # "... id r5"
            own = next(e["content"] for e in entries if e["id"] == target)
            assert own in originals.values() and record["question"]["instructions"].endswith("Is it positive?")
            assert sum(e["content"] == own for e in entries) == 1  # the record's own state appears once
            assert record["answer"] == (int(own.rsplit(" ", 1)[-1]) % 2 == 0)  # and it is the one the answer is about
        else:
            assert record["state"] == originals[record["id"]]
    again = [{"id": f"id{i}", "source": "s", "state": f"review number {i}", "question": q, "answer": i % 2 == 0} for i in range(60)]
    assert bury(again, 0.5, 8, 3) == n and again == records  # the same rows, neighbours and positions every time
    assert bury(records[:5], 1.0, 8, 3) == 0  # too few rows to bury anything among


def test_a_dataset_that_names_its_classes_supplies_the_options(tmp_path):
    pytest.importorskip("datasets")
    from datasets import ClassLabel, Dataset, Features, Value

    from stagetrain.tracks.decision.data import build_source

    features = Features({"text": Value("string"), "label": ClassLabel(names=["cat", "dog", "bird"])})
    Dataset.from_dict({"text": [f"animal {i}" for i in range(30)], "label": [i % 3 for i in range(30)]}, features=features) \
        .save_to_disk(str(tmp_path / "animals"))
    source = spec("classification", data={"path": str(tmp_path / "animals")}, eval_fraction=0.0, calibration_fraction=0.0)
    splits, counters = build_source(source, seed=0)
    record = splits["train"][0]
    assert list(record["question"]["criteria"]) == ["cat", "dog", "bird"] and record["answer"] in ("cat", "dog", "bird")
    assert len(splits["train"]) == 30 and counters["skipped"] == 0
    with pytest.raises(ValueError, match="outside the 3 label names"):
        to_record({"text": "x", "label": -1}, spec("classification", label_names=["a", "b", "c"]))


def test_paraphrased_instructions_are_stable_per_row():
    s = spec("binary", instructions=["Is it positive?", "Does the review praise the film?"])
    seen = {to_record({"text": f"row {i}", "label": 1}, s)["question"]["instructions"] for i in range(40)}
    assert seen == set(s.instructions)
    assert to_record({"text": "row 1", "label": 1}, s) == to_record({"text": "row 1", "label": 1}, s)


def test_split_is_decided_by_content():
    s = spec("binary", eval_fraction=0.2, calibration_fraction=0.2)
    keys = [content_key("src", f"text {i}") for i in range(2000)]
    counts = {name: sum(split_of(k, s) == name for k in keys) for name in ("train", "eval", "calibration")}
    assert 300 < counts["eval"] < 500 and 300 < counts["calibration"] < 500 and counts["train"] > 1000
    assert [split_of(k, s) for k in keys] == [split_of(k, s) for k in keys]
    with_eval = spec("binary", eval_data={"path": "unused"}, eval_fraction=0.2, calibration_fraction=0.2)
    assert {split_of(k, with_eval) for k in keys} == {"train", "calibration"}


def test_audit_reports_shortcuts():
    q = {"type": "choice", "instructions": "x", "criteria": {"a": None, "b": None, "c": None}}
    skewed = [{"state": "s", "question": q, "answer": "a"}] * 19 + [{"state": "s", "question": q, "answer": "b"}]
    report = audit_source(skewed)
    assert report["majority_baseline"] == 0.95 and report["answer_position"] == {"0": 19, "1": 1}
    assert any("one answer covers" in w for w in report["warnings"])
    assert any("keep shuffle_options on" in w for w in report["warnings"])
    balanced = [{"state": "s", "question": q, "answer": a} for a in "abc" * 10]
    assert audit_source(balanced)["warnings"] == []
    assert audit_source([])["warnings"] == ["no rows"]


def test_mixture_resamples_sources():
    rows = [{"source": "a"}] * 100 + [{"source": "b"}] * 100 + [{"source": "c"}] * 100
    out = apply_mixture(rows, {"a": 0, "b": 2}, 0.5, seed=1)
    counts = {s: sum(r["source"] == s for r in out) for s in "abc"}
    assert counts["a"] == 0 and counts["b"] == 200 and 30 < counts["c"] < 70
    assert apply_mixture(rows, {}, 1.0, seed=1) is rows


# ----------------------------------------------------------------- configs


@pytest.mark.parametrize("path", sorted(Path(__file__).parent.parent.glob("configs/decision/*.yaml")), ids=lambda p: p.name)
def test_shipped_configs_validate(path):
    Pipeline(load_config(path)).validate()
