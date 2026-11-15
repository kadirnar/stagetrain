"""What the classifier research added, end to end on CPU with the tiny random BERT of
`test_classifiers_pipeline`: a baseline, a learning-rate x seed sweep judged by its seed mean,
early stopping, measured input length, intervals and a paired comparison, a cross-validated
label audit and an ensemble teacher. Nothing is downloaded."""

import json
import random

import pytest
from test_classifiers_pipeline import TOPICS, sentence, tiny_model, train_args  # noqa: F401  (tiny_model is a fixture)

from stagetrain.core import Pipeline, PipelineError, build_config, read_jsonl, write_jsonl

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("sklearn")
pytestmark = pytest.mark.torch

NAMES = list(TOPICS)


def gold_rows(n: int, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    return [{"text": sentence(rng, [NAMES[i % 3]]), "label": NAMES[i % 3]} for i in range(n)]


def test_recipe_baseline_sweep_selection_and_paired_evaluation(tmp_path, tiny_model):  # noqa: F811
    write_jsonl(tmp_path / "gold.jsonl", gold_rows(240))
    cfg = build_config({
        "track": "classifiers", "name": "recipe", "output_dir": str(tmp_path / "runs"), "seed": 3,
        "stages": [
            {"name": "data", "kind": "classifiers.prepare_data", "params": {
                "train": {"path": str(tmp_path / "gold.jsonl")},
                "eval_fraction": 0.15, "calibration_fraction": 0.15, "test_fraction": 0.2}},
            {"name": "baseline", "kind": "classifiers.baseline", "params": {
                "labels": "${data.labels}", "train_data": "${data.train}", "data": "${data.test}",
                "bootstrap_samples": 200}},
            {"name": "finetune", "kind": "classifiers.finetune",
             "matrix": {"train.learning_rate": [3e-3, 1e-6], "seed": [1, 2]},
             "params": {
                 "init_from": tiny_model, "labels": "${data.labels}", "train_data": "${data.train}",
                 "eval_data": "${data.eval}", "early_stopping_patience": 3,
                 "train": train_args(3e-3, 40, eval_steps=10)}},
            {"name": "best", "kind": "classifiers.select", "params": {
                "group": "finetune", "metric": "eval/macro_f1", "average_over": ["seed"]}},
            {"name": "calibration", "kind": "classifiers.calibrate", "params": {
                "model": "${best.model}", "data": "${data.calibration}", "device": "cpu"}},
            {"name": "test", "kind": "classifiers.eval", "params": {
                "model": "${best.model}", "data": "${data.test}", "calibration": "${calibration.calibration}",
                "compare_to": "${baseline.predictions}", "bootstrap_samples": 200, "device": "cpu"},
             "gate": {"metric": "eval/delta_macro_f1_ci_high", "min": 0.0}},  # not worse than the baseline beyond chance
            {"name": "export", "kind": "classifiers.export", "needs": ["test"], "params": {
                "model": "${best.model}", "calibration": "${calibration.calibration}"}},
        ],
    })
    pipeline = Pipeline(cfg)
    m = {x.name: x for x in pipeline.run()}
    run = pipeline.run_dir

    # four disjoint splits; calibration is its own
    data = m["data"].metrics
    assert (data["n_train"], data["n_eval"], data["n_calibration"], data["n_test"]) == (120, 36, 36, 48)
    texts = {s: {r["text"] for r in read_jsonl(run / "data" / f"{s}.jsonl")} for s in ("train", "eval", "calibration", "test")}
    assert sum(len(t) for t in texts.values()) == len(set().union(*texts.values())) == 240

    # the baseline is measured like a model: same metrics, intervals, predictions on disk
    base = m["baseline"].metrics
    assert base["eval/accuracy"] >= 0.9 and base["eval/macro_f1_ci_low"] <= base["eval/macro_f1"] <= base["eval/macro_f1_ci_high"]
    assert base["baseline/n_train"] == 120 and base["baseline/features"] > 20
    first = next(read_jsonl(run / "baseline" / "predictions.jsonl"))
    assert set(first) == {"id", "text", "gold", "pred", "predicted", "probs"} and first["predicted"] == NAMES_SORTED[first["pred"]]

    # the sweep: the learning rate that learns wins on its seed mean; the typical run is handed on
    best = m["best"]
    assert best.notes["setting"] == {"train.learning_rate": 3e-3} and best.notes["winner_runs"] == ["finetune-1", "finetune-2"]
    assert best.metrics["select/configurations"] == 2 and best.metrics["select/mean"] >= 0.9
    assert m["finetune-3"].metrics["eval/macro_f1"] < 0.8  # 1e-6 learns nothing in the same number of steps
    # early stopping: the good runs stop long before their 40 epochs; the best checkpoint is kept
    good = m[best.notes["winner"]]
    assert good.metrics["train/steps"] < 40 * 8 and good.metrics["eval/macro_f1"] >= 0.9

    # the input length was measured, not assumed, and travels with the model
    assert good.metrics["data/max_length"] == 8 and good.metrics["data/truncated_fraction"] == 0.0
    prep = json.loads((run / best.notes["winner"] / "model" / "preprocessing.json").read_text())
    assert prep == {"max_length": 8, "truncation": "head"}
    exported = json.loads((run / "export" / "model" / "classifier_config.json").read_text())
    assert exported["max_length"] == 8 and exported["truncation"] == "head"

    # the test evaluation: intervals, and a paired difference against the baseline on the same rows
    test = m["test"].metrics
    assert test["eval/accuracy_ci_low"] <= test["eval/accuracy"] <= test["eval/accuracy_ci_high"]
    assert test["eval/delta_n"] == 48
    assert test["eval/delta_macro_f1"] == pytest.approx(test["eval/macro_f1"] - base["eval/macro_f1"])
    assert test["eval/delta_macro_f1_ci_low"] <= test["eval/delta_macro_f1"] <= test["eval/delta_macro_f1_ci_high"]
    report = json.loads((run / "test" / "report.json").read_text())
    assert set(report["comparison"]) == {"accuracy", "macro_f1"} and len(report["intervals"]["macro_f1"]) == 2
    assert {s.state for s in Pipeline(cfg).status()} == {"cached"}


NAMES_SORTED = sorted(NAMES)


def test_label_audit_finds_flipped_labels_and_an_ensemble_teaches_a_student(tmp_path, tiny_model):  # noqa: F811
    rows = gold_rows(210, seed=1)
    flipped = {5, 40, 77, 110, 150, 190}
    for i in flipped:
        rows[i]["label"] = NAMES[(NAMES.index(rows[i]["label"]) + 1) % 3]  # a wrong label
    wrong_texts = {rows[i]["text"] for i in flipped}
    write_jsonl(tmp_path / "gold.jsonl", rows)
    rng = random.Random(9)
    write_jsonl(tmp_path / "pool.jsonl", [{"text": sentence(rng, [NAMES[i % 3]])} for i in range(60)])
    finetune = {"init_from": tiny_model, "labels": "${data.labels}", "train_data": "${data.train}",
                "train": train_args(3e-3, 12)}
    cfg = build_config({
        "track": "classifiers", "name": "audit", "output_dir": str(tmp_path / "runs"), "seed": 3,
        "stages": [
            {"name": "data", "kind": "classifiers.prepare_data", "params": {
                "train": {"path": str(tmp_path / "gold.jsonl")}, "unlabeled": {"path": str(tmp_path / "pool.jsonl")},
                "eval_fraction": 0.0}},
            {"name": "cv", "kind": "classifiers.finetune", "matrix": {"fold": [0, 1, 2]},
             "params": {**finetune, "folds": 3}},
            {"name": "audit", "kind": "classifiers.audit_labels", "params": {
                "group": "cv", "labels": "${data.labels}", "train_data": "${data.train}",
                "review_fraction": 0.06, "drop_fraction": 0.02}},
            {"name": "teacher", "kind": "classifiers.finetune", "matrix": {"seed": [1, 2]},
             "params": {**finetune, "train_data": "${audit.train_clean}"}},
            {"name": "pick", "kind": "classifiers.select", "params": {
                "group": "teacher", "metric": "train/loss", "average_over": ["seed"]}},
            {"name": "distill", "kind": "classifiers.pseudo_label", "params": {
                "group": "teacher", "ensemble": "winner", "model": "${pick.model}", "data": "${data.unlabeled}",
                "keep_all": True, "device": "cpu"}},
            {"name": "student", "kind": "classifiers.finetune", "params": {
                **finetune, "train_data": "${audit.train_clean}",
                "extra_train_data": [{"path": "${distill.data}", "weight": 0.5}]}},
        ],
    })
    pipeline = Pipeline(cfg)
    m = {x.name: x for x in pipeline.run()}
    run = pipeline.run_dir

    # every training row is predicted exactly once, by the run that did not train on it
    folds = [list(read_jsonl(run / f"cv-{i}" / "oof.jsonl")) for i in (1, 2, 3)]
    assert sum(len(f) for f in folds) == 210 and all(len(f) > 40 for f in folds)
    assert {r["fold"] for r in folds[1]} == {1} and m["cv-2"].metrics["n_oof"] == len(folds[1])
    assert len({r["text"] for f in folds for r in f}) == 210

    audit = m["audit"].metrics
    assert audit["audit/rows"] == 210 and audit["audit/folds"] == 3 and audit["audit/coverage"] == 1.0
    assert audit["audit/oof_accuracy"] >= 0.9  # an honest estimate: no row was scored by a model that saw it
    suspects = list(read_jsonl(run / "audit" / "suspects.jsonl"))
    assert len(suspects) == round(210 * 0.06) == audit["audit/suspects"]
    assert [s["loss"] for s in suspects] == sorted((s["loss"] for s in suspects), reverse=True)
    found = wrong_texts & {s["text"] for s in suspects}
    assert len(found) >= 5, "the rows with a flipped label are the ones the model is surest are wrong"
    assert all(s["disagrees"] and s["given"] != s["predicted"] for s in suspects if s["text"] in wrong_texts)
    clean = list(read_jsonl(run / "audit" / "train_clean.jsonl"))
    assert len(clean) == 210 - audit["audit/dropped"] == 210 - 4
    assert len(wrong_texts - {r["text"] for r in clean}) == 4  # the four worst, all of them flipped rows

    # the teacher is the two seed runs of the chosen setting, averaged; every pool row is kept
    distill = m["distill"]
    assert distill.metrics["pseudo/teachers"] == 2 and distill.metrics["pseudo/kept"] == 60
    assert [t.split("/")[-2] for t in distill.notes["teachers"]] == ["teacher-1", "teacher-2"]
    soft = list(read_jsonl(run / "distill" / "pseudo.jsonl"))
    assert all(abs(sum(r["probs"]) - 1) < 1e-3 for r in soft) and any(max(r["probs"]) < 0.999 for r in soft)
    assert m["student"].metrics["n_soft"] == 60 and m["student"].metrics["n_train"] == 206 + 60


def test_head_tail_truncation_and_the_logit_adjusted_loss(tmp_path, tiny_model):  # noqa: F811
    from transformers import AutoTokenizer

    from stagetrain.core import StageContext
    from stagetrain.tracks.classifiers.labels import LabelSpace
    from stagetrain.tracks.classifiers.modeling import Predictor, auto_max_length, encode, token_lengths
    from stagetrain.tracks.classifiers.train import Finetune, FinetuneParams

    tokenizer = AutoTokenizer.from_pretrained(tiny_model)
    long_text = "goal " * 30 + "pasta sauce bread"
    head = encode(tokenizer, [{"text": long_text}], max_length=12)[0]["input_ids"]
    both = encode(tokenizer, [{"text": long_text}], max_length=12, truncation="head_tail")[0]
    words = tokenizer.convert_ids_to_tokens(both["input_ids"])
    assert len(head) == len(both["input_ids"]) == len(both["attention_mask"]) == 12
    assert words[0] == "[CLS]" and words[-1] == "[SEP]" and words[-4:-1] == ["pasta", "sauce", "bread"]
    assert words[1:3] == ["goal", "goal"] and "pasta" not in tokenizer.convert_ids_to_tokens(head)
    short = encode(tokenizer, [{"text": "pasta sauce"}], max_length=12, truncation="head_tail")[0]["input_ids"]
    assert short == encode(tokenizer, [{"text": "pasta sauce"}], max_length=12)[0]["input_ids"]  # nothing to cut
    pair = [{"text": long_text, "text_pair": "chip code"}]
    assert len(encode(tokenizer, pair, max_length=12, truncation="head_tail")[0]["input_ids"]) == 12

    assert token_lengths(tokenizer, [{"text": "pasta sauce"}, {"text": long_text}, pair[0]]) == [4, 35, 38]
    assert auto_max_length([4] * 98 + [35, 400], percentile=99, limit=64) == 40  # 35 fits; rounded up to 8s
    assert auto_max_length([4] * 98 + [35, 400], percentile=100, limit=64) == 64  # capped at the model's limit
    assert auto_max_length([], limit=64) == 8 and auto_max_length([3], limit=64) == 8

    # An imbalanced task: 80 / 10 / 10. The stage trains with the prior added to the logits.
    rng = random.Random(4)
    names = sorted(TOPICS)
    counts = [80, 10, 10]
    rows = [{"text": sentence(rng, [names[c]]) + (" the" * 40 if i % 7 == 0 else ""), "label": c}
            for c, n in enumerate(counts) for i in range(n)]
    write_jsonl(tmp_path / "train.jsonl", rows)
    LabelSpace(names, counts=counts).save(tmp_path / "labels.json")
    out = tmp_path / "ft"
    out.mkdir()
    params = FinetuneParams(
        init_from=tiny_model, labels=str(tmp_path / "labels.json"), train_data=str(tmp_path / "train.jsonl"),
        eval_data=str(tmp_path / "train.jsonl"), loss="logit_adjusted", truncation="head_tail", max_length_cap=24,
        classifier_dropout=0.2, pooling="mean", train=train_args(3e-3, 5),
    )
    result = Finetune(params).run(StageContext("classifiers", "p", "ft", out, tmp_path, seed=0))
    assert result.metrics["data/max_length"] == 24 and 0.1 < result.metrics["data/truncated_fraction"] < 0.2
    assert 0 < result.metrics["train/loss"] < 2 and "eval/macro_f1" in result.metrics
    assert result.notes["classifier_dropout"] == "classifier_dropout=0.2"
    assert result.notes["pooling"].startswith("not set")  # BERT has no pooling option; nothing is raised
    predictor = Predictor(result.artifacts["model"], device="cpu")
    assert predictor.preprocessing == {"max_length": 24, "truncation": "head_tail"}
    assert predictor.logits([{"text": long_text}]).shape == (1, 3)


@pytest.mark.parametrize("params, message", [
    ({"early_stopping_patience": 2}, "needs eval_data"),
    ({"fold": 5, "folds": 5}, "fold must be below folds"),
    ({"loss": "logit_adjusted", "class_weights": "balanced"}, "without class_weights"),
    ({"max_length": "longest"}, "max_length"),
])
def test_finetune_rejects_inconsistent_parameters(tmp_path, params, message):
    cfg = build_config({"track": "classifiers", "name": "bad", "output_dir": str(tmp_path), "stages": [
        {"name": "ft", "kind": "classifiers.finetune", "params": {
            "init_from": "x", "labels": "l.json", "train_data": "t.jsonl", **params}}]})
    with pytest.raises(PipelineError, match=message):
        Pipeline(cfg).validate()


def test_dropout_and_pooling_use_the_field_each_architecture_has(tmp_path, tiny_model):  # noqa: F811
    """`classifier_dropout` is ModernBERT's and BERT's name; DeBERTa's head reads `cls_dropout`
    and EuroBERT's has no dropout at all. None of them may make the stage fail."""
    from transformers import (
        AutoTokenizer,
        DebertaV2Config,
        DebertaV2Model,
        EuroBertConfig,
        EuroBertModel,
        ModernBertConfig,
        ModernBertModel,
    )

    from stagetrain.tracks.classifiers.labels import LabelSpace
    from stagetrain.tracks.classifiers.modeling import load_classifier, model_limit

    tokenizer = AutoTokenizer.from_pretrained(tiny_model)
    size = {"vocab_size": len(tokenizer), "hidden_size": 32, "num_hidden_layers": 1, "num_attention_heads": 2,
            "intermediate_size": 64, "max_position_embeddings": 64}
    encoders = {
        "deberta": DebertaV2Model(DebertaV2Config(**size, pooler_hidden_size=32)),
        "modernbert": ModernBertModel(ModernBertConfig(**size, pad_token_id=0)),
        "eurobert": EuroBertModel(EuroBertConfig(**size, pad_token_id=0)),
    }
    for name, encoder in encoders.items():
        tokenizer.save_pretrained(tmp_path / name)
        encoder.save_pretrained(tmp_path / name)
    labels = LabelSpace(["a", "b"])

    model, _, notes = load_classifier(str(tmp_path / "deberta"), labels, dropout=0.3, pooling="mean")
    assert model.dropout.p == 0.3 and notes["classifier_dropout"] == "cls_dropout=0.3"
    assert notes["pooling"].startswith("not set")

    model, tok, notes = load_classifier(str(tmp_path / "modernbert"), labels, dropout=0.1, pooling="mean")
    assert model.config.classifier_pooling == "mean" and model.config.classifier_dropout == 0.1
    assert notes == {"classifier_dropout": "classifier_dropout=0.1", "pooling": "mean"}
    assert load_classifier(str(tmp_path / "modernbert"), labels)[2] == {"pooling": "cls"}  # reported, not changed
    assert model_limit(tok, model.config) == 64

    model, _, notes = load_classifier(str(tmp_path / "eurobert"), labels, dropout=0.1, pooling="mean")
    assert notes["classifier_dropout"].startswith("not set") and model.config.classifier_pooling == "mean"
