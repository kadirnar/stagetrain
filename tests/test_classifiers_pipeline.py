"""The classifiers track end to end on CPU: a tiny random BERT and a word-level tokenizer built
here, synthetic topic data, and the real stages run through `Pipeline`. Nothing is downloaded."""

import json
import random

import pytest

from stagetrain.core import Pipeline, build_config, read_jsonl, write_jsonl

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytestmark = pytest.mark.torch

TOPICS = {
    "sports": "goal match team coach league score player stadium".split(),
    "tech": "chip code server cloud laptop pixel robot software".split(),
    "food": "pasta sauce bread cheese soup spice salad oven".split(),
}
NOISE = "the a of and today very some new old big".split()
CPU = {"use_cpu": True}


def sentence(rng: random.Random, topics: list[str]) -> str:
    words = [w for t in topics for w in rng.sample(TOPICS[t], 3)] + rng.sample(NOISE, 3)
    rng.shuffle(words)
    return " ".join(words)


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    """A 2-layer, 32-dim BERT with a masked-LM head and its tokenizer, randomly initialised."""
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import BertConfig, BertForMaskedLM, PreTrainedTokenizerFast

    path = tmp_path_factory.mktemp("tiny-bert")
    specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    words = specials + NOISE + [w for ws in TOPICS.values() for w in ws]
    tok = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B:1 [SEP]:1", special_tokens=[("[CLS]", 2), ("[SEP]", 3)]
    )
    PreTrainedTokenizerFast(
        tokenizer_object=tok, pad_token="[PAD]", unk_token="[UNK]", cls_token="[CLS]", sep_token="[SEP]",
        mask_token="[MASK]",
    ).save_pretrained(path)
    torch.manual_seed(0)
    config = BertConfig(
        vocab_size=len(words), hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
        max_position_embeddings=64,
    )
    BertForMaskedLM(config).save_pretrained(path)
    return str(path)


def train_args(lr: float, epochs: float, **over) -> dict:
    return {"learning_rate": lr, "num_train_epochs": epochs, "per_device_train_batch_size": 16, "warmup": 0.1,
            "logging_steps": 5, "extra": CPU, **over}


def test_single_label_pipeline(tmp_path, tiny_model):
    rng = random.Random(0)
    names = list(TOPICS)
    gold = [{"body": sentence(rng, [names[i % 3]]), "topic": names[i % 3]} for i in range(150)]
    gold += [dict(gold[0]), {"body": gold[1]["body"].upper(), "topic": gold[1]["topic"]}]  # two duplicates
    gold += [{"body": "the of and", "topic": "sports"}, {"body": "the of and", "topic": "food"}]  # a conflict
    pool = [{"text": sentence(rng, [names[i % 3]])} for i in range(90)] + [{"text": gold[5]["body"]}]
    write_jsonl(tmp_path / "gold.jsonl", gold)
    write_jsonl(tmp_path / "pool.jsonl", pool)

    finetune = {"labels": "${data.labels}", "train_data": "${data.train}", "eval_data": "${data.eval}", "max_length": 32}
    infer = {"device": "cpu", "max_length": 32, "batch_size": 32}
    cfg = build_config({
        "track": "classifiers", "name": "single", "output_dir": str(tmp_path / "runs"), "seed": 3,
        "stages": [
            {"name": "data", "kind": "classifiers.prepare_data", "params": {
                "train": {"path": str(tmp_path / "gold.jsonl"), "columns": {"text": "body", "label": "topic"}},
                "unlabeled": {"path": str(tmp_path / "pool.jsonl")},
                "eval_fraction": 0.2, "test_fraction": 0.2}},
            {"name": "adapt", "kind": "classifiers.mlm_adapt", "params": {
                "init_from": tiny_model, "data": ["${data.train}", "${data.unlabeled}"], "max_length": 32,
                "eval_fraction": 0.2, "train": train_args(3e-3, 4)},
             "gate": {"metric": "mlm/loss_drop", "min": 0.0}},
            {"name": "teacher", "kind": "classifiers.finetune", "params": {
                **finetune, "init_from": "${adapt.model}", "llrd": 0.8, "head_lr_multiplier": 3,
                "head_warmup_epochs": 1, "class_weights": "balanced", "label_smoothing": 0.05,
                "train": train_args(3e-3, 25, load_best_model_at_end=True, metric_for_best_model="eval_macro_f1",
                                    greater_is_better=True)},
             "gate": {"metric": "eval/accuracy", "min": 0.8}},
            {"name": "pseudo", "kind": "classifiers.pseudo_label", "params": {
                **infer, "model": "${teacher.model}", "data": "${data.unlabeled}", "min_confidence": 0.6,
                "max_per_class": 20},
             "gate": {"metric": "pseudo/classes_covered", "min": 3}},
            {"name": "student", "kind": "classifiers.finetune", "params": {
                **finetune, "init_from": "${adapt.model}", "loss": "focal", "soft_label_weight": 0.5,
                "extra_train_data": [{"path": "${pseudo.data}", "weight": 0.8}], "train": train_args(3e-3, 25)},
             "gate": {"metric": "eval/accuracy", "baseline": "teacher.eval/accuracy", "max_drop": 0.1}},
            {"name": "calibration", "kind": "classifiers.calibrate", "params": {
                **infer, "model": "${student.model}", "data": "${data.eval}"}},
            {"name": "test", "kind": "classifiers.eval", "params": {
                **infer, "model": "${student.model}", "data": "${data.test}",
                "calibration": "${calibration.calibration}", "save_predictions": True},
             "gate": {"metric": "eval/macro_f1", "min": 0.8}},
            {"name": "export", "kind": "classifiers.export", "needs": ["test"], "params": {
                "model": "${student.model}", "calibration": "${calibration.calibration}", "max_length": 32,
                "onnx": True}},
        ],
    })
    pipeline = Pipeline(cfg)
    manifests = {m.name: m for m in pipeline.run()}
    run = tmp_path / "runs" / "classifiers" / "single"

    # prepare_data: one copy of each duplicate, the conflicting text dropped, no leakage, 60/20/20.
    data = manifests["data"].metrics
    assert data["removed/duplicates"] == 2 and data["removed/conflicts"] == 2 and data["n_labels"] == 3
    assert (data["n_train"], data["n_eval"], data["n_test"]) == (90, 30, 30)
    assert data["n_unlabeled"] == 90 and data["removed/unlabeled_overlap"] == 1
    splits = {s: [r["text"].casefold() for r in read_jsonl(run / "data" / f"{s}.jsonl")] for s in ("train", "eval", "test", "unlabeled")}
    assert not set(splits["train"]) & set(splits["eval"]) and not set(splits["train"]) & set(splits["test"])
    assert not set(splits["unlabeled"]) & (set(splits["train"]) | set(splits["eval"]) | set(splits["test"]))
    labels = json.loads((run / "data" / "labels.json").read_text())
    assert labels["labels"] == ["food", "sports", "tech"] and labels["counts"] == [30, 30, 30]

    # mlm_adapt: the masked-LM loss went down and the checkpoints were cleaned up.
    assert manifests["adapt"].metrics["mlm/loss_drop"] > 0
    assert (run / "adapt" / "model" / "model.safetensors").is_file() and not (run / "adapt" / "checkpoints").exists()

    # finetune: learns the topics; the saved config carries the label names and the problem type.
    teacher = manifests["teacher"].metrics
    assert teacher["eval/accuracy"] >= 0.8 and teacher["warmup/loss"] > 0 and teacher["lr/lowest"] < 3e-3
    config = json.loads((run / "teacher" / "model" / "config.json").read_text())
    assert config["id2label"] == {"0": "food", "1": "sports", "2": "tech"}
    assert config["problem_type"] == "single_label_classification"
    assert (run / "teacher" / "model" / "labels.json").is_file()

    # pseudo_label -> student: soft rows are written and used.
    pseudo = list(read_jsonl(run / "pseudo" / "pseudo.jsonl"))
    assert 0 < len(pseudo) <= 60 and manifests["pseudo"].metrics["pseudo/kept"] == len(pseudo)
    assert all(r["source"] == "pseudo" and abs(sum(r["probs"]) - 1) < 1e-3 and r["confidence"] >= 0.6 for r in pseudo)
    assert manifests["student"].metrics["n_soft"] == len(pseudo)
    assert manifests["student"].notes["train_sources"] == {"gold": 90, "pseudo": len(pseudo)}

    # calibrate: a temperature that does not make the held-out NLL worse.
    cal = json.loads((run / "calibration" / "calibration.json").read_text())
    assert cal["temperature"] > 0 and cal["nll_after"] <= cal["nll_before"] + 1e-9 and cal["thresholds"] is None

    # eval: metrics for gates, a per-class report, a confusion matrix, predictions.
    test = manifests["test"].metrics
    assert test["eval/macro_f1"] >= 0.8 and 0 <= test["eval/ece"] <= 1 and test["eval/n"] == 30
    report = json.loads((run / "test" / "report.json").read_text())
    assert set(report["per_class"]) == {"food", "sports", "tech"} and report["calibration"]["temperature"] == cal["temperature"]
    matrix = json.loads((run / "test" / "confusion_matrix.json").read_text())["matrix"]
    assert sum(map(sum, matrix)) == 30 and sum(matrix[i][i] for i in range(3)) == round(test["eval/accuracy"] * 30)
    assert len(list(read_jsonl(run / "test" / "predictions.jsonl"))) == 30

    # export: a self-contained directory; ONNX is skipped with a note when `onnx` is not installed.
    export = manifests["export"]
    out = run / "export" / "model"
    cfg_json = json.loads((out / "classifier_config.json").read_text())
    assert cfg_json["labels"] == ["food", "sports", "tech"] and cfg_json["temperature"] == cal["temperature"]
    assert {"model.safetensors", "tokenizer.json", "labels.json", "calibration.json"} <= {f.name for f in out.iterdir()}
    import importlib.util

    if importlib.util.find_spec("onnx") is None or importlib.util.find_spec("onnxscript") is None:
        assert export.metrics["export/onnx"] == 0 and "not installed" in export.notes["onnx"]
    else:
        assert export.metrics["export/onnx"] == 1 and (out / "model.onnx").is_file(), export.notes

    # The exported directory is usable on its own.
    from stagetrain.tracks.classifiers.inference import Classifier

    clf = Classifier(out, device="cpu")
    got = clf.predict(["goal match team coach the", "pasta sauce bread cheese a", "chip code server cloud of"])
    assert [g["label"] for g in got] == ["sports", "food", "tech"]
    assert all(abs(sum(g["probabilities"].values()) - 1) < 1e-6 and g["confidence"] == max(g["probabilities"].values()) for g in got)

    # A second run is a no-op.
    again = Pipeline(cfg)
    assert {s.state for s in again.status()} == {"cached"}


def test_multi_label_pipeline(tmp_path, tiny_model):
    rng = random.Random(1)
    names = list(TOPICS)
    combos = [["sports"], ["tech"], ["food"], ["sports", "tech"], ["tech", "food"], ["sports", "food"]]
    rows = [{"text": sentence(rng, combos[i % 6]), "labels": ",".join(combos[i % 6])} for i in range(180)]
    write_jsonl(tmp_path / "train.jsonl", rows[:132])
    write_jsonl(tmp_path / "eval.jsonl", rows[132:] + [rows[0]])  # one row leaks from train
    infer = {"device": "cpu", "max_length": 32}
    cfg = build_config({
        "track": "classifiers", "name": "multi", "output_dir": str(tmp_path / "runs"), "seed": 5,
        "stages": [
            {"name": "data", "kind": "classifiers.prepare_data", "params": {
                "problem_type": "multi_label", "labels": names,
                "train": {"path": str(tmp_path / "train.jsonl")}, "eval": {"path": str(tmp_path / "eval.jsonl")},
                "test_fraction": 0.0}},
            {"name": "finetune", "kind": "classifiers.finetune", "params": {
                "init_from": tiny_model, "labels": "${data.labels}", "train_data": "${data.train}",
                "eval_data": "${data.eval}", "max_length": 32, "class_weights": "balanced", "loss": "focal",
                "focal_gamma": 1.0, "train": train_args(3e-3, 30)},
             "gate": {"metric": "eval/micro_f1", "min": 0.7}},
            {"name": "calibration", "kind": "classifiers.calibrate", "params": {
                **infer, "model": "${finetune.model}", "data": "${data.eval}"}},
            {"name": "test", "kind": "classifiers.eval", "params": {
                **infer, "model": "${finetune.model}", "data": "${data.eval}",
                "calibration": "${calibration.calibration}"}},
            {"name": "export", "kind": "classifiers.export", "params": {
                "model": "${finetune.model}", "calibration": "${calibration.calibration}", "max_length": 32}},
        ],
    })
    manifests = {m.name: m for m in Pipeline(cfg).run()}
    run = tmp_path / "runs" / "classifiers" / "multi"

    data = manifests["data"].metrics
    assert data["removed/leaked"] == 1 and data["n_train"] == 131 and data["n_eval"] == 49 and data["n_test"] == 0
    labels = json.loads((run / "data" / "labels.json").read_text())
    assert labels["problem_type"] == "multi_label" and labels["labels"] == names  # the given order is kept
    assert all(isinstance(r["labels"], list) for r in read_jsonl(run / "data" / "train.jsonl"))

    config = json.loads((run / "finetune" / "model" / "config.json").read_text())
    assert config["problem_type"] == "multi_label_classification"
    assert manifests["finetune"].metrics["eval/micro_f1"] >= 0.7

    cal = json.loads((run / "calibration" / "calibration.json").read_text())
    assert len(cal["thresholds"]) == 3 and all(0 < t < 1 for t in cal["thresholds"])
    test = manifests["test"].metrics
    # Thresholds were fitted on this very split, so they cannot do worse than 0.5 did.
    assert test["eval/micro_f1"] >= manifests["finetune"].metrics["eval/micro_f1"] - 1e-9
    assert "eval/hamming_loss" in test and not (run / "test" / "confusion_matrix.json").exists()

    from stagetrain.tracks.classifiers.inference import Classifier

    got = Classifier(run / "export" / "model", device="cpu").predict(["goal match team chip code server the"])
    assert set(got[0]["labels"]) == {"sports", "tech"} and set(got[0]["probabilities"]) == set(names)


def test_prepare_data_reads_class_label_features(tmp_path):
    """Hub datasets often store labels as `ClassLabel` ids: the names come from the features."""
    from datasets import ClassLabel, Dataset, Features, List, Value

    from stagetrain.core import StageContext
    from stagetrain.tracks.classifiers.prepare import PrepareData, PrepareParams

    def prepare(name, features, rows, **params):
        Dataset.from_list(rows, features=features).save_to_disk(str(tmp_path / name))
        out = tmp_path / f"{name}-out"
        out.mkdir()
        ctx = StageContext("classifiers", "p", "data", out, tmp_path, seed=0)
        result = PrepareData(PrepareParams(train={"path": str(tmp_path / name)}, eval_fraction=0.0, **params)).run(ctx)
        return result, list(read_jsonl(out / "train.jsonl")), json.loads((out / "labels.json").read_text())

    single = Features({"text": Value("string"), "label": ClassLabel(names=["World", "Sports"])})
    result, rows, labels = prepare("single", single, [{"text": f"t{i}", "label": i % 2} for i in range(6)])
    assert labels["labels"] == ["Sports", "World"]  # inferred names sort alphabetically
    assert [r["label"] for r in rows] == [1, 0, 1, 0, 1, 0] and result.metrics["n_train"] == 6
    assert set(result.artifacts) == {"labels", "train"}  # no eval, test or unlabeled file is promised

    multi = Features({"text": Value("string"), "labels": List(ClassLabel(names=["joy", "anger", "fear"]))})
    data = [{"text": "a", "labels": [0, 2]}, {"text": "b", "labels": [1]}, {"text": "c", "labels": []}]
    result, rows, labels = prepare("multi", multi, data, problem_type="multi_label", labels=["joy", "anger", "fear"])
    assert [r["labels"] for r in rows] == [[0, 2], [1], []] and labels["counts"] == [1, 1, 1]
    assert labels["class_weights"] == [2.0, 2.0, 2.0]  # negatives / positives

    with pytest.raises(KeyError, match="no column 'label'"):
        prepare("wrong", multi, data)


def test_finetune_rejects_rows_without_labels(tmp_path, tiny_model):
    from stagetrain.tracks.classifiers.labels import LabelSpace
    from stagetrain.tracks.classifiers.train import read_rows

    write_jsonl(tmp_path / "bad.jsonl", [{"text": "no label here"}])
    with pytest.raises(KeyError, match="neither 'label' nor 'probs'"):
        read_rows(str(tmp_path / "bad.jsonl"), LabelSpace(["a", "b"]))


# --------------------------------------------------------------------------- #
# single stages
# --------------------------------------------------------------------------- #


def labelled_files(tmp_path, n: int = 48):
    from stagetrain.tracks.classifiers.labels import LabelSpace

    rng = random.Random(2)
    names = sorted(TOPICS)
    write_jsonl(tmp_path / "train.jsonl", [{"text": sentence(rng, [names[i % 3]]), "label": i % 3} for i in range(n)])
    LabelSpace(names, class_weights=[1.0, 1.0, 1.0]).save(tmp_path / "labels.json")
    return {"labels": str(tmp_path / "labels.json"), "train_data": str(tmp_path / "train.jsonl"), "max_length": 32}


def stage_context(tmp_path, name: str):
    from stagetrain.core import StageContext

    out = tmp_path / name
    out.mkdir(exist_ok=True)
    return StageContext("classifiers", "p", name, out, tmp_path, seed=0)


def test_finetune_resumes_only_when_the_pipeline_says_so(tmp_path, tiny_model):
    import shutil

    from stagetrain.tracks.classifiers.train import Finetune, FinetuneParams

    files = labelled_files(tmp_path)
    params = FinetuneParams(init_from=tiny_model, llrd=0.8, keep_checkpoints=True,
                            train=train_args(3e-3, 2, save_steps=3, save_total_limit=None), **files)
    ctx = stage_context(tmp_path, "ft")
    first = Finetune(params).run(ctx)
    checkpoints = ctx.out_dir / "checkpoints"
    assert "resumed_from" not in first.notes
    assert sorted(p.name for p in checkpoints.glob("checkpoint-*")) == ["checkpoint-3", "checkpoint-6"]

    # An interrupted attempt of the same run (the pipeline sets ctx.resumed): continue from the
    # last checkpoint and train the remaining steps.
    shutil.rmtree(checkpoints / "checkpoint-6")
    ctx.resumed = True
    resumed = Finetune(params).run(ctx)
    assert resumed.notes["resumed_from"] == "checkpoint-3" and (checkpoints / "checkpoint-6").is_dir()
    assert "warmup/loss" not in resumed.metrics

    # `resume: false` starts over even then, and so does a directory that is not a resumed run.
    (checkpoints / "checkpoint-99").mkdir()
    fresh = Finetune(params.model_copy(update={"resume": False})).run(ctx)
    assert "resumed_from" not in fresh.notes and not (checkpoints / "checkpoint-99").exists()
    ctx.resumed = False
    (checkpoints / "checkpoint-99").mkdir()
    assert "resumed_from" not in Finetune(params).run(ctx).notes and not (checkpoints / "checkpoint-99").exists()


def test_finetune_on_a_modernbert_style_encoder(tmp_path, tiny_model):
    """ModernBERT names its backbone `model.` and its layers `model.layers.N.`: layer-wise rates,
    the head warm-up and the custom loss work on it as on BERT."""
    from transformers import AutoTokenizer, ModernBertConfig, ModernBertForMaskedLM

    from stagetrain.tracks.classifiers.modeling import layer_depths
    from stagetrain.tracks.classifiers.post import EvalParams, Evaluate
    from stagetrain.tracks.classifiers.train import Finetune, FinetuneParams

    tokenizer = AutoTokenizer.from_pretrained(tiny_model)
    base = tmp_path / "tiny-modernbert"
    tokenizer.save_pretrained(base)
    torch.manual_seed(0)
    ModernBertForMaskedLM(ModernBertConfig(
        vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64, num_hidden_layers=3, num_attention_heads=2,
        max_position_embeddings=64, pad_token_id=0, cls_token_id=2, sep_token_id=3, bos_token_id=2, eos_token_id=3,
    )).save_pretrained(base)

    files = labelled_files(tmp_path, n=90)
    params = FinetuneParams(init_from=str(base), eval_data=files["train_data"], llrd=0.8, head_lr_multiplier=2,
                            head_warmup_epochs=1, classifier_dropout=0.1, train=train_args(3e-3, 20), **files)
    result = Finetune(params).run(stage_context(tmp_path, "ft"))
    assert result.metrics["eval/accuracy"] >= 0.8
    assert result.metrics["lr/lowest"] == pytest.approx(3e-3 * 0.8 ** 4)  # embeddings, 4 levels below the final norm

    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(result.artifacts["model"])
    assert model.base_model_prefix == "model" and model.config.classifier_dropout == 0.1
    depths = layer_depths([n for n, _ in model.named_parameters()], model.base_model_prefix)
    assert set(depths.values()) == {0, 1, 2, 3, 4, None}
    assert {n.split(".")[0] for n, d in depths.items() if d is None} == {"head", "classifier"}

    ev = Evaluate(EvalParams(model=str(result.artifacts["model"]), data=files["train_data"], device="cpu", max_length=32))
    assert ev.run(stage_context(tmp_path, "ev")).metrics["eval/accuracy"] == pytest.approx(result.metrics["eval/accuracy"])


def test_mlm_chunks_long_texts_and_encodes_pairs(tiny_model):
    from transformers import AutoTokenizer

    from stagetrain.tracks.classifiers.modeling import encode
    from stagetrain.tracks.classifiers.train import mlm_features

    tokenizer = AutoTokenizer.from_pretrained(tiny_model)
    long_text = " ".join(["goal match team coach"] * 20)  # 80 words
    chunks = mlm_features(tokenizer, [long_text, "pasta sauce"], max_length=16, chunk=True, stride=4)
    assert len(chunks) > 5 and all(len(c["input_ids"]) <= 16 for c in chunks)
    assert sum(len(c["input_ids"]) - sum(c["special_tokens_mask"]) for c in chunks) >= 82  # no word is lost
    assert len(mlm_features(tokenizer, [long_text, "pasta sauce"], max_length=16, chunk=False, stride=4)) == 2

    rows = [{"text": "goal match", "text_pair": "pasta sauce bread"}, {"text": "chip"}]
    enc = encode(tokenizer, rows, max_length=32)
    ids = tokenizer.convert_tokens_to_ids
    assert enc[0]["input_ids"] == ids(["[CLS]", "goal", "match", "[SEP]", "pasta", "sauce", "bread", "[SEP]"])
    assert enc[1]["input_ids"] == ids(["[CLS]", "chip", "[SEP]"]) and set(enc[0]) == set(enc[1])
    assert len(encode(tokenizer, [{"text": long_text}], max_length=10)[0]["input_ids"]) == 10
    assert encode(tokenizer, [], max_length=10) == []


def test_predictor_keeps_row_order(tmp_path, tiny_model):
    import numpy as np

    from stagetrain.tracks.classifiers.modeling import Predictor
    from stagetrain.tracks.classifiers.train import Finetune, FinetuneParams

    files = labelled_files(tmp_path)
    model = Finetune(FinetuneParams(init_from=tiny_model, train=train_args(3e-3, 1), **files)).run(
        stage_context(tmp_path, "ft")).artifacts["model"]
    predictor = Predictor(model, device="cpu")
    rng = random.Random(4)
    rows = [{"text": " ".join(rng.choices(NOISE + TOPICS["tech"], k=rng.randint(1, 20)))} for _ in range(23)]
    batched = predictor.logits(rows, batch_size=4, max_length=32)  # sorted by length internally
    single = np.concatenate([predictor.logits([r], batch_size=1, max_length=32) for r in rows])
    assert batched.shape == (23, 3) and np.allclose(batched, single, atol=1e-5)
