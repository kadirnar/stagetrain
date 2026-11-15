"""The embeddings track end to end on CPU with a tiny random BERT built here: nothing is downloaded."""

import json
from pathlib import Path

import pytest

from stagetrain.core import GateFailed, Pipeline, build_config, read_jsonl, write_jsonl
from stagetrain.core.pipeline import CACHED

torch = pytest.importorskip("torch")
pytest.importorskip("sentence_transformers")

pytestmark = pytest.mark.torch

TOPICS, PER_TOPIC = 8, 8
CPU = {"use_cpu": True}


def pairs():
    rows = []
    for t in range(TOPICS):
        for i in range(PER_TOPIC):
            rows.append({"question": f"where is item{t}x{i} of topic{t}", "answer": f"topic{t} holds item{t}x{i} here"})
    rows.append(dict(rows[0]))  # a duplicate, which prepare_data drops
    rows.append({"question": "", "answer": "empty query"})
    return rows


@pytest.fixture(scope="module")
def workdir(tmp_path_factory):
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

    root = tmp_path_factory.mktemp("embeddings")
    words = sorted({w for r in pairs() for text in r.values() for w in text.split()} | {"query", "passage", ":"})
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", *words]
    tok = Tokenizer(models.WordLevel({w: i for i, w in enumerate(vocab)}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)]
    )
    encoder = root / "tiny-bert"
    PreTrainedTokenizerFast(
        tokenizer_object=tok, pad_token="[PAD]", unk_token="[UNK]", cls_token="[CLS]", sep_token="[SEP]",
        model_max_length=64,
    ).save_pretrained(encoder)
    torch.manual_seed(0)
    BertModel(BertConfig(
        vocab_size=len(vocab), hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
        max_position_embeddings=64,
    )).save_pretrained(encoder)
    write_jsonl(root / "pairs.jsonl", pairs())
    return root


def config(root, **over):
    encoder = {"device": "cpu"}
    train = {"per_device_train_batch_size": 16, "logging_steps": 2, "save_total_limit": 1, "warmup": 0, "extra": CPU}
    stages = [
        {"name": "data", "kind": "embeddings.prepare_data", "params": {
            "sources": [{"path": str(root / "pairs.jsonl"), "columns": {"query": "question", "positive": "answer"}}],
            "eval_fraction": 0.25,
        }},
        {"name": "pretrain", "kind": "embeddings.pretrain", "params": {
            "init_from": str(root / "tiny-bert"),
            "train_data": "${data.train}", "eval_pairs": "${data.eval_pairs}", "eval_set": "${data.eval_set}",
            "encoder": {**encoder, "pooling": "mean", "max_seq_length": 32,
                        "prompts": {"query": "query : ", "document": "passage : "}},
            "loss": {"name": "cached_mnrl", "mini_batch_size": 8},
            "train": {**train, "num_train_epochs": 2, "learning_rate": 1e-3},
        }},
        {"name": "mine", "kind": "embeddings.mine_negatives", "params": {
            "model": "${pretrain.model}", "train_data": "${data.train}", "corpus": "${data.corpus}",
            "num_negatives": 2, "range_min": 1, "range_max": 20, "relative_margin": None, "output_scores": True,
            "encoder": encoder,
        }},
        {"name": "finetune", "kind": "embeddings.finetune", "params": {
            "init_from": "${pretrain.model}", "train_data": "${mine.train}", "eval_set": "${data.eval_set}",
            "encoder": encoder, "eval_dims": [16],
            "loss": {"matryoshka_dims": [32, 16, 8], "distill_weight": 0.1, "distill_similarity": "cosine",
                     "distill_scale": 20},
            "train": {**train, "num_train_epochs": 1, "learning_rate": 1e-4},
        }},
        {"name": "eval", "kind": "embeddings.eval", "params": {
            "model": "${finetune.model}", "eval_set": "${data.eval_set}", "dims": [16, 8], "encoder": encoder,
        }, "gate": [
            {"metric": "eval/ndcg@10", "baseline": "pretrain.eval/ndcg@10", "max_drop": 1.0},
            {"metric": "eval/dim16/ndcg@10", "min": 0.0},
        ]},
        {"name": "export", "kind": "embeddings.export", "params": {
            "model": "${finetune.model}", "truncate_dim": 16, "normalize": True, "onnx": True, "encoder": encoder,
        }},
    ]
    for s in stages:
        s.update(over.get(s["name"], {}))
    return build_config({"track": "embeddings", "name": "tiny", "output_dir": str(root / "runs"), "stages": stages})


@pytest.fixture(scope="module")
def run(workdir):
    pipeline = Pipeline(config(workdir))
    pipeline.run()
    return pipeline


def test_prepare_data(run):
    m = run.manifest("data")
    n = TOPICS * PER_TOPIC
    assert m.metrics["data/read"] == n + 2
    assert m.metrics["data/dropped_duplicates"] == 1 and m.metrics["data/dropped_empty"] == 1
    assert m.metrics["data/train_pairs"] + m.metrics["data/eval_pairs"] == n
    assert 0 < m.metrics["data/eval_queries"] < n / 2
    train = list(read_jsonl(m.artifact("train", run.run_dir)))
    held = list(read_jsonl(m.artifact("eval_pairs", run.run_dir)))
    assert not {r["query"] for r in train} & {r["query"] for r in held}
    corpus = {r["text"] for r in read_jsonl(m.artifact("corpus", run.run_dir))}
    assert corpus == {r["positive"] for r in train}  # mining candidates hold no eval document
    assert m.metrics["data/eval_corpus"] == n  # eval documents plus the training documents as distractors


def test_pretrain(run):
    from sentence_transformers import SentenceTransformer

    m = run.manifest("pretrain")
    for key in ("train/loss", "eval/loss", "eval/ndcg@10", "eval/mrr@10", "eval/recall@10", "eval/accuracy@1"):
        assert key in m.metrics, sorted(m.metrics)
    assert not any("precision" in k for k in m.metrics)
    assert 0.0 <= m.metrics["eval/ndcg@10"] <= 1.0
    model = SentenceTransformer(m.artifact("model", run.run_dir), device="cpu")
    assert model.get_embedding_dimension() == 32 and model.max_seq_length == 32
    assert model.prompts["query"] == "query : " and model.prompts["document"] == "passage : "


def test_mine_negatives(run):
    m = run.manifest("mine")
    rows = list(read_jsonl(m.artifact("train", run.run_dir)))
    assert rows and m.metrics["mine/rows_out"] == len(rows)
    assert 0 < m.metrics["mine/kept_fraction"] <= 1
    assert m.metrics["mine/candidates"] == run.manifest("data").metrics["data/corpus"]
    for r in rows:
        assert list(r) == ["query", "positive", "negative_1", "negative_2", "scores"]
        assert len(r["scores"]) == 3 and len({r["positive"], r["negative_1"], r["negative_2"]}) == 3
    assert "mine/margin" in m.metrics


def test_finetune_eval_and_gate(run):
    fin, ev = run.manifest("finetune"), run.manifest("eval")
    assert fin.metrics["train/negatives_per_pair"] == 2
    assert "eval/dim16/ndcg@10" in fin.metrics and "eval/loss" not in fin.metrics
    for key in ("eval/ndcg@10", "eval/dim16/ndcg@10", "eval/dim8/ndcg@10", "eval/recall@1", "eval/map@100"):
        assert 0.0 <= ev.metrics[key] <= 1.0
    assert ev.metrics["eval/dimension"] == 32
    assert ev.metrics["eval/ndcg@10"] == pytest.approx(fin.metrics["eval/ndcg@10"])  # one model, one eval set
    assert [g["passed"] for g in ev.gates] == [True, True]
    assert ev.gates[0]["baseline"] == run.manifest("pretrain").metrics["eval/ndcg@10"]
    saved = json.loads(Path(ev.artifact("metrics", run.run_dir)).read_text())
    assert saved["eval/ndcg@10"] == ev.metrics["eval/ndcg@10"]


def test_export(run):
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.sentence_transformer.modules import Normalize

    m = run.manifest("export")
    assert m.metrics == {"export/dimension": 16.0, "export/onnx": 0.0}
    assert m.notes["onnx"].startswith("skipped:") and m.notes["full_dimension"] == 32
    model = SentenceTransformer(m.artifact("model", run.run_dir), device="cpu")
    vectors = model.encode_query(["where is item0x0 of topic0"], convert_to_tensor=True)
    assert vectors.shape == (1, 16)
    assert isinstance(model[len(model) - 1], Normalize)


def test_resume_and_a_failing_relative_gate(run, workdir):
    again = Pipeline(config(workdir))
    assert [s.state for s in again.status()] == [CACHED] * 6
    strict = {"gate": {"metric": "eval/ndcg@10", "baseline": "pretrain.eval/ndcg@10", "max_drop": -2.0}}
    with pytest.raises(GateFailed, match="dropped more than"):
        Pipeline(config(workdir, eval=strict)).run()  # re-checked on the cached metrics, nothing retrained


def test_mining_that_keeps_nothing_says_why(run, workdir):
    from stagetrain.core.stage import StageContext
    from stagetrain.tracks.embeddings.mine import MineNegatives, MineParams

    data, pre = run.manifest("data"), run.manifest("pretrain")
    out = workdir / "mine-empty"
    out.mkdir(exist_ok=True)
    params = MineParams(
        model=pre.artifact("model", run.run_dir), train_data=data.artifact("train", run.run_dir),
        num_negatives=2, range_min=0, range_max=20, relative_margin=None, max_score=-5.0, encoder={"device": "cpu"},
    )
    ctx = StageContext("embeddings", "tiny", "mine-empty", out, workdir, seed=0)
    with pytest.raises(ValueError, match="mining kept no row"):
        MineNegatives(params).run(ctx)


def test_mining_filters_and_a_cross_encoder_teacher(run, workdir):
    from stagetrain.core.stage import StageContext
    from stagetrain.tracks.embeddings.mine import MineNegatives, MineParams

    data, pre = run.manifest("data"), run.manifest("pretrain")
    common = {
        "model": pre.artifact("model", run.run_dir), "train_data": data.artifact("train", run.run_dir),
        "corpus": data.artifact("corpus", run.run_dir), "range_min": 0, "range_max": 30, "output_scores": True,
        "encoder": {"device": "cpu"},
    }
    # The relative margin is the false-negative filter: no kept negative scores within 2% of its positive.
    out = workdir / "mine-margin"
    result = MineNegatives(MineParams(num_negatives=1, relative_margin=0.02, **common)).run(
        StageContext("embeddings", "tiny", "mine-margin", out, workdir, seed=0)
    )
    rows = list(read_jsonl(result.artifacts["train"]))
    assert rows and all(r["scores"][1] <= r["scores"][0] - abs(r["scores"][0]) * 0.02 + 1e-6 for r in rows)
    assert result.metrics["mine/min_margin"] > 0

    # A cross-encoder teacher (here a tiny random one) rescores the candidates; its scores are what is written.
    out = workdir / "mine-teacher"
    params = MineParams(num_negatives=2, relative_margin=None, cross_encoder=str(workdir / "tiny-bert"), **common)
    result = MineNegatives(params).run(StageContext("embeddings", "tiny", "mine-teacher", out, workdir, seed=0))
    teacher_rows = list(read_jsonl(result.artifacts["train"]))
    assert teacher_rows and all(len(r["scores"]) == 3 for r in teacher_rows)
    bi_encoder = {r["query"]: r["scores"][0] for r in read_jsonl(run.manifest("mine").artifact("train", run.run_dir))}
    assert any(abs(r["scores"][0] - bi_encoder[r["query"]]) > 1e-4 for r in teacher_rows)


def test_checkpoints_resume_only_an_interrupted_run_of_the_same_configuration(tmp_path):
    from stagetrain.tracks.embeddings.train import RUN_MARKER, claim_checkpoints

    ckpt = tmp_path / "checkpoints"
    assert claim_checkpoints(ckpt, "key-a") is False  # nothing yet: a fresh run, marked unfinished
    assert json.loads((ckpt / RUN_MARKER).read_text()) == {"key": "key-a", "finished": False}
    assert claim_checkpoints(ckpt, "key-a") is False  # interrupted before the first checkpoint
    (ckpt / "checkpoint-10").mkdir()
    assert claim_checkpoints(ckpt, "key-a") is True  # interrupted with a checkpoint: resume
    assert (ckpt / "checkpoint-10").is_dir()
    assert claim_checkpoints(ckpt, "key-a", resume=False) is False and not (ckpt / "checkpoint-10").exists()
    (ckpt / "checkpoint-10").mkdir()
    assert claim_checkpoints(ckpt, "key-b") is False  # other parameters: old checkpoints are removed
    assert not (ckpt / "checkpoint-10").exists()
    (ckpt / "checkpoint-10").mkdir()
    (ckpt / RUN_MARKER).write_text(json.dumps({"key": "key-b", "finished": True}))
    assert claim_checkpoints(ckpt, "key-b") is False  # a finished run that is run again starts over
    assert not (ckpt / "checkpoint-10").exists()


def test_an_interrupted_stage_resumes_from_its_checkpoint(run, workdir):
    import shutil

    from sentence_transformers import SentenceTransformer

    from stagetrain.core.stage import StageContext
    from stagetrain.tracks.embeddings.train import RUN_MARKER, Pretrain, PretrainParams

    data = run.manifest("data")
    params = PretrainParams(
        init_from=str(workdir / "tiny-bert"), train_data=data.artifact("train", run.run_dir),
        encoder={"device": "cpu", "pooling": "mean"}, loss={"name": "mnrl"},
        train={"per_device_train_batch_size": 16, "num_train_epochs": 2, "logging_steps": 1, "warmup": 0, "extra": CPU},
    )
    out = workdir / "resume"
    ctx = StageContext("embeddings", "tiny", "resume", out, workdir, seed=0)
    Pretrain(params).run(ctx)
    marker = out / "checkpoints" / RUN_MARKER
    assert json.loads(marker.read_text())["finished"] is True
    last = max(int(p.name.split("-")[1]) for p in (out / "checkpoints").glob("checkpoint-*"))
    texts = ["where is item0x0 of topic0", "topic3 holds item3x1 here"]
    trained = SentenceTransformer(str(out / "model"), device="cpu").encode(texts, convert_to_tensor=True)
    fresh = SentenceTransformer(str(workdir / "tiny-bert"), device="cpu").encode(texts, convert_to_tensor=True)
    assert not torch.allclose(trained, fresh, atol=1e-4)

    # As if the process had died right after the last checkpoint: the rerun continues from it (here with
    # no step left) instead of training the initial model again, and saves the same weights.
    marker.write_text(json.dumps({**json.loads(marker.read_text()), "finished": False}))
    shutil.rmtree(out / "model")
    resumed = Pretrain(params).run(ctx)
    assert resumed.metrics["epoch"] == 2.0 and resumed.metrics["train/samples_per_second"] == 0.0
    assert json.loads(marker.read_text())["finished"] is True
    assert max(int(p.name.split("-")[1]) for p in (out / "checkpoints").glob("checkpoint-*")) == last
    again = SentenceTransformer(str(out / "model"), device="cpu").encode(texts, convert_to_tensor=True)
    assert torch.allclose(again, trained, atol=1e-6)


def test_margin_mse_alone_and_plain_triplets(run, workdir):
    """The two other loss paths, called directly: distillation only, and prepared (query, positive, negative) rows."""
    from stagetrain.core.stage import StageContext
    from stagetrain.tracks.embeddings.train import Finetune, FinetuneParams

    pre, mine = run.manifest("pretrain"), run.manifest("mine")
    train = {"per_device_train_batch_size": 16, "max_steps": 2, "logging_steps": 1, "warmup": 0, "extra": CPU}
    common = {"init_from": pre.artifact("model", run.run_dir), "encoder": {"device": "cpu"}, "train": train}

    out = workdir / "distill-only"
    params = FinetuneParams(train_data=mine.artifact("train", run.run_dir), loss={"name": "margin_mse"}, max_negatives=1, **common)
    result = Finetune(params).run(StageContext("embeddings", "tiny", "distill-only", out, workdir, seed=0))
    assert result.metrics["train/negatives_per_pair"] == 1 and (out / "model" / "modules.json").is_file()

    triplets = workdir / "triplets.jsonl"
    rows = list(read_jsonl(mine.artifact("train", run.run_dir)))
    write_jsonl(triplets, ({"query": r["query"], "positive": r["positive"], "negative": r["negative_1"]} for r in rows))
    out = workdir / "triplets"
    # ... evaluated every step on the retrieval set, keeping the best checkpoint by the in-training metric name.
    best = {**train, "eval_steps": 1, "load_best_model_at_end": True,
            "metric_for_best_model": "eval_ir_cosine_ndcg@10", "greater_is_better": True}
    params = FinetuneParams(
        train_data=str(triplets), loss={"symmetric": True}, batch_sampler="batch_sampler",
        eval_set=run.manifest("data").artifact("eval_set", run.run_dir), **{**common, "train": best},
    )
    result = Finetune(params).run(StageContext("embeddings", "tiny", "triplets", out, workdir, seed=0))
    assert result.metrics["train/negatives_per_pair"] == 1 and "train/loss" in result.metrics
    assert 0.0 <= result.metrics["eval/ndcg@10"] <= 1.0
    with pytest.raises(KeyError, match="no 'scores' column"):
        Finetune(FinetuneParams(train_data=str(triplets), loss={"distill_weight": 1.0}, **common)).run(
            StageContext("embeddings", "tiny", "x", workdir / "x", workdir, seed=0)
        )
