"""The embeddings track end to end on CPU with a tiny random BERT built here: nothing is downloaded."""

import json
import math
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
    plain = {**encoder, "pooling": "mean", "max_seq_length": 32, "prompts": {"query": "query : ", "document": "passage : "}}
    train = {"per_device_train_batch_size": 16, "logging_steps": 2, "save_total_limit": 1, "warmup": 0, "extra": CPU}
    stages = [
        {"name": "data", "kind": "embeddings.prepare_data", "params": {
            "sources": [{"path": str(root / "pairs.jsonl"), "columns": {"query": "question", "positive": "answer"}}],
            "eval_fraction": 0.25, "test_fraction": 0.2,
        }},
        # The two baselines, on the test queries nothing is ever selected on.
        {"name": "bm25", "kind": "embeddings.bm25", "params": {"eval_set": "${data.test_set}"}},
        {"name": "untuned", "kind": "embeddings.eval", "params": {
            "model": str(root / "tiny-bert"), "eval_set": "${data.test_set}", "encoder": plain,
            "compare_to": "${bm25.scores}",
        }},
        {"name": "pretrain", "kind": "embeddings.pretrain", "params": {
            "init_from": str(root / "tiny-bert"),
            "train_data": "${data.train}", "eval_pairs": "${data.eval_pairs}", "eval_set": "${data.eval_set}",
            "encoder": plain,
            "loss": {"name": "cached_mnrl", "mini_batch_size": 8},
            "train": {**train, "num_train_epochs": 2, "learning_rate": 1e-3},
        }},
        {"name": "mine", "kind": "embeddings.mine_negatives", "params": {
            "model": "${pretrain.model}", "train_data": "${data.train}", "corpus": "${data.corpus}",
            "num_negatives": 2, "range_max": 20, "relative_margin": None, "output_scores": True,
            "encoder": encoder,
        }},
        # Listwise distillation of the mined scores plus a small contrastive term, at two temperatures.
        {"name": "finetune", "kind": "embeddings.finetune", "matrix": {"loss.scale": [20, 50]}, "params": {
            "init_from": "${pretrain.model}", "train_data": "${mine.train}", "eval_set": "${data.eval_set}",
            "encoder": encoder, "eval_dims": [16],
            "loss": {"matryoshka_dims": [32, 16, 8], "distill_weight": 1.0, "contrastive_weight": 0.1},
            "train": {**train, "num_train_epochs": 1, "learning_rate": 1e-4},
        }},
        {"name": "best", "kind": "embeddings.select", "params": {"group": "finetune", "metric": "eval/ndcg@10"}},
        {"name": "merge", "kind": "embeddings.merge", "params": {
            "models": ["${best.model}"], "base": "${pretrain.model}", "base_weight": 0.5,
            "eval_set": "${data.eval_set}", "encoder": encoder,
        }},
        {"name": "eval", "kind": "embeddings.eval", "params": {
            "model": "${best.model}", "eval_set": "${data.eval_set}", "dims": [16, 8], "encoder": encoder,
        }, "gate": [
            {"metric": "eval/ndcg@10", "baseline": "pretrain.eval/ndcg@10", "max_drop": 1.0},
            {"metric": "eval/dim16/ndcg@10", "min": 0.0},
        ]},
        {"name": "test", "kind": "embeddings.eval", "params": {
            "model": "${merge.model}", "eval_set": "${data.test_set}", "compare_to": "${untuned.scores}",
            "encoder": encoder,
        }, "gate": {"metric": "eval/delta_ci_high", "min": -1.0}},
        {"name": "export", "kind": "embeddings.export", "params": {
            "model": "${merge.model}", "truncate_dim": 16, "normalize": True, "onnx": True, "encoder": encoder,
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
    assert m.metrics["data/train_pairs"] + m.metrics["data/eval_pairs"] + m.metrics["data/test_pairs"] == n
    assert 0 < m.metrics["data/eval_queries"] < n / 2 and 0 < m.metrics["data/test_queries"] < n / 2
    train = list(read_jsonl(m.artifact("train", run.run_dir)))
    held = list(read_jsonl(m.artifact("eval_pairs", run.run_dir)))
    test = list(read_jsonl(m.artifact("test_pairs", run.run_dir)))
    queries = [{r["query"] for r in part} for part in (train, held, test)]
    assert not queries[0] & queries[1] and not queries[0] & queries[2] and not queries[1] & queries[2]
    corpus = {r["text"] for r in read_jsonl(m.artifact("corpus", run.run_dir))}
    assert corpus == {r["positive"] for r in train}  # mining candidates hold no held-out document
    # each held-out corpus: its own documents plus the training documents as distractors
    assert m.metrics["data/eval_corpus"] == len(train) + len(held)
    assert m.metrics["data/test_corpus"] == len(train) + len(test)


def test_baselines_are_measured_on_the_same_queries(run):
    from stagetrain.tracks.embeddings.data import read_eval_set

    bm25, untuned = run.manifest("bm25"), run.manifest("untuned")
    queries, _, _ = read_eval_set(run.manifest("data").artifact("test_set", run.run_dir))
    # The synthetic questions share their rarest word with their answer, so keyword search is nearly perfect here.
    assert bm25.metrics["eval/ndcg@10"] > 0.9 and bm25.metrics["eval/recall@10"] > 0.9
    assert bm25.metrics["eval/ndcg@10_ci_low"] <= bm25.metrics["eval/ndcg@10"] <= bm25.metrics["eval/ndcg@10_ci_high"]
    scores = {name: json.loads(Path(m.artifact("scores", run.run_dir)).read_text()) for name, m in (("bm25", bm25), ("untuned", untuned))}
    assert set(scores["bm25"]) == set(scores["untuned"]) == set(queries)
    # The per-query scores are the evaluator's own numbers, query by query.
    mean = sum(scores["untuned"].values()) / len(queries)
    assert mean == pytest.approx(untuned.metrics["eval/ndcg@10"], abs=1e-6)
    # A random encoder against BM25, paired on the queries: no better, and the interval says so.
    assert untuned.metrics["eval/delta_ndcg@10"] == pytest.approx(mean - bm25.metrics["eval/ndcg@10"], abs=1e-6)
    assert untuned.metrics["eval/delta_ndcg@10"] < 0 and untuned.metrics["eval/delta_ci_high"] <= 0
    assert untuned.metrics["eval/delta_ci_low"] < untuned.metrics["eval/delta_ndcg@10"]
    assert untuned.metrics["eval/delta_n"] == len(queries)


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


def test_finetune_sweep_eval_and_gate(run):
    runs = [run.manifest("finetune-1"), run.manifest("finetune-2")]
    best, ev = run.manifest("best"), run.manifest("eval")
    assert [m.variant for m in runs] == [{"loss.scale": 20}, {"loss.scale": 50}]
    for fin in runs:
        assert fin.metrics["train/negatives_per_pair"] == 2
        assert "eval/dim16/ndcg@10" in fin.metrics and "eval/loss" not in fin.metrics
    assert best.metrics["eval/ndcg@10"] == max(m.metrics["eval/ndcg@10"] for m in runs)
    for key in ("eval/ndcg@10", "eval/dim16/ndcg@10", "eval/dim8/ndcg@10", "eval/recall@1", "eval/map@100"):
        assert 0.0 <= ev.metrics[key] <= 1.0
    assert ev.metrics["eval/dimension"] == 32
    assert ev.metrics["eval/ndcg@10"] == pytest.approx(best.metrics["eval/ndcg@10"])  # one model, one eval set
    assert [g["passed"] for g in ev.gates] == [True, True]
    assert ev.gates[0]["baseline"] == run.manifest("pretrain").metrics["eval/ndcg@10"]
    saved = json.loads(Path(ev.artifact("metrics", run.run_dir)).read_text())
    assert saved["eval/ndcg@10"] == ev.metrics["eval/ndcg@10"]


def test_merge_is_the_weighted_average_and_the_test_split_reports_a_paired_difference(run):
    from sentence_transformers import SentenceTransformer

    merge, test = run.manifest("merge"), run.manifest("test")
    assert merge.metrics["merge/members"] == 1 and merge.metrics["merge/base_weight"] == 0.5
    assert 0.0 <= merge.metrics["eval/ndcg@10"] <= 1.0
    states = {
        name: SentenceTransformer(run.manifest(name).artifact("model", run.run_dir), device="cpu").state_dict()
        for name in ("merge", "best", "pretrain")
    }
    key = next(k for k, v in states["best"].items() if v.is_floating_point() and not torch.equal(v, states["pretrain"][k]))
    assert torch.allclose(states["merge"][key], (states["best"][key] + states["pretrain"][key]) / 2, atol=1e-6)
    # The reported number: the merged model on the test queries, against the untuned model on the same queries.
    untuned = run.manifest("untuned")
    assert test.metrics["eval/delta_ndcg@10"] == pytest.approx(test.metrics["eval/ndcg@10"] - untuned.metrics["eval/ndcg@10"], abs=1e-6)
    assert test.metrics["eval/delta_ci_low"] <= test.metrics["eval/delta_ndcg@10"] <= test.metrics["eval/delta_ci_high"]
    assert test.gates[0]["passed"]


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
    assert {s.state for s in again.status()} == {CACHED} and len(again.status()) == 12
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


def stage_context(workdir, name):
    from stagetrain.core.stage import StageContext

    return StageContext("embeddings", "tiny", name, workdir / name, workdir, seed=0)


def test_teacher_scores_are_probabilities_unless_logits_are_asked_for(run, workdir):
    from stagetrain.tracks.embeddings.mine import MineNegatives, MineParams

    data, pre = run.manifest("data"), run.manifest("pretrain")
    common = {
        "model": pre.artifact("model", run.run_dir), "train_data": data.artifact("train", run.run_dir),
        "corpus": data.artifact("corpus", run.run_dir), "num_negatives": 2, "range_max": 30, "relative_margin": None,
        "output_scores": True, "cross_encoder": str(workdir / "tiny-bert"), "encoder": {"device": "cpu"},
    }
    torch.manual_seed(7)  # the tiny teacher has no trained head: the same seed gives the same random one
    result = MineNegatives(MineParams(**common)).run(stage_context(workdir, "mine-probability"))
    probabilities = [s for r in read_jsonl(result.artifacts["train"]) for s in r["scores"]]
    assert probabilities and all(0.0 < s < 1.0 for s in probabilities)  # a sigmoid was applied
    assert result.notes == {"scores": "teacher (probability)"}

    torch.manual_seed(7)
    result = MineNegatives(MineParams(teacher_scores="logit", **common)).run(stage_context(workdir, "mine-logit"))
    logits = [s for r in read_jsonl(result.artifacts["train"]) for s in r["scores"]]
    assert result.notes == {"scores": "teacher (logit)"}
    # The same teacher, read before its sigmoid: every probability is the sigmoid of some logit.
    assert sorted(round(float(torch.sigmoid(torch.tensor(s))), 4) for s in logits) == sorted(round(s, 4) for s in probabilities)

    # In probability units a margin of 1 can keep nothing, which is caught before anything runs.
    with pytest.raises(ValueError, match="probabilities in \\(0, 1\\)"):
        MineParams(absolute_margin=1.0, **common)
    assert MineParams(absolute_margin=1.0, teacher_scores="logit", **common).absolute_margin == 1.0
    assert MineParams(absolute_margin=0.1, **common).absolute_margin == 0.1


def test_distillation_losses_and_guided_masking(run, workdir):
    from stagetrain.tracks.embeddings.train import Finetune, FinetuneParams

    pre, mine = run.manifest("pretrain"), run.manifest("mine")
    train = {"per_device_train_batch_size": 16, "max_steps": 2, "logging_steps": 1, "warmup": 0, "extra": CPU}
    common = {
        "init_from": pre.artifact("model", run.run_dir), "train_data": mine.artifact("train", run.run_dir),
        "encoder": {"device": "cpu"}, "train": train,
    }
    for name, loss in (
        ("kl-only", {"name": "kl_div"}),  # listwise distillation alone
        ("kl-raw", {"name": "kl_div", "normalize_scores": False, "teacher_temperature": 0.05}),
        ("mse-term", {"distill_weight": 0.5, "distill_loss": "margin_mse", "distill_scale": 2.0}),
        ("gist", {"name": "gist", "guide": pre.artifact("model", run.run_dir), "distill_weight": 1.0, "contrastive_weight": 0.1}),
        ("cached-gist", {"name": "cached_gist", "guide": pre.artifact("model", run.run_dir), "mini_batch_size": 8}),
        ("cached-symmetric", {"name": "cached_mnrl", "symmetric": True, "mini_batch_size": 8}),
    ):
        one_negative = {"max_negatives": 1} if name == "gist" else {}  # the cached variant reads both negatives
        result = Finetune(FinetuneParams(loss=loss, **one_negative, **common)).run(stage_context(workdir, name))
        assert math.isfinite(result.metrics["train/loss"]) and result.metrics["train/loss"] >= 0, name
        assert (workdir / name / "model" / "modules.json").is_file(), name
    with pytest.raises(ValueError, match="at most one hard negative"):
        guided = {"name": "gist", "guide": pre.artifact("model", run.run_dir)}
        Finetune(FinetuneParams(loss=guided, **common)).run(stage_context(workdir, "gist-two-negatives"))


def test_kl_term_is_on_the_scale_of_a_plain_kl_divergence(run, workdir):
    """The library scales its KL by student_temperature squared; the track undoes that, so that a
    weight of 1 next to a contrastive term means a plain KL divergence."""
    from sentence_transformers import SentenceTransformer

    from stagetrain.tracks.embeddings.train import LossSpec, build_loss

    model = SentenceTransformer(run.manifest("pretrain").artifact("model", run.run_dir), device="cpu")
    texts = [["where is item0x0 of topic0", "where is item1x1 of topic1"], ["topic0 holds item0x0 here", "topic1 holds item1x1 here"],
             ["topic1 holds item1x0 here", "topic0 holds item0x1 here"]]
    features = [model.tokenize(column) for column in texts]
    labels = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    with torch.no_grad():
        loss = build_loss(model, LossSpec(name="kl_div", student_temperature=0.05, teacher_temperature=0.3))(features, labels)
        embeddings = [model(f)["sentence_embedding"] for f in features]
        student = torch.stack([torch.nn.functional.cosine_similarity(embeddings[0], e) for e in embeddings[1:]], dim=1) / 0.05
        expected = torch.nn.functional.kl_div(torch.log_softmax(student, dim=1), torch.softmax(labels / 0.3, dim=1), reduction="batchmean")
    assert loss.item() == pytest.approx(expected.item(), rel=1e-4)


def test_merge_averages_the_runs_of_a_matrix(run, workdir):
    from sentence_transformers import SentenceTransformer

    from stagetrain.tracks.embeddings.merge import Merge, MergeParams

    ctx = stage_context(workdir, "merge-group")
    ctx.run_dir = run.run_dir
    ctx.upstream = {name: run.manifest(name) for name in ("finetune-1", "finetune-2", "pretrain")}
    params = MergeParams(group="finetune", base=run.manifest("pretrain").artifact("model", run.run_dir), base_weight=0.2,
                         encoder={"device": "cpu"})
    result = Merge(params).run(ctx)
    assert result.metrics == {"merge/members": 2.0, "merge/base_weight": 0.2}
    assert result.notes["weights"] == pytest.approx([0.4, 0.4])
    states = {name: SentenceTransformer(run.manifest(name).artifact("model", run.run_dir), device="cpu").state_dict()
              for name in ("finetune-1", "finetune-2", "pretrain")}
    merged = SentenceTransformer(str(result.artifacts["model"]), device="cpu")
    key = next(k for k, v in states["finetune-1"].items() if v.is_floating_point() and not torch.equal(v, states["pretrain"][k]))
    expected = 0.4 * states["finetune-1"][key] + 0.4 * states["finetune-2"][key] + 0.2 * states["pretrain"][key]
    assert torch.allclose(merged.state_dict()[key], expected, atol=1e-6)
    assert merged.prompts["query"] == "query : "  # the merged model keeps the fine-tuned model's prompts

    other = workdir / "other-shape"
    from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

    PreTrainedTokenizerFast.from_pretrained(str(workdir / "tiny-bert")).save_pretrained(other)
    BertModel(BertConfig(vocab_size=len(PreTrainedTokenizerFast.from_pretrained(str(workdir / "tiny-bert"))), hidden_size=16,
                         num_hidden_layers=2, num_attention_heads=2, intermediate_size=32, max_position_embeddings=64)
              ).save_pretrained(other)
    with pytest.raises(ValueError, match="shape|not one architecture"):
        Merge(MergeParams(models=[run.manifest("pretrain").artifact("model", run.run_dir), str(other)],
                          encoder={"device": "cpu", "pooling": "mean"})).run(stage_context(workdir, "merge-bad"))
