import math
from pathlib import Path

import pytest

from stagetrain.core import Pipeline, load_config
from stagetrain.tracks.embeddings.bm25 import BM25, tokenize
from stagetrain.tracks.embeddings.data import (
    build_eval_set,
    clean_pairs,
    documents,
    read_eval_set,
    split_by_query,
    write_eval_set,
)
from stagetrain.tracks.embeddings.merge import MergeParams, member_weights
from stagetrain.tracks.embeddings.mine import MineParams, score_stats
from stagetrain.tracks.embeddings.model import PromptSpec
from stagetrain.tracks.embeddings.ranking import (
    bootstrap_mean,
    mrr_at_k,
    ndcg_at_k,
    paired_difference,
    recall_at_k,
    summarize_rankings,
)
from stagetrain.tracks.embeddings.train import LossSpec, normalize_scores, order_columns, trim_scores

CONFIGS = sorted((Path(__file__).parent.parent / "configs" / "embeddings").glob("*.yaml"))


def test_clean_pairs():
    rows = [
        {"query": "  what   is\tx ", "positive": "X is a letter", "negative": "Y is a letter"},
        {"query": "What is X", "positive": "x is a letter"},  # the same pair, differently cased
        {"query": "", "positive": "nothing"},
        {"query": "no positive", "positive": None},
        {"query": "same text", "positive": "Same  Text"},
        {"query": "q", "positive": "long enough"},
        {"query": "what is z", "positive": "Z is last", "negative": "z is last"},  # negative equals the positive
        {"query": "what is w", "positive": "w" * 50},
    ]
    clean, stats = clean_pairs(rows, min_chars=2, max_chars=40)
    assert clean == [
        {"query": "what is x", "positive": "X is a letter", "negative": "Y is a letter"},
        {"query": "what is z", "positive": "Z is last"},
    ]
    assert stats == {
        "read": 8, "dropped_empty": 2, "dropped_length": 2, "dropped_identical": 1, "dropped_duplicates": 1,
    }
    kept, _ = clean_pairs([{"query": "same text", "positive": "same text"}], drop_identical=False)
    assert len(kept) == 1


def test_split_keeps_a_query_on_one_side_and_ignores_row_order():
    rows = [{"query": f"q{i % 40}", "positive": f"d{i}"} for i in range(120)]
    train, held = split_by_query(rows, 0.25, seed=3)
    train_q, eval_q = {r["query"] for r in train}, {r["query"] for r in held}
    assert train_q and eval_q and not train_q & eval_q
    assert len(train) + len(held) == len(rows)
    _, again = split_by_query(list(reversed(rows)), 0.25, seed=3)
    assert {r["query"] for r in again} == eval_q
    _, other_seed = split_by_query(rows, 0.25, seed=4)
    assert {r["query"] for r in other_seed} != eval_q
    _, capped = split_by_query(rows, 0.25, seed=3, max_eval_queries=2)
    assert len({r["query"] for r in capped}) == 2 and {r["query"] for r in capped} <= eval_q
    assert split_by_query(rows, 0.0, seed=3) == (rows, [])


def test_eval_set(tmp_path):
    held = [
        {"query": "q a", "positive": "doc a1"},
        {"query": "Q  A", "positive": "doc a2"},  # the same query: a second relevant document
        {"query": "q b", "positive": "doc b", "negative": "doc b hard"},
        {"query": "q c", "positive": "doc a1"},  # a document shared by two queries is stored once
    ]
    queries, corpus, qrels = build_eval_set(held, ["doc b", "train 1", "train 2", "train 3"], max_distractors=2)
    assert queries == {"q0": "q a", "q1": "q b", "q2": "q c"}
    assert corpus == {"d0": "doc a1", "d1": "doc a2", "d2": "doc b", "d3": "doc b hard", "d4": "train 1", "d5": "train 2"}
    assert qrels == {"q0": ["d0", "d1"], "q1": ["d2"], "q2": ["d0"]}
    write_eval_set(tmp_path / "e", queries, corpus, qrels)
    assert read_eval_set(tmp_path / "e") == (queries, corpus, {"q0": {"d0", "d1"}, "q1": {"d2"}, "q2": {"d0"}})
    with pytest.raises(FileNotFoundError, match="not an eval set"):
        read_eval_set(tmp_path)
    assert documents(held) == ["doc a1", "doc a2", "doc b", "doc b hard"]


def test_training_columns():
    cols = ["scores", "negative_2", "positive", "id", "negative_10", "query", "negative_1"]
    assert order_columns(cols, None, False) == ["query", "positive", "negative_1", "negative_2", "negative_10"]
    assert order_columns(cols, 2, True) == ["query", "positive", "negative_1", "negative_2", "scores"]
    assert order_columns(["query", "positive", "negative"], None, False) == ["query", "positive", "negative"]
    with pytest.raises(KeyError, match="no 'scores' column"):
        order_columns(["query", "positive", "negative"], None, True)
    with pytest.raises(KeyError, match="no 'positive' column"):
        order_columns(["query", "text"], None, False)
    assert trim_scores({"scores": [0.9, 0.5, 0.4, 0.3]}, 2) == {"scores": [0.9, 0.5, 0.4]}


def test_specs():
    prompts = PromptSpec(query="query: ", document="passage: ")
    assert prompts.named() == {"query": "query: ", "document": "passage: "}
    assert prompts.for_columns(["query", "positive", "negative_1"]) == {
        "query": "query: ", "positive": "passage: ", "negative_1": "passage: ",
    }
    assert PromptSpec(query="q: ").for_columns(["query", "positive"]) == {"query": "q: "}
    assert LossSpec(distill_weight=0.5).needs_scores and LossSpec(name="margin_mse").needs_scores
    assert not LossSpec(name="cached_mnrl").needs_scores
    # which distillation loss is in use: alone by name, or as a term next to a contrastive loss
    assert LossSpec().distillation is None and LossSpec(name="kl_div").distillation == "kl_div"
    assert LossSpec(distill_weight=1.0).distillation == "kl_div"  # listwise by default
    assert LossSpec(distill_weight=1.0, distill_loss="margin_mse").distillation == "margin_mse"
    assert LossSpec(name="gist", guide="g", distill_weight=1.0).needs_scores
    assert LossSpec().scale == 50.0  # a temperature of 0.02
    with pytest.raises(ValueError, match="combines with `mnrl` or `gist` only"):
        LossSpec(name="cached_mnrl", distill_weight=0.5)
    with pytest.raises(ValueError, match="needs a guide model"):
        LossSpec(name="cached_gist")
    with pytest.raises(ValueError, match="must match matryoshka_dims"):
        LossSpec(matryoshka_dims=[32, 16], matryoshka_weights=[1.0])
    with pytest.raises(ValueError, match="smaller than num_negatives"):
        MineParams(model="m", train_data="t", num_negatives=8, range_min=10, range_max=15)
    # the positive-aware filter alone: no ranks are skipped after it
    default = MineParams(model="m", train_data="t")
    assert (default.range_min, default.relative_margin, default.teacher_scores) == (0, 0.05, "probability")
    with pytest.raises(ValueError, match="removes every candidate"):
        MineParams(model="m", train_data="t", cross_encoder="teacher", absolute_margin=1.0)
    with pytest.raises(ValueError, match="max_score must be above 0"):
        MineParams(model="m", train_data="t", cross_encoder="teacher", max_score=0.0)
    assert MineParams(model="m", train_data="t", absolute_margin=1.0).absolute_margin == 1.0  # no teacher: the miner's own scale
    stats = score_stats([[0.9, 0.5, 0.3], [0.7, 0.6, 0.2]])
    assert stats["mine/positive_score"] == pytest.approx(0.8)
    assert stats["mine/negative_score"] == pytest.approx(0.4)
    assert stats["mine/margin"] == pytest.approx(0.4) and stats["mine/min_margin"] == pytest.approx(0.1)
    assert score_stats([]) == {}


def test_teacher_scores_are_normalised_per_query():
    assert normalize_scores({"scores": [8.0, 2.0, -4.0]}) == {"scores": [1.0, 0.5, 0.0]}  # logits
    assert normalize_scores({"scores": [0.9, 0.6, 0.3]})["scores"] == pytest.approx([1.0, 0.5, 0.0])  # probabilities: the same
    assert normalize_scores({"scores": [0.4, 0.4]}) == {"scores": [0.5, 0.5]}  # no preference, no division by zero


def test_per_query_metrics():
    relevant = {"d1", "d2"}
    assert ndcg_at_k(["d1", "d2", "x"], relevant) == pytest.approx(1.0)
    assert ndcg_at_k(["x", "d1"], {"d1"}) == pytest.approx(1 / math.log2(3))
    assert ndcg_at_k(["x", "y"], relevant) == 0.0 and ndcg_at_k(["d1"], set()) == 0.0
    assert ndcg_at_k(["x"] * 10 + ["d1"], {"d1"}) == 0.0  # found, but past the cutoff
    assert mrr_at_k(["x", "y", "d2"], relevant) == pytest.approx(1 / 3) and mrr_at_k(["x"], relevant) == 0.0
    assert recall_at_k(["d1", "x", "d2"], relevant, 1) == 0.5 and recall_at_k(["d1", "x", "d2"], relevant, 3) == 1.0
    metrics, per_query = summarize_rankings(
        {"q1": ["d1", "x"], "q2": ["x", "d9"]}, {"q1": {"d1"}, "q2": {"d9"}, "q3": {"d5"}}, recall_at=[1, 10]
    )
    assert per_query == {"q1": 1.0, "q2": pytest.approx(1 / math.log2(3)), "q3": 0.0}  # a query with no ranking scores 0
    assert metrics["ndcg@10"] == pytest.approx((1 + 1 / math.log2(3)) / 3)
    assert metrics["accuracy@1"] == pytest.approx(1 / 3) and metrics["recall@10"] == pytest.approx(2 / 3)
    assert metrics["mrr@10"] == pytest.approx(0.5)


def test_bootstrap_interval_and_paired_difference():
    stats = bootstrap_mean([1.0] * 70 + [0.0] * 30)
    assert stats["mean"] == 0.7 and 0.59 < stats["ci_low"] < 0.65 and 0.75 < stats["ci_high"] < 0.81
    assert bootstrap_mean([]) == {"n": 0.0} and bootstrap_mean([0.4])["ci_low"] == 0.4
    assert bootstrap_mean([1.0] * 70 + [0.0] * 30) == stats  # seeded: the same interval every time

    base = {f"q{i}": float(i % 10 < 6) for i in range(300)}
    tuned = {q: 1.0 if i % 10 == 6 else base[q] for i, q in enumerate(base)}  # the same answers plus one more in ten
    diff = paired_difference(tuned, base)
    assert diff["delta"] == pytest.approx(0.1) and diff["wins"] == 30 and diff["losses"] == 0
    assert diff["ci_low"] > 0  # better beyond chance...
    separate = bootstrap_mean(list(tuned.values()))["ci_low"] - bootstrap_mean(list(base.values()))["ci_high"]
    assert separate < 0  # ...which two unpaired intervals of this size cannot show
    assert paired_difference({"a": 1.0}, {"b": 1.0}) == {"n": 0.0}
    assert paired_difference({"a": 1.0, "x": 0.0}, {"a": 0.0, "y": 1.0})["n"] == 1


def test_bm25_ranks_by_rare_shared_words():
    pytest.importorskip("numpy")
    corpus = {
        "d1": "The capital of France is Paris.",
        "d2": "Berlin is the capital of Germany.",
        "d3": "Paris Paris Paris: a film about the city of Paris.",
        "d4": "The stock market fell today.",
    }
    index = BM25(corpus)
    assert tokenize("The Capital, of FRANCE!") == ["the", "capital", "of", "france"]
    assert index.search("capital of france")[0] == "d1"
    assert index.search("capital of germany")[0] == "d2"
    assert set(index.search("paris")) == {"d1", "d3"}  # only documents that share a word are returned
    assert index.search("quantum chromodynamics") == []
    assert len(index.search("the", k=2)) == 2 and len(index.search("the")) == 4  # k cuts the list, nothing else
    assert BM25({}).search("anything") == []
    # length normalisation: with b = 0 the document repeating the word wins outright
    assert BM25(corpus, b=0.0).search("paris")[0] == "d3"


def test_merge_weights():
    assert member_weights(2, None, False, 0.5) == ([0.5, 0.5], 0.0)  # no base: the share is ignored
    weights, base = member_weights(2, [3.0, 1.0], True, 0.2)
    assert weights == pytest.approx([0.6, 0.2]) and base == 0.2 and sum(weights) + base == pytest.approx(1.0)
    assert member_weights(1, None, True, 0.0) == ([1.0], 0.0)
    with pytest.raises(ValueError, match="needs `models` or a `group`"):
        MergeParams()
    with pytest.raises(ValueError, match="weights must match"):
        MergeParams(models=["a", "b"], weights=[1.0])
    with pytest.raises(ValueError, match="weights must match"):
        MergeParams(group="finetune", weights=[1.0])
    assert MergeParams(group="finetune", base="b").base_weight == 0.5


def test_track_imports_without_training_libraries():
    import subprocess
    import sys

    code = (
        "import sys; sys.modules['torch'] = None; sys.modules['sentence_transformers'] = None\n"
        "sys.modules['transformers'] = None; sys.modules['datasets'] = None\n"
        "from stagetrain.core.stage import kinds\n"
        "print(sorted(kinds('embeddings')))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == str([
        "embeddings.bm25", "embeddings.eval", "embeddings.export", "embeddings.finetune", "embeddings.merge",
        "embeddings.mine_negatives", "embeddings.prepare_data", "embeddings.pretrain", "embeddings.select",
    ])


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_example_configs_validate(path):
    pipeline = Pipeline(load_config(path))
    pipeline.validate()
    assert pipeline.config.track == "embeddings"


def test_there_is_an_example_config():
    assert CONFIGS
