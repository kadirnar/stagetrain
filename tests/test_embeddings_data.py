from pathlib import Path

import pytest

from stagetrain.core import Pipeline, load_config
from stagetrain.tracks.embeddings.data import (
    build_eval_set,
    clean_pairs,
    documents,
    read_eval_set,
    split_by_query,
    write_eval_set,
)
from stagetrain.tracks.embeddings.mine import MineParams, score_stats
from stagetrain.tracks.embeddings.model import PromptSpec
from stagetrain.tracks.embeddings.train import LossSpec, order_columns, trim_scores

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
    with pytest.raises(ValueError, match="combines with `mnrl` only"):
        LossSpec(name="cached_mnrl", distill_weight=0.5)
    with pytest.raises(ValueError, match="must match matryoshka_dims"):
        LossSpec(matryoshka_dims=[32, 16], matryoshka_weights=[1.0])
    with pytest.raises(ValueError, match="smaller than num_negatives"):
        MineParams(model="m", train_data="t", num_negatives=8, range_min=10, range_max=15)
    stats = score_stats([[0.9, 0.5, 0.3], [0.7, 0.6, 0.2]])
    assert stats["mine/positive_score"] == pytest.approx(0.8)
    assert stats["mine/negative_score"] == pytest.approx(0.4)
    assert stats["mine/margin"] == pytest.approx(0.4) and stats["mine/min_margin"] == pytest.approx(0.1)
    assert score_stats([]) == {}


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
        "embeddings.eval", "embeddings.export", "embeddings.finetune", "embeddings.mine_negatives",
        "embeddings.prepare_data", "embeddings.pretrain",
    ])


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_example_configs_validate(path):
    pipeline = Pipeline(load_config(path))
    pipeline.validate()
    assert pipeline.config.track == "embeddings"


def test_there_is_an_example_config():
    assert CONFIGS
