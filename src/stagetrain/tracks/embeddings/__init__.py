"""Embedding (bi-encoder) track:

    prepare_data -> bm25, eval (baselines) -> mine_negatives -> finetune x N -> select -> merge -> eval -> export

Importing this package registers the stages; torch and sentence-transformers are imported only
when a stage runs.
"""

from stagetrain.core.select import register_select  # noqa: E402

from . import bm25, data, evaluate, export, merge, mine, train  # noqa: F401

register_select("embeddings")
