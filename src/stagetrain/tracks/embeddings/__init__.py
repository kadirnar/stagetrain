"""Embedding (bi-encoder) track: prepare_data -> pretrain -> mine_negatives -> finetune -> eval -> export.

Importing this package registers the stages; torch and sentence-transformers are imported only
when a stage runs.
"""

from . import data, evaluate, export, mine, train  # noqa: F401
