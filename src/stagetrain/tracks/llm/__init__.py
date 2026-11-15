"""The llm track: multi-stage LoRA fine-tuning of a causal language model with Unsloth.

    llm.prepare_data -> llm.cpt -> llm.sft (one or more) -> llm.dpo -> llm.eval -> llm.export

Importing this package registers the stages. Unsloth, torch and TRL are imported only when a
stage runs (`backend.py`), so planning and validating a pipeline needs none of them.
"""

from . import evaluate, export, prepare, train  # noqa: F401
from .backend import UnslothBackend, get_backend, set_backend

__all__ = ["UnslothBackend", "get_backend", "set_backend"]
