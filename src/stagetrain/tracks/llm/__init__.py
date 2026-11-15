"""The llm track: multi-stage LoRA fine-tuning of a causal language model with Unsloth.

    llm.prepare_data -> llm.check_data -> llm.eval (base, prompt-only) -> llm.sft x N (matrix)
        -> llm.select -> llm.soup / llm.select_checkpoint -> llm.distill -> llm.generate_data
        -> llm.sft / llm.dpo / llm.grpo -> llm.eval -> llm.export

Importing this package registers the stages. Unsloth, torch and TRL are imported only when a
stage runs (`backend.py`), so planning and validating a pipeline needs none of them.
"""

from . import check, evaluate, export, external, generate, prepare, select, soup, train  # noqa: F401
from .backend import BACKENDS, UnslothBackend, get_backend, set_backend

__all__ = ["BACKENDS", "UnslothBackend", "get_backend", "set_backend"]

from stagetrain.core.select import register_select  # noqa: E402

register_select("llm")
