"""Encoder text classifiers, single- and multi-label.

Stages: prepare_data -> mlm_adapt -> finetune -> pseudo_label -> finetune -> calibrate -> eval -> export.
Importing this package registers them; torch and transformers load only when a stage runs.
"""

from . import post, prepare, train  # noqa: F401  (registers the stages)
from .labels import LabelSpace

__all__ = ["LabelSpace"]

from stagetrain.core.select import register_select  # noqa: E402

register_select("classifiers")
