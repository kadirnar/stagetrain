"""Decision models: a causal LM read at one position, answering typed questions over a state in
one forward pass with no generated tokens. Importing this package registers the track's stages."""

from stagetrain.core.select import register_select  # noqa: E402

from . import data, evaluate, export, soup, train  # noqa: F401

register_select("decision")
