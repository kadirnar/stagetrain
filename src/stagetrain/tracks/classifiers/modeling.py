"""Torch side of the classifiers track: encoding, the loss, layer-wise learning rates, inference.

Imported from inside the stages' `run`, never at package import.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import Trainer

from .labels import MULTI, LabelSpace, row_target

LAYER = re.compile(r"\.(?:layer|layers|h|block|blocks)\.(\d+)\.")
NO_DECAY = re.compile(r"(?:^|\.)(?:bias|LayerNorm\.weight|layer_norm\.weight|norm\d*\.weight|.*_norm\.weight)$")


def pick_device(device: str | None = None) -> torch.device:
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #


PREPROCESSING_FILE = "preprocessing.json"
HEAD_SHARE = 0.25  # of a head+tail window: the first quarter and the last three quarters


def _tokenize(tokenizer, rows: Sequence[dict[str, Any]], idx: list[int], with_pair: bool, **kwargs):
    texts = [str(rows[i]["text"]) for i in idx]
    pairs = [str(rows[i]["text_pair"]) for i in idx] if with_pair else None
    return tokenizer(texts, pairs, **kwargs)


def _groups(rows: Sequence[dict[str, Any]]) -> tuple[tuple[list[int], bool], tuple[list[int], bool]]:
    paired = [i for i, r in enumerate(rows) if r.get("text_pair")]
    single = [i for i, r in enumerate(rows) if not r.get("text_pair")]
    return (single, False), (paired, True)


def token_lengths(tokenizer, rows: Sequence[dict[str, Any]]) -> list[int]:
    """How many tokens every row takes when nothing is cut, special tokens included."""
    out = [0] * len(rows)
    for idx, with_pair in _groups(rows):
        if idx:
            enc = _tokenize(tokenizer, rows, idx, with_pair, truncation=False, verbose=False)
            for j, i in enumerate(idx):
                out[i] = len(enc["input_ids"][j])
    return out


def model_limit(tokenizer, config=None) -> int:
    """The longest input the model takes: the tokenizer's limit, or the config's positions."""
    limits = [n for n in (getattr(tokenizer, "model_max_length", None),
                          getattr(config, "max_position_embeddings", None)) if n and n < 1_000_000]
    return int(min(limits)) if limits else 512


def auto_max_length(lengths: Sequence[int], percentile: float = 99.0, limit: int = 512, multiple: int = 8) -> int:
    """The length that fits `percentile` percent of the rows, rounded up to a multiple of
    `multiple` and capped at `limit`."""
    if not lengths:
        return min(limit, multiple)
    ordered = sorted(lengths)
    rank = min(len(ordered) - 1, max(0, -(-len(ordered) * percentile // 100) - 1))
    need = int(ordered[int(rank)])
    return int(min(limit, max(multiple, -(-need // multiple) * multiple)))


def _head_tail(ids: list[int], max_length: int) -> list[int]:
    """Positions kept of an over-long row: the start (with the leading special token) and the end
    (with the closing one). Where the signal sits is not known in advance; the two ends of a
    document carry more of it than its middle."""
    head = max(1, int(max_length * HEAD_SHARE))
    return [*range(head), *range(len(ids) - (max_length - head), len(ids))]


def encode(
    tokenizer, rows: Sequence[dict[str, Any]], max_length: int, truncation: str = "head"
) -> list[dict[str, list[int]]]:
    """Token ids for every row's `text` (and `text_pair`, for the rows that have one), cut to
    `max_length`, not padded. `truncation: head` keeps the start of an over-long text;
    `head_tail` keeps its start and its end. Sentence pairs are always cut from the longer side."""
    out: list[dict[str, list[int]]] = [{} for _ in rows]
    for idx, with_pair in _groups(rows):
        if not idx:
            continue
        both_ends = truncation == "head_tail" and not with_pair
        if both_ends:
            enc = _tokenize(tokenizer, rows, idx, with_pair, truncation=False, verbose=False)
        else:
            enc = _tokenize(tokenizer, rows, idx, with_pair, truncation=True, max_length=max_length)
        keys = [k for k in tokenizer.model_input_names if k in enc]
        for j, i in enumerate(idx):
            row = {k: enc[k][j] for k in keys}
            if both_ends and len(row["input_ids"]) > max_length:
                keep = _head_tail(row["input_ids"], max_length)
                row = {k: [v[t] for t in keep] for k, v in row.items()}
            out[i] = row
    return out


def read_preprocessing(model_dir: str | Path) -> dict[str, Any]:
    """How a model's inputs were cut when it was trained, so later stages read it the same way."""
    import json

    path = Path(model_dir) / PREPROCESSING_FILE
    if not path.is_file():
        return {"max_length": 256, "truncation": "head"}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {"max_length": int(data.get("max_length", 256)), "truncation": data.get("truncation", "head")}


def write_preprocessing(model_dir: str | Path, max_length: int, truncation: str) -> Path:
    import json

    path = Path(model_dir) / PREPROCESSING_FILE
    path.write_text(json.dumps({"max_length": max_length, "truncation": truncation}, indent=2) + "\n", encoding="utf-8")
    return path


class RowDataset(torch.utils.data.Dataset):
    """Encoded rows, each with its target distribution and loss weight."""

    def __init__(self, features: list[dict[str, Any]]):
        self.features = features

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, i: int) -> dict[str, Any]:
        return self.features[i]


def build_dataset(
    tokenizer, rows: Sequence[dict[str, Any]], labels: LabelSpace, max_length: int, smoothing: float = 0.0,
    soft_label_weight: float = 1.0, truncation: str = "head",
) -> RowDataset:
    """Rows as model inputs plus `labels` (the target distribution, see `row_target`) and
    `sample_weight` (a row's own `weight`, times `soft_label_weight` when it carries `probs`)."""
    features = encode(tokenizer, rows, max_length, truncation)
    for feat, row in zip(features, rows, strict=True):
        feat["labels"] = row_target(row, len(labels), labels.problem_type, smoothing)
        weight = float(row.get("weight", 1.0))
        feat["sample_weight"] = weight * (soft_label_weight if row.get("probs") is not None else 1.0)
    return RowDataset(features)


class Collator:
    """Pads a batch to its longest row; targets and weights become float tensors."""

    def __init__(self, tokenizer, pad_to_multiple_of: int | None = None):
        self.tokenizer, self.pad_to_multiple_of = tokenizer, pad_to_multiple_of

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        extra = ("labels", "sample_weight")
        batch = self.tokenizer.pad(
            [{k: v for k, v in f.items() if k not in extra} for f in features],
            padding=True, pad_to_multiple_of=self.pad_to_multiple_of, return_tensors="pt",
        )
        out = dict(batch)
        if "labels" in features[0]:
            out["labels"] = torch.tensor([f["labels"] for f in features], dtype=torch.float32)
            out["sample_weight"] = torch.tensor([f["sample_weight"] for f in features], dtype=torch.float32)
        return out


# --------------------------------------------------------------------------- #
# loss
# --------------------------------------------------------------------------- #


def classification_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
    problem_type: str = "single_label",
    class_weights: torch.Tensor | None = None,
    focal_gamma: float = 0.0,
    logit_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cross-entropy against target distributions `(B, C)`, so gold, smoothed and soft
    (pseudo-labelled) rows share one loss.

    Single-label: `-sum_c w_c t_c (1 - p_c)^gamma log p_c`, averaged with the weights
    `sum_c w_c t_c` like `F.cross_entropy(weight=w)`; with one-hot targets and `gamma=0` it equals
    it. Multi-label: BCE per label with `pos_weight=w` and the focal factor `(1 - p_t)^gamma`,
    averaged over labels. `sample_weight` scales each row.

    `logit_bias` (single-label, `tau * log prior`) is added to the logits inside the loss only:
    the logit-adjusted loss. A frequent class then needs a larger raw logit to win during
    training, so the raw logits used at prediction time are balanced across classes.
    """
    logits, targets = logits.float(), targets.float()
    if logit_bias is not None:
        if problem_type == MULTI:
            raise ValueError("the logit-adjusted loss is for single-label tasks")
        logits = logits + logit_bias.to(logits.device)
    if sample_weight is None:
        sample_weight = torch.ones(len(logits), device=logits.device)
    sample_weight = sample_weight.float()
    if problem_type == MULTI:
        per = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=class_weights, reduction="none")
        if focal_gamma:
            p = torch.sigmoid(logits)
            per = per * (1 - (targets * p + (1 - targets) * (1 - p))).pow(focal_gamma)
        return (per.mean(-1) * sample_weight).sum() / sample_weight.sum().clamp_min(1e-8)
    logp = F.log_softmax(logits, dim=-1)
    per = -targets * logp
    if focal_gamma:
        per = per * (1 - logp.exp()).pow(focal_gamma)
    w = class_weights if class_weights is not None else torch.ones(logits.shape[-1], device=logits.device)
    row_weight = (targets * w).sum(-1) * sample_weight
    return ((per * w).sum(-1) * sample_weight).sum() / row_weight.sum().clamp_min(1e-8)


class ClassifierTrainer(Trainer):
    """`Trainer` whose loss is `loss_fn(logits, targets, sample_weight)`. The targets never reach
    the model's own loss, so the model's `problem_type` is not guessed from their dtype."""

    def __init__(self, *args, loss_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = loss_fn

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = dict(inputs)
        targets = inputs.pop("labels")
        weight = inputs.pop("sample_weight", None)
        outputs = model(**inputs)
        loss = self.loss_fn(outputs.logits, targets, weight)
        return (loss, outputs) if return_outputs else loss


# --------------------------------------------------------------------------- #
# layer-wise learning rates
# --------------------------------------------------------------------------- #


def layer_depths(names: Iterable[str], backbone_prefix: str) -> dict[str, int | None]:
    """Each parameter's depth in the backbone: 0 for the embeddings, `i + 1` for encoder layer
    `i`, `L + 1` for backbone parameters after the last layer (a pooler, a final norm), and `None`
    for the head, i.e. everything outside `backbone_prefix`."""
    names = list(names)
    prefix = f"{backbone_prefix}." if backbone_prefix else ""
    layers = [int(m.group(1)) for n in names if n.startswith(prefix) and (m := LAYER.search(n))]
    top = (max(layers) + 1 if layers else 0) + 1
    out: dict[str, int | None] = {}
    for n in names:
        if not prefix or not n.startswith(prefix):
            out[n] = None
        elif m := LAYER.search(n):
            out[n] = int(m.group(1)) + 1
        else:
            out[n] = 0 if "embed" in n.lower() else top
    return out


def llrd_groups(
    named_parameters: Iterable[tuple[str, Any]],
    lr: float,
    decay: float = 1.0,
    weight_decay: float = 0.01,
    head_lr_multiplier: float = 1.0,
    backbone_prefix: str = "",
) -> list[dict[str, Any]]:
    """Optimizer parameter groups with layer-wise learning-rate decay.

    A backbone parameter at depth `d` of `top` gets `lr * decay ** (top - d)`: the last layer
    trains at `lr * decay`, the embeddings slowest. The head gets `lr * head_lr_multiplier`.
    Biases and normalisation weights get no weight decay. Parameters with `requires_grad=False`
    are left out. Every group carries a `name` for logging.
    """
    params = [(n, p) for n, p in named_parameters if getattr(p, "requires_grad", True)]
    depths = layer_depths([n for n, _ in params], backbone_prefix)
    top = max([d for d in depths.values() if d is not None], default=0)
    groups: dict[tuple[float, float], dict[str, Any]] = {}
    for name, p in params:
        depth = depths[name]
        rate = lr * head_lr_multiplier if depth is None else lr * decay ** (top - depth)
        wd = 0.0 if getattr(p, "ndim", 2) < 2 or NO_DECAY.search(name) else weight_decay
        label = "head" if depth is None else f"depth{depth}"
        group = groups.setdefault((rate, wd), {"params": [], "lr": rate, "weight_decay": wd, "name": label})
        group["params"].append(p)
    return list(groups.values())


def freeze_backbone(model, frozen: bool = True) -> int:
    """Freeze (or unfreeze) everything under the model's base-model prefix; returns how many
    parameters stay trainable."""
    prefix = f"{model.base_model_prefix}."
    for name, p in model.named_parameters():
        if name.startswith(prefix):
            p.requires_grad = not frozen
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --------------------------------------------------------------------------- #
# loading and inference
# --------------------------------------------------------------------------- #


# The config field that holds the classification head's dropout, by architecture family.
DROPOUT_FIELDS = ("classifier_dropout", "cls_dropout")


def log_prior(counts: Sequence[int], tau: float = 1.0) -> list[float]:
    """`tau * log(class frequency)`, the bias of the logit-adjusted loss. A class that never
    occurs gets the smallest observed frequency, so the bias stays finite."""
    import math

    total = sum(counts)
    if total <= 0:
        raise ValueError("the label map carries no class counts; they are written by classifiers.prepare_data")
    floor = min(c for c in counts if c > 0)
    return [tau * math.log(max(c, floor) / total) for c in counts]


def load_classifier(
    init_from: str, labels: LabelSpace, trust_remote_code: bool = False, dropout: float | None = None,
    pooling: str | None = None,
):
    """`(model, tokenizer, notes)`: an encoder with a fresh or matching classification head for
    `labels`. `dropout` and `pooling` are set on the config field the architecture has for them;
    an architecture without one keeps its own behaviour, which `notes` records."""
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(init_from, trust_remote_code=trust_remote_code)
    config = AutoConfig.from_pretrained(
        init_from,
        num_labels=len(labels),
        id2label=dict(enumerate(labels.names)),
        label2id={n: i for i, n in enumerate(labels.names)},
        problem_type="multi_label_classification" if labels.multi else "single_label_classification",
        trust_remote_code=trust_remote_code,
    )
    notes: dict[str, Any] = {}
    if dropout is not None:
        # DeBERTa reads `cls_dropout` with getattr, so the field counts although its config does not declare it.
        declared = [f for f in DROPOUT_FIELDS if hasattr(config, f)]
        field = declared[0] if declared else ("cls_dropout" if config.model_type.startswith("deberta") else None)
        if field is None:
            notes["classifier_dropout"] = f"not set: a {config.model_type} head has no dropout"
        else:
            setattr(config, field, dropout)
            notes["classifier_dropout"] = f"{field}={dropout}"
    if pooling is not None:
        if hasattr(config, "classifier_pooling"):
            config.classifier_pooling = pooling
            notes["pooling"] = pooling
        else:
            notes["pooling"] = f"not set: {config.model_type} has no classifier_pooling option"
    elif hasattr(config, "classifier_pooling"):
        notes["pooling"] = config.classifier_pooling
    model = AutoModelForSequenceClassification.from_pretrained(
        init_from, config=config, ignore_mismatched_sizes=True, trust_remote_code=trust_remote_code
    )
    return model, tokenizer, notes


class Predictor:
    """A trained classifier held in memory: rows in, logits out."""

    def __init__(self, model_dir: str | Path, device: str | None = None, trust_remote_code: bool = False):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=trust_remote_code)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(model_dir), trust_remote_code=trust_remote_code
        )
        self.device = pick_device(device)
        self.model.to(self.device).eval()
        self.collate = Collator(self.tokenizer)
        self.preprocessing = read_preprocessing(model_dir)

    @torch.inference_mode()
    def logits(
        self, rows: Sequence[dict[str, Any]], batch_size: int = 64, max_length: int | None = None,
        truncation: str | None = None,
    ):
        """`(N, C)` float32 numpy logits in the order of `rows`. Rows are batched by length, and
        cut the way the model was trained unless `max_length` / `truncation` say otherwise."""
        import numpy as np

        features = encode(
            self.tokenizer, rows, max_length or self.preprocessing["max_length"],
            truncation or self.preprocessing["truncation"],
        )
        order = sorted(range(len(features)), key=lambda i: len(features[i]["input_ids"]))
        out = np.zeros((len(features), self.model.config.num_labels), dtype=np.float32)
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            batch = {k: v.to(self.device) for k, v in self.collate([features[i] for i in idx]).items()}
            out[idx] = self.model(**batch).logits.float().cpu().numpy()
        return out


def gold_targets(rows: Sequence[dict[str, Any]], labels: LabelSpace):
    """Evaluation targets: class indices `(N,)`, or a 0/1 matrix `(N, C)` for multi-label."""
    from .metrics import targets_array

    return targets_array(rows, len(labels), labels.problem_type)
