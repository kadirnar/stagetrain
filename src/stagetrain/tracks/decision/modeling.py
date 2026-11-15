"""Batches, the answer-slot readout, the loss, and the scoring loop of a decision model.

A decision model is a causal LM read at one position. Rows are left-padded, so the answer slot
is the last position of every row and the LM head runs there only (`logits_to_keep=1`).
"""

from __future__ import annotations

import inspect
import json
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from .prompt import PromptStyle, Verbalizer, encode_prompt
from .schema import CHOICE, SCORE, TYPES, Record

META_FILE = "decision.json"


class LossConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label_smoothing: float = 0.0  # on yes/no and choice targets
    ordinal_sigma: float = 0.0  # score targets are blurred over neighbouring levels with this width
    vocab_mass_weight: float = 0.1  # weight of -log P(any option token) over the whole vocabulary


class AugmentConfig(BaseModel):
    """Train-time variation of the prompt; each one removes a shortcut the data could offer."""

    model_config = ConfigDict(extra="forbid")

    shuffle_options: bool = True  # a choice's options in a new order every time it is seen
    description_dropout: float = 0.0  # chance that a choice shows option names without descriptions
    mix_option_format: bool = False  # alternate between "name: description" and description-only lines
    mix_state_format: bool = False  # alternate between JSON and sectioned states


# --------------------------------------------------------------------------- #
# batches
# --------------------------------------------------------------------------- #


class DecisionCollator:
    """Turns records into tensors: left-padded prompts, each option's token ids, and the target
    distribution over the options."""

    def __init__(
        self,
        tokenizer,
        style: PromptStyle,
        max_length: int | None,
        augment: AugmentConfig | None = None,
        seed: int = 0,
        blind: bool = False,
        permute_seed: int | None = None,
        with_meta: bool = False,
    ):
        self.tokenizer = tokenizer
        self.verbalizer = Verbalizer(tokenizer)
        self.style = style
        self.max_length = max_length
        self.augment = augment
        self.seed = seed
        self.blind = blind  # hide the state: what the question alone gives away
        self.permute_seed = permute_seed  # evaluation under a fixed reshuffle of every choice's options
        self.with_meta = with_meta
        self._rng: random.Random | None = None
        self.truncated = 0

    def rng(self) -> random.Random:
        if self._rng is None:
            from torch.utils.data import get_worker_info

            worker = get_worker_info()
            self._rng = random.Random(self.seed + (worker.id + 1) * 100_003 if worker else self.seed)
        return self._rng

    def encode(self, record: Record) -> dict[str, Any]:
        question, target, style = record.question, list(record.target), self.style
        order = list(range(len(target)))
        if question.type == CHOICE:
            if self.permute_seed is not None:
                random.Random(f"{self.permute_seed}:{record.id}").shuffle(order)
            elif self.augment and self.augment.shuffle_options:
                self.rng().shuffle(order)
            if order != sorted(order):
                question = question.permuted(order)
                target = [target[i] for i in order]
        if self.augment:
            rng = self.rng()
            if question.type == CHOICE and rng.random() < self.augment.description_dropout:
                question = question.without_descriptions()
            if self.augment.mix_option_format and rng.random() < 0.5:
                style = style.model_copy(update={"option_format": "desc"})
            if self.augment.mix_state_format and rng.random() < 0.5:
                style = style.model_copy(update={"state_format": "sections"})
        codes, groups = self.verbalizer.groups(question)
        ids, truncated = encode_prompt(
            self.tokenizer, None if self.blind else record.state, question, codes, style, self.max_length
        )
        self.truncated += truncated
        return {
            "ids": ids, "groups": groups, "target": target, "type": question.type, "order": order,
            "id": record.id, "source": record.source, "tokens": len(ids),
        }

    def __call__(self, rows: Sequence[Any]) -> dict[str, Any]:
        import torch

        records = [r if isinstance(r, Record) else Record.parse(json.loads(r["record"]) if "record" in r else r)
                   for r in rows]
        enc = [self.encode(r) for r in records]
        width = max(len(e["ids"]) for e in enc)
        n_options = max(len(e["groups"]) for e in enc)
        n_forms = max(len(g) for e in enc for g in e["groups"])
        pad = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        input_ids = torch.full((len(enc), width), pad, dtype=torch.long)
        attention = torch.zeros((len(enc), width), dtype=torch.long)
        option_ids = torch.zeros((len(enc), n_options, n_forms), dtype=torch.long)
        option_mask = torch.zeros((len(enc), n_options, n_forms), dtype=torch.bool)
        targets = torch.zeros((len(enc), n_options), dtype=torch.float32)
        for i, e in enumerate(enc):
            input_ids[i, width - len(e["ids"]):] = torch.tensor(e["ids"], dtype=torch.long)
            attention[i, width - len(e["ids"]):] = 1
            for j, group in enumerate(e["groups"]):
                option_ids[i, j, : len(group)] = torch.tensor(group, dtype=torch.long)
                option_mask[i, j, : len(group)] = True
            targets[i, : len(e["target"])] = torch.tensor(e["target"])
        batch: dict[str, Any] = {
            "input_ids": input_ids, "attention_mask": attention, "option_ids": option_ids,
            "option_mask": option_mask, "targets": targets,
            "qtype": torch.tensor([TYPES.index(e["type"]) for e in enc], dtype=torch.long),
        }
        if self.with_meta:
            batch["meta"] = [{k: e[k] for k in ("id", "source", "type", "order", "target", "tokens")} for e in enc]
        return batch


# --------------------------------------------------------------------------- #
# readout and loss
# --------------------------------------------------------------------------- #

_KEEPS_LOGITS: dict[type, bool] = {}


def _accepts_logits_to_keep(model) -> bool:
    base = getattr(model, "module", model)  # DataParallel / DistributedDataParallel
    base = base.get_base_model() if hasattr(base, "get_base_model") else base
    kind = type(base)
    if kind not in _KEEPS_LOGITS:
        _KEEPS_LOGITS[kind] = "logits_to_keep" in inspect.signature(base.forward).parameters
    return _KEEPS_LOGITS[kind]


def slot_logits(model, input_ids, attention_mask):
    """Vocabulary logits at the answer slot, `(batch, vocab)` in fp32."""
    kwargs = {"input_ids": input_ids, "attention_mask": attention_mask, "use_cache": False}
    if _accepts_logits_to_keep(model):
        kwargs["logits_to_keep"] = 1
    return model(**kwargs).logits[:, -1, :].float()


def option_scores(slot, option_ids, option_mask):
    """One logit per option, `(batch, options)`: the log-sum-exp over the option's token forms, so
    an option's probability is the mass of every way of writing it. Padding options are -inf."""
    import torch

    batch, n_options, n_forms = option_ids.shape
    picked = slot.gather(1, option_ids.reshape(batch, -1)).reshape(batch, n_options, n_forms)
    picked = picked.masked_fill(~option_mask, float("-inf"))
    return torch.logsumexp(picked, dim=-1)


def smooth_targets(targets, qtype, valid, cfg: LossConfig):
    """Targets after label smoothing (yes/no, choice) and ordinal blurring (score)."""
    import torch

    out = targets
    is_score = (qtype == TYPES.index(SCORE))[:, None]
    if cfg.ordinal_sigma > 0 and bool(is_score.any()):
        levels = torch.arange(targets.shape[1], device=targets.device, dtype=targets.dtype)
        kernel = torch.exp(-((levels[:, None] - levels[None, :]) ** 2) / (2 * cfg.ordinal_sigma**2))
        blurred = (targets @ kernel) * valid
        blurred = blurred / blurred.sum(-1, keepdim=True).clamp_min(1e-12)
        out = torch.where(is_score, blurred, out)
    if cfg.label_smoothing > 0:
        uniform = valid / valid.sum(-1, keepdim=True)
        smoothed = (1 - cfg.label_smoothing) * out + cfg.label_smoothing * uniform
        out = torch.where(is_score, out, smoothed)
    return out


def decision_loss(slot, batch: Mapping[str, Any], cfg: LossConfig):
    """`(loss, option scores)`.

    The main term is the cross-entropy between the target and the softmax over the options only,
    which is exactly the distribution the model is read with. The second term keeps the
    vocabulary's mass on option tokens, so that the same model still writes a valid code when it
    is sampled from, and so that the options' logits stay on a comparable scale.
    """
    import torch

    scores = option_scores(slot, batch["option_ids"], batch["option_mask"])
    valid = batch["option_mask"].any(-1)
    logp = torch.log_softmax(scores, dim=-1).masked_fill(~valid, 0.0)
    target = smooth_targets(batch["targets"], batch["qtype"], valid.to(batch["targets"].dtype), cfg)
    loss = -(target * logp).sum(-1)
    if cfg.vocab_mass_weight > 0:
        mass = torch.logsumexp(scores, dim=-1) - torch.logsumexp(slot, dim=-1)
        loss = loss - cfg.vocab_mass_weight * mass
    return loss.mean(), scores


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #


def score_records(
    model,
    tokenizer,
    records: Sequence[Record],
    style: PromptStyle,
    max_length: int | None,
    batch_size: int = 16,
    blind: bool = False,
    permute_seed: int | None = None,
) -> list[dict[str, Any]]:
    """One prediction per record, in order: `{"id", "source", "type", "scores", "target", "order", "tokens"}`.
    `scores` and `target` follow the order the options were shown in, and `order` maps that back
    to the record's own option order."""
    import torch

    collator = DecisionCollator(
        tokenizer, style, max_length, blind=blind, permute_seed=permute_seed, with_meta=True
    )
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    # Longest first: similar lengths share a batch, and an out-of-memory error comes at once.
    by_length = sorted(range(len(records)), key=lambda i: -len(json.dumps(records[i].state, default=str)))
    out: list[dict[str, Any] | None] = [None] * len(records)
    with torch.inference_mode():
        for start in range(0, len(by_length), batch_size):
            index = by_length[start : start + batch_size]
            batch = collator([records[i] for i in index])
            slot = slot_logits(model, batch["input_ids"].to(device), batch["attention_mask"].to(device))
            scores = option_scores(slot, batch["option_ids"].to(device), batch["option_mask"].to(device)).cpu()
            for row, i in enumerate(index):
                meta = batch["meta"][row]
                n = len(meta["target"])
                out[i] = {**meta, "scores": [float(x) for x in scores[row, :n]]}
    if was_training:
        model.train()
    return out  # type: ignore[return-value]


# --------------------------------------------------------------------------- #
# loading and saving
# --------------------------------------------------------------------------- #


def read_meta(model_dir: str | Path) -> dict[str, Any]:
    """`decision.json` of a model directory; empty for a plain language model or a Hub id."""
    path = Path(model_dir) / META_FILE
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_meta(
    model_dir: str | Path, style: PromptStyle, max_length: int | None, calibration: Mapping[str, float] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    path = Path(model_dir) / META_FILE
    meta = {
        "format": 1, "prompt": style.model_dump(), "max_length": max_length,
        "calibration": dict(calibration) if calibration else None, **(extra or {}),
    }
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def load_tokenizer(path: str, trust_remote_code: bool = False):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(path: str, precision: str = "auto", trust_remote_code: bool = False, device: str | None = None, **kwargs):
    """A causal LM, on `device` (CUDA when there is one, otherwise where it loads)."""
    import torch
    from transformers import AutoModelForCausalLM

    from stagetrain.core.training import torch_dtype

    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch_dtype(precision), trust_remote_code=trust_remote_code, **kwargs
    )
    if device is None and torch.cuda.is_available():
        device = "cuda"
    return model.to(device) if device else model


def load_records(path: str | Path, limit: int | None = None) -> list[Record]:
    from stagetrain.core.data import read_jsonl

    records = []
    for row in read_jsonl(path):
        records.append(Record.parse(row))
        if limit is not None and len(records) >= limit:
            break
    return records
