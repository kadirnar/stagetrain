"""A tiny random causal LM and word-level tokenizer for the decision tests. Nothing is downloaded."""

from __future__ import annotations

import random
from pathlib import Path

CHATML = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)
POSITIVE = ["great", "wonderful", "excellent", "superb"]
NEUTRAL = ["okay", "average", "fine", "plain"]
NEGATIVE = ["awful", "terrible", "boring", "dreadful"]
FILLER = ["the", "movie", "film", "was", "really", "quite", "and", "story", "acting", "ending"]
PROMPT_WORDS = (
    "QUESTION Options Answer with the code of one option yes or no number from to state question context "
    "Which sentiment does review have Is positive How good is it Choose correct answer "
    "positive neutral negative praise complaint neither user assistant system [...] billing tech refund "
    "Proposed proposed correct None of the other options applies none above About record with id records content"
)
LEVELS = ["negative", "neutral", "positive"]


def build_tokenizer(extra_words: list[str] | None = None):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    pre = pre_tokenizers.Whitespace()
    words: dict[str, None] = {}
    texts = [PROMPT_WORDS, " ".join(POSITIVE + NEUTRAL + NEGATIVE + FILLER), " ".join(extra_words or []),
             '{ } [ ] " : , . ? _ -']
    for text in texts:
        for piece, _ in pre.pre_tokenize_str(text):
            words[piece] = None
    for code in [chr(ord("A") + i) for i in range(26)] + [str(i) for i in range(10)] + [f"{i:02d}" for i in range(40)]:
        words[code] = None
    for form in ("yes", "Yes", "YES", "no", "No", "NO"):
        words[form] = None
    special = ["[PAD]", "[UNK]", "<|im_start|>", "<|im_end|>"]
    vocab = {w: i for i, w in enumerate(special + [w for w in words if w not in special])}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre
    tok.add_special_tokens(special)
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]", eos_token="<|im_end|>",
    )
    fast.chat_template = CHATML
    return fast


def build_model_dir(path: Path, seed: int = 0, extra_words: list[str] | None = None) -> Path:
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    tokenizer = build_tokenizer(extra_words)
    torch.manual_seed(seed)
    config = LlamaConfig(
        vocab_size=len(tokenizer), hidden_size=48, intermediate_size=96, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=512,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, bos_token_id=None,
    )
    LlamaForCausalLM(config).save_pretrained(str(path))
    tokenizer.save_pretrained(str(path))
    return path


def reviews(n: int, seed: int = 0) -> list[dict]:
    """Synthetic reviews whose sentiment word decides the label: 0 negative, 1 neutral, 2 positive."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        level = rng.randrange(3)
        word = rng.choice([NEGATIVE, NEUTRAL, POSITIVE][level])
        filler = rng.sample(FILLER, 4)
        text = f"{filler[0]} {filler[1]} {filler[2]} {word} {filler[3]}"
        rows.append({"text": text, "label": level, "name": LEVELS[level], "is_positive": int(level == 2)})
    return rows
