"""Checks on training data. Each one looks at text and says what is wrong with it; none loads a
model. `llm.check_data` runs them, drops or flags rows, and writes the report.

Everything here is plain Python except near-duplicate detection, which uses numpy when it is
installed and is exact but slow without it.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from typing import Any

# --------------------------------------------------------------------------- #
# text-level checks: each returns the names of the defects it found
# --------------------------------------------------------------------------- #

# Chat-template markers and special tokens. Inside a message they are leftovers of an earlier
# rendering, and a model trained on them learns to write them.
TEMPLATE_MARKERS = re.compile(
    r"<\|im_start\|>|<\|im_end\|>|<\|eot_id\|>|<\|start_header_id\|>|<\|end_header_id\|>|<\|endoftext\|>|"
    r"<\|begin_of_text\|>|<\|end_of_text\|>|<start_of_turn>|<end_of_turn>|\[INST\]|\[/INST\]|<<SYS>>|<</SYS>>|"
    r"<\|user\|>|<\|assistant\|>|<\|system\|>|<s>|</s>|<bos>|<eos>|<\|start\|>|<\|end\|>|<\|channel\|>|<\|message\|>"
)
# UTF-8 read as Latin-1 / Windows-1252 and the replacement character.
MOJIBAKE = re.compile(r"Ã[\x80-\xbf©¨¤¶¼ª§±¡³º]|â€[\x80-\xbf™œ\x9d“”˜¦¢]|Â[\xa0-\xbf ]|ï¿½|�")
CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
HTML_TAG = re.compile(r"</?(?:div|span|p|br|a|img|table|tr|td|ul|li|script|style|html|body|head|font|h[1-6])\b[^>]*>", re.I)
REFUSAL = re.compile(
    r"\bas an ai(?: language)? model\b|\bas a large language model\b|\bi(?:'m| am) (?:just |only )?an ai\b|"
    r"\bi (?:cannot|can't|can not|am unable to|'m unable to|won't be able to) (?:assist|help|provide|fulfil|fulfill|comply|"
    r"answer|create|generate|write)\b|\bi(?:'m| am) sorry,? but\b|\bi apologi[sz]e,? but\b|"
    r"\bi do(?: not|n't) have (?:access to|the ability|personal)\b|\bmy (?:knowledge|training) (?:cutoff|data)\b",
    re.I,
)
PLACEHOLDER = re.compile(
    r"\blorem ipsum\b|\[(?:insert|your|placeholder|todo|name|company|link|url|date)[^\]]{0,40}\]|<(?:insert|your)[^>]{0,40}>|"
    r"\bTODO\b|\bFIXME\b|\bXXX+\b|\byour[_ ]?(?:api[_ ]?key|name here|text here)\b",
    re.I,
)
TERMINAL = re.compile(r"""[.!?:;…。！？)\]}"'`»”’*|>]\s*$""")
WORD = re.compile(r"\w+", re.UNICODE)

PII_PATTERNS = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "phone": re.compile(r"(?<![\w.])(?:\+\d{1,3}[ .-]?)?(?:\(\d{2,4}\)[ .-]?)?\d{3}[ .-]\d{3,4}[ .-]\d{2,4}(?![\w.])"),
    "ip_address": re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]?){13,19}\b"),
    "secret": re.compile(
        r"\bsk-[A-Za-z0-9_-]{20,}\b|\bAKIA[0-9A-Z]{16}\b|\bghp_[A-Za-z0-9]{36}\b|\bgithub_pat_[A-Za-z0-9_]{40,}\b|"
        r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b|\bhf_[A-Za-z0-9]{30,}\b|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|"
        r"\bAIza[0-9A-Za-z_-]{35}\b"
    ),
}


def luhn(digits: str) -> bool:
    """Whether a digit string passes the checksum credit-card numbers carry."""
    total, double = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if double:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
        double = not double
    return total % 10 == 0


def find_pii(text: str) -> dict[str, list[str]]:
    """What looks like personal data or a credential, by kind. Regular expressions only: they
    find formats (an e-mail address, a key), not names or addresses in prose."""
    found: dict[str, list[str]] = {}
    for kind, pattern in PII_PATTERNS.items():
        hits = pattern.findall(text)
        if kind == "credit_card":
            hits = [h for h in hits if 13 <= len(re.sub(r"\D", "", h)) <= 19 and luhn(re.sub(r"\D", "", h))]
        if kind == "ip_address":
            hits = [h for h in hits if not h.startswith(("0.", "127.", "255."))]
        if hits:
            found[kind] = hits
    return found


def redact_pii(text: str) -> str:
    for kind in find_pii(text):
        pattern = PII_PATTERNS[kind]
        if kind == "credit_card":
            text = pattern.sub(lambda m: "[CREDIT_CARD]" if luhn(re.sub(r"\D", "", m.group(0))) else m.group(0), text)
        else:
            text = pattern.sub(f"[{kind.upper()}]", text)
    return text


def repetition(text: str, n: int = 10) -> float:
    """The share of word n-grams that repeat an earlier one: 0 for ordinary prose, near 1 for a
    generation stuck in a loop."""
    words = WORD.findall(text.lower())
    if len(words) < 2 * n:
        return 0.0
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def looks_truncated(text: str, min_chars: int = 200) -> bool:
    """A long answer that stops without closing punctuation, or inside a code block: probably cut
    off by a token limit when it was generated."""
    stripped = text.rstrip()
    if stripped.count("```") % 2 == 1:
        return True
    if len(stripped) < min_chars or TERMINAL.search(stripped):
        return False
    # A short last line is a closing line of its own ("#### 72", "Answer: B"), not a sentence cut off.
    return len(WORD.findall(stripped.splitlines()[-1])) > 6


def text_defects(text: str, special_tokens: Sequence[str] = ()) -> list[str]:
    """Defects of one piece of text, whoever wrote it. `special_tokens` are the target
    tokenizer's own control tokens, checked on top of the common ones."""
    out = []
    if not text.strip():
        return ["empty"]
    if TEMPLATE_MARKERS.search(text) or any(token in text for token in special_tokens):
        out.append("template_marker")
    if MOJIBAKE.search(text):
        out.append("mojibake")
    if CONTROL.search(text):
        out.append("control_character")
    if len(HTML_TAG.findall(text)) >= 3:
        out.append("html")
    return out


def answer_defects(
    answer: str, prompt: str = "", *, min_chars: int = 1, max_repetition: float = 0.5, truncation_chars: int = 200
) -> list[str]:
    """Defects only an answer can have."""
    out = []
    if len(answer.strip()) < min_chars:
        out.append("too_short")
    if REFUSAL.search(answer[:400]):
        out.append("refusal")
    if PLACEHOLDER.search(answer):
        out.append("placeholder")
    if repetition(answer) > max_repetition:
        out.append("repetition")
    if looks_truncated(answer, truncation_chars):
        out.append("truncated")
    if prompt and len(answer) >= 40 and jaccard(shingles(answer), shingles(prompt)) >= 0.9:
        out.append("echoes_prompt")
    return out


# --------------------------------------------------------------------------- #
# scripts
# --------------------------------------------------------------------------- #

_SCRIPTS = (
    ("latin", "LATIN"), ("cyrillic", "CYRILLIC"), ("arabic", "ARABIC"), ("greek", "GREEK"), ("hebrew", "HEBREW"),
    ("devanagari", "DEVANAGARI"), ("thai", "THAI"), ("hangul", "HANGUL"), ("hiragana", "HIRAGANA"),
    ("katakana", "KATAKANA"), ("han", "CJK"),
)


def script_shares(text: str, limit: int = 2000) -> dict[str, float]:
    """The share of letters per writing system. It tells Latin from Cyrillic from Han, not
    Turkish from English: for that, name a language detector in the stage."""
    counts: Counter = Counter()
    for ch in text[:limit]:
        if not ch.isalpha():
            continue
        name = unicodedata.name(ch, "")
        counts[next((script for script, key in _SCRIPTS if key in name), "other")] += 1
    total = sum(counts.values())
    return {script: n / total for script, n in counts.most_common()} if total else {}


# --------------------------------------------------------------------------- #
# duplicates
# --------------------------------------------------------------------------- #


def normalise(text: str) -> str:
    return " ".join(WORD.findall(text.lower()))


def shingles(text: str, n: int = 5) -> set[str]:
    """Word n-grams of the normalised text (shorter texts: the text itself)."""
    words = WORD.findall(text.lower())
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _hash32(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=4).digest(), "big")


class MinHasher:
    """MinHash signatures with locality-sensitive hashing: finds pairs of texts whose shingle
    sets overlap by about `threshold` or more without comparing every pair."""

    PRIME = (1 << 61) - 1

    def __init__(self, num_perm: int = 64, bands: int = 16, seed: int = 0):
        import random

        if num_perm % bands:
            raise ValueError("num_perm must be a multiple of bands")
        rng = random.Random(seed)
        self.num_perm, self.bands, self.rows = num_perm, bands, num_perm // bands
        self.a = [rng.randrange(1, self.PRIME) for _ in range(num_perm)]
        self.b = [rng.randrange(0, self.PRIME) for _ in range(num_perm)]
        try:
            import numpy as np

            self._np = np
            self._a = np.array(self.a, dtype=np.uint64)[:, None]
            self._b = np.array(self.b, dtype=np.uint64)[:, None]
        except ImportError:
            self._np = None

    def signature(self, items: set[str]) -> tuple[int, ...]:
        hashes = [_hash32(s) for s in items] or [0]
        if self._np is not None:
            np = self._np
            # 32-bit shingle hashes times 61-bit multipliers wrap in uint64; the wrap is itself a hash.
            h = np.array(hashes, dtype=np.uint64)[None, :]
            return tuple(int(x) for x in ((self._a * h + self._b) % np.uint64(self.PRIME)).min(axis=1))
        return tuple(min((a * h + b) % self.PRIME for h in hashes) for a, b in zip(self.a, self.b, strict=True))

    def band_keys(self, signature: Sequence[int]) -> list[tuple[int, tuple[int, ...]]]:
        return [(i, tuple(signature[i * self.rows : (i + 1) * self.rows])) for i in range(self.bands)]


def near_duplicates(
    texts: Sequence[str], threshold: float = 0.85, against: Sequence[str] = (), shingle_words: int = 5, seed: int = 0
) -> dict[int, tuple[str, int, float]]:
    """For each text that nearly repeats another: `index -> (where, other_index, similarity)`.

    `where` is `"self"` when the other text is an earlier one of `texts` (the later copy is the
    duplicate), or `"against"` when it is one of `against` (e.g. the held-out rows: a leak).
    Candidates come from LSH; every reported pair is confirmed by its exact Jaccard similarity.
    """
    # Fewer rows per band find less similar pairs: with 4 rows a pair at 0.85 is found with
    # probability above 0.999 but one at 0.5 only two times in three; with 2 rows that one is
    # found too, at the price of more candidates to verify.
    hasher = MinHasher(bands=16 if threshold >= 0.8 else 32, seed=seed)
    buckets: dict[tuple[int, tuple[int, ...]], list[tuple[str, int]]] = defaultdict(list)
    sets_against = [shingles(t, shingle_words) for t in against]
    for j, items in enumerate(sets_against):
        for key in hasher.band_keys(hasher.signature(items)):
            buckets[key].append(("against", j))
    found: dict[int, tuple[str, int, float]] = {}
    sets_self: list[set[str]] = []
    for i, text in enumerate(texts):
        items = shingles(text, shingle_words)
        sets_self.append(items)
        keys = hasher.band_keys(hasher.signature(items))
        candidates = {c for key in keys for c in buckets.get(key, ())}
        best: tuple[str, int, float] | None = None
        for where, j in candidates:
            similarity = jaccard(items, sets_against[j] if where == "against" else sets_self[j])
            # a leak outranks a repeat inside the same file
            if similarity >= threshold and (best is None or (where == "against", similarity) > (best[0] == "against", best[2])):
                best = (where, j, similarity)
        if best is not None:
            found[i] = best
        if best is None or best[0] == "against":
            for key in keys:  # only texts that are kept (or leak) stand as originals for later ones
                buckets[key].append(("self", i))
    return found


def ngram_leaks(prompts: Sequence[str], held_out: Sequence[str], n: int = 8, coverage: float = 0.5) -> dict[int, tuple[int, float]]:
    """Training prompts that share a large part of a held-out prompt: `index -> (held-out index,
    share)`, where `share` is the fraction of the held-out prompt's word n-grams that also occur
    in the training prompt. An exact repeat (case and punctuation aside) counts as 1.0. The rule
    of 8-grams covering more than half of an evaluation prompt is the one Tulu 3 used to
    decontaminate its training data."""
    index: dict[tuple[str, ...], list[int]] = defaultdict(list)
    sizes, exact = [], {}
    for j, text in enumerate(held_out):
        words = WORD.findall(text.lower())
        exact.setdefault(" ".join(words), j)
        grams = {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}
        sizes.append(len(grams))
        for gram in grams:
            index[gram].append(j)
    found: dict[int, tuple[int, float]] = {}
    for i, text in enumerate(prompts):
        words = WORD.findall(text.lower())
        key = " ".join(words)
        if key and key in exact:
            found[i] = (exact[key], 1.0)
            continue
        hits: Counter = Counter()
        for gram in {tuple(words[k : k + n]) for k in range(len(words) - n + 1)}:
            for j in index.get(gram, ()):
                hits[j] += 1
        best = max(((count / sizes[j], j) for j, count in hits.items()), default=(0.0, -1))
        if best[0] > coverage:
            found[i] = (best[1], best[0])
    return found


def repair_text(text: str) -> str:
    """Text with broken encodings repaired, when `ftfy` is installed; otherwise as it is."""
    try:
        import ftfy
    except ImportError:
        return text
    return ftfy.fix_text(text)


# --------------------------------------------------------------------------- #
# set-level measures
# --------------------------------------------------------------------------- #


def distinct_n(texts: Iterable[str], n: int = 2) -> float:
    """Distinct word n-grams over all n-grams: how varied the wording of a set is."""
    total, seen = 0, set()
    for text in texts:
        words = WORD.findall(text.lower())
        for i in range(len(words) - n + 1):
            seen.add(tuple(words[i : i + n]))
            total += 1
    return len(seen) / total if total else 0.0


def opening_share(texts: Sequence[str], words: int = 3) -> tuple[str, float]:
    """The most common opening of a set of texts and the share of texts that start with it: one
    template writing most of a dataset shows up here."""
    openings = Counter(" ".join(WORD.findall(t.lower())[:words]) for t in texts if t.strip())
    if not openings:
        return "", 0.0
    opening, count = openings.most_common(1)[0]
    return opening, count / len(texts)


def percentiles(values: Sequence[float], points: Sequence[int] = (5, 50, 95, 99)) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    return {f"p{p}": float(ordered[min(len(ordered) - 1, int(len(ordered) * p / 100))]) for p in points} | {
        "max": float(ordered[-1]), "mean": sum(ordered) / len(ordered)}


def conflicting_answers(prompts: Sequence[str], answers: Sequence[str]) -> set[int]:
    """Indices of rows whose prompt also appears with a different answer."""
    by_prompt: dict[str, set[str]] = defaultdict(set)
    for prompt, answer in zip(prompts, answers, strict=True):
        by_prompt[normalise(prompt)].add(normalise(answer))
    return {i for i, prompt in enumerate(prompts) if len(by_prompt[normalise(prompt)]) > 1}


def summarise_issues(flags: Sequence[dict[str, Any]], total: int) -> dict[str, float]:
    """`{issue: share of rows}` from per-row flags."""
    counts: Counter = Counter(issue for row in flags for issue in row["issues"])
    return {issue: n / total for issue, n in sorted(counts.items())} if total else {}
