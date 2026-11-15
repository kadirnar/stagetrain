"""`llm.check_data`: measure the quality of training data, drop or flag bad rows, write a report.

It sits between `llm.prepare_data` and the training stages:

    - name: quality
      kind: llm.check_data
      params:
        chat: ${data.chat_train}
        held_out: ["${data.chat_eval}", "${data.chat_test}"]
      gate: {metric: quality/chat_drop_rate, max: 0.2}
    - name: sft
      kind: llm.sft
      params: {train_data: "${quality.chat}", ...}

Every check has an action: `drop` (the row is left out of the cleaned file), `flag` (the row is
kept and listed in `flags.jsonl` with the reason) or `off`. No check needs a model; `judge` adds
one that does.

The defaults follow what is known (docs/research/llm-data.md). Wrong answers damage a fine-tuned
model and surface noise barely does, so only rows that are certainly useless are dropped: empty,
looping, echoing the prompt, carrying control tokens, or repeating a held-out question. The rest
is flagged for a person to look at, because filtering on a guess can remove good rows.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from stagetrain.core import (
    Stage,
    StageContext,
    StageParams,
    StageResult,
    read_jsonl,
    register,
    write_json,
    write_jsonl,
)

from . import quality as q
from .scoring import ScorerSpec, build_scorer

Action = Literal["drop", "flag", "off"]


class Checks(BaseModel):
    """What to do about each defect, and the thresholds that define it."""

    model_config = ConfigDict(extra="forbid")

    # --- defects of any message
    empty: Action = "drop"
    template_marker: Action = "drop"  # chat-template markers or special tokens inside a message
    mojibake: Action = "flag"  # text decoded with the wrong encoding; repaired in place when ftfy is installed
    control_character: Action = "flag"
    html: Action = "flag"  # three or more HTML tags
    # --- defects of an answer
    too_short: Action = "drop"
    min_answer_chars: int = 1
    refusal: Action = "flag"  # "As an AI language model ...": drop it unless refusing is what you teach
    placeholder: Action = "flag"  # TODO, lorem ipsum, [insert ...]
    repetition: Action = "drop"  # an answer stuck in a loop
    max_repetition: float = Field(default=0.5, ge=0.0, le=1.0)  # share of word 10-grams that repeat an earlier one
    truncated: Action = "flag"  # a long answer with no closing punctuation, or an open code block
    truncation_chars: int = 200
    echoes_prompt: Action = "drop"  # the answer repeats the prompt
    # --- defects of a row among the others
    # Nearly the same row appears earlier. Flagged, not dropped: no controlled result shows that
    # removing near-duplicates inside the training set raises accuracy, and in templated data
    # they are legitimate.
    near_duplicate: Action = "flag"
    near_duplicate_threshold: float = Field(default=0.7, gt=0.0, le=1.0)  # Jaccard over word 5-grams
    # The question repeats a held-out one: exactly, by word 8-grams covering more than half of the
    # held-out question, or by overall similarity. Always worth dropping: it inflates the reported
    # number, by 10 points in one measurement.
    leak: Action = "drop"
    leak_threshold: float = Field(default=0.8, gt=0.0, le=1.0)  # Jaccard over word 5-grams
    leak_coverage: float = Field(default=0.5, gt=0.0, le=1.0)  # share of a held-out question's 8-grams
    conflict: Action = "flag"  # the same question with different answers
    # --- privacy
    pii: Literal["drop", "flag", "redact", "off"] = "flag"  # e-mail, phone, card number, IP address, credential
    # --- length
    overlength: Action = "flag"  # longer than max_seq_length tokens; the training stages drop these themselves
    # --- writing system
    script: Action = "off"
    expected_scripts: list[str] = Field(default_factory=list)  # e.g. [latin]; a row mostly in another script is a defect
    # --- with a scorer (rows whose answers a scorer has to check, e.g. for llm.grpo)
    unverifiable: Action = "drop"  # the scorer does not accept the reference answer as correct
    answer_in_prompt: Action = "flag"  # the reference answer can be copied from the prompt
    # --- with a judge
    low_quality: Action = "flag"  # a model's grade is a reason to look, not proof
    # --- preference pairs
    identical_pair: Action = "drop"  # chosen equals rejected
    similar_pair: Action = "flag"  # chosen and rejected nearly the same
    similar_pair_threshold: float = Field(default=0.95, gt=0.0, le=1.0)


class JudgeSpec(BaseModel):
    """A model grades each chat row from 1 to 5; rows under `min_score` are `low_quality`. It
    costs one generation per row, and its grades are only as good as the model: validate it on
    rows you have graded yourself before trusting what it drops."""

    model_config = ConfigDict(extra="forbid")

    init_from: str
    backend: Literal["unsloth", "hf"] = "unsloth"
    load_in: Literal["4bit", "8bit", "16bit"] = "4bit"
    max_seq_length: int = 4096
    min_score: int = Field(default=3, ge=1, le=5)
    max_samples: int | None = None  # grade only the first rows; the rest are left ungraded
    rubric: str = (
        "Grade the assistant's answer to the user's request from 1 to 5.\n"
        "5: correct, complete and directly useful. 3: partly correct or incomplete. 1: wrong, off-topic or empty.\n"
        "Reply with the grade only, as one digit."
    )
    chat_template_kwargs: dict[str, Any] = Field(default_factory=dict)


class CheckParams(StageParams):
    chat: str | None = None  # chat JSONL, e.g. ${data.chat_train}
    text: str | None = None  # raw-text JSONL for continued pretraining
    preference: str | None = None  # prompt / chosen / rejected JSONL
    held_out: list[str] = Field(default_factory=list)  # chat files no training row may repeat (eval, test)
    checks: Checks = Field(default_factory=Checks)
    max_seq_length: int | None = None
    tokenizer: str | None = None  # counts tokens exactly; without it a token is taken as four characters
    scorer: ScorerSpec | None = None  # set it when a scorer will judge answers to these rows
    judge: JudgeSpec | None = None
    examples: int = 5  # rows quoted in the report per defect


def prompt_of(messages: list[dict[str, str]]) -> str:
    """What is asked: everything before the last assistant turn, the system message aside."""
    return "\n".join(m["content"] for m in messages[:-1] if m["role"] != "system")


def chat_text(messages: list[dict[str, str]]) -> str:
    return "\n".join(m["content"] for m in messages)


class Auditor:
    """Collects, per row, the defects found and whether one of them drops the row."""

    def __init__(self, checks: Checks, n_rows: int):
        self.checks = checks
        self.issues: list[list[str]] = [[] for _ in range(n_rows)]
        self.details: list[dict[str, Any]] = [{} for _ in range(n_rows)]

    def add(self, i: int, issue: str, **detail: Any) -> None:
        action = "flag" if getattr(self.checks, issue) == "redact" else getattr(self.checks, issue)
        if action == "off" or issue in self.issues[i]:
            return
        self.issues[i].append(issue)
        if detail:
            self.details[i][issue] = detail

    def dropped(self, i: int) -> bool:
        return any(getattr(self.checks, issue) == "drop" for issue in self.issues[i])


def token_counter(tokenizer_name: str | None):
    """`(count, exact, special_tokens)`: tokens of a text, exactly with a tokenizer or as
    characters / 4, and the tokenizer's own control tokens."""
    if not tokenizer_name:
        return (lambda text: len(text) // 4 + 1), False, ()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    special = tuple(t for t in tokenizer.all_special_tokens if len(t) >= 3)
    return (lambda text: len(tokenizer(text, add_special_tokens=False)["input_ids"])), True, special


def audit_texts(audit: Auditor, texts: list[str], p: CheckParams, count, special: tuple[str, ...] = ()) -> None:
    """Checks that apply to any text: defects, privacy, length, writing system, near-duplicates."""
    c = p.checks
    for i, text in enumerate(texts):
        for issue in q.text_defects(text, special):
            audit.add(i, issue)
        if c.pii != "off":
            found = q.find_pii(text)
            if found:
                audit.add(i, "pii", kinds=sorted(found))
        if p.max_seq_length and c.overlength != "off" and count(text) > p.max_seq_length:
            audit.add(i, "overlength", tokens=count(text))
        if c.script != "off" and c.expected_scripts:
            shares = q.script_shares(text)
            if shares and next(iter(shares)) not in c.expected_scripts:
                audit.add(i, "script", script=next(iter(shares)))
    if c.near_duplicate != "off":
        for i, (_, other, similarity) in q.near_duplicates(texts, c.near_duplicate_threshold).items():
            audit.add(i, "near_duplicate", of=other, similarity=round(similarity, 3))


def audit_chat(
    rows: list[dict[str, Any]], held_out: list[dict[str, Any]], p: CheckParams, count, special: tuple[str, ...] = ()
) -> Auditor:
    c = p.checks
    audit = Auditor(c, len(rows))
    prompts = [prompt_of(r["messages"]) for r in rows]
    answers = [r["messages"][-1]["content"] if r["messages"][-1]["role"] == "assistant" else "" for r in rows]
    audit_texts(audit, [chat_text(r["messages"]) for r in rows], p, count, special)
    for i, row in enumerate(rows):
        for message in row["messages"]:
            if message["role"] == "assistant":
                for issue in q.answer_defects(
                    message["content"], prompts[i], min_chars=c.min_answer_chars, max_repetition=c.max_repetition,
                    truncation_chars=c.truncation_chars,
                ):
                    audit.add(i, issue)
    if c.conflict != "off":
        for i in q.conflicting_answers(prompts, answers):
            audit.add(i, "conflict")
    if c.leak != "off" and held_out:
        against = [prompt_of(r["messages"]) for r in held_out]
        for i, (other, share) in q.ngram_leaks(prompts, against, coverage=c.leak_coverage).items():
            audit.add(i, "leak", held_out_row=other, covered=round(share, 3))
        for i, (where, other, similarity) in q.near_duplicates(prompts, c.leak_threshold, against=against).items():
            if where == "against":
                audit.add(i, "leak", held_out_row=other, similarity=round(similarity, 3))
    if p.scorer is not None:
        judge = build_scorer(p.scorer)
        for i, row in enumerate(rows):
            if judge(answers[i], answers[i], row) < 1.0:
                audit.add(i, "unverifiable")  # e.g. a numeric scorer and a reference with no number in it
            elif len(answers[i].strip()) >= 4 and q.normalise(answers[i]) and q.normalise(answers[i]) in q.normalise(prompts[i]):
                audit.add(i, "answer_in_prompt")
    return audit


def audit_preference(rows: list[dict[str, Any]], p: CheckParams, count, special: tuple[str, ...] = ()) -> Auditor:
    c = p.checks
    audit = Auditor(c, len(rows))
    chosen = [chat_text(r["chosen"]) for r in rows]
    rejected = [chat_text(r["rejected"]) for r in rows]
    audit_texts(audit, [f"{chat_text(r['prompt'])}\n{a}" for r, a in zip(rows, chosen, strict=True)], p, count, special)
    for i, (good, bad) in enumerate(zip(chosen, rejected, strict=True)):
        if q.normalise(good) == q.normalise(bad):
            audit.add(i, "identical_pair")
        elif q.jaccard(q.shingles(good, 3), q.shingles(bad, 3)) >= c.similar_pair_threshold:
            audit.add(i, "similar_pair")
        for issue in q.answer_defects(good, max_repetition=c.max_repetition, truncation_chars=c.truncation_chars):
            audit.add(i, issue)
    return audit


def grade(rows: list[dict[str, Any]], audit: Auditor, spec: JudgeSpec) -> dict[str, float]:
    """Ask a model for a 1-5 grade of each row that is not already dropped."""
    from .backend import LoadSpec, get_backend

    backend = get_backend(spec.backend)
    picked = [i for i in range(len(rows)) if not audit.dropped(i)][: spec.max_samples]
    if not picked:
        return {}
    model, tokenizer = backend.load(LoadSpec(model_name=spec.init_from, max_seq_length=spec.max_seq_length, load_in=spec.load_in))
    prompts = [
        [{"role": "user", "content": f"{spec.rubric}\n\n[Request]\n{prompt_of(rows[i]['messages'])}\n\n"
                                     f"[Answer]\n{rows[i]['messages'][-1]['content']}"}]
        for i in picked
    ]
    replies = backend.generate(
        model, tokenizer, prompts, max_new_tokens=8, temperature=0.0, template_kwargs=spec.chat_template_kwargs or None
    )
    model = tokenizer = None
    backend.release()
    grades = []
    for i, reply in zip(picked, replies, strict=True):
        digit = next((ch for ch in reply if ch in "12345"), None)
        if digit is None:
            continue  # an unreadable grade is not a low grade
        grades.append(int(digit))
        if int(digit) < spec.min_score:
            audit.add(i, "low_quality", grade=int(digit))
    if not grades:
        return {"judged": float(len(picked)), "judge_unreadable": 1.0}
    return {
        "judged": float(len(picked)), "judge_mean": sum(grades) / len(grades),
        "judge_unreadable": 1.0 - len(grades) / len(picked),
    }


def finish_kind(
    ctx: StageContext, kind: str, rows: list[dict[str, Any]], audit: Auditor, p: CheckParams, text_of
) -> tuple[dict[str, float], dict[str, Any], list[dict[str, Any]], Any]:
    """Write the cleaned file; return its metrics, report section and per-row flags."""
    kept, flags = [], []
    examples: dict[str, list[str]] = {}
    for i, row in enumerate(rows):
        if audit.issues[i]:
            dropped = audit.dropped(i)
            flags.append({"kind": kind, "row": i, "issues": audit.issues[i], "dropped": dropped, "details": audit.details[i]})
            for issue in audit.issues[i]:
                if len(examples.setdefault(issue, [])) < p.examples:
                    examples[issue].append(text_of(row)[:300])
            if dropped:
                continue
        if p.checks.pii == "redact" and "pii" in audit.issues[i]:
            row = rewrite(row, q.redact_pii)
        if "mojibake" in audit.issues[i]:
            row = rewrite(row, q.repair_text)  # a no-op without ftfy
        kept.append(row)
    path = ctx.path(f"{kind}.jsonl")
    write_jsonl(path, kept)
    n = len(rows)
    rates = q.summarise_issues(flags, n)
    metrics = {f"quality/{kind}_rows": float(n), f"quality/{kind}_kept": float(len(kept)),
               f"quality/{kind}_drop_rate": 1 - len(kept) / n if n else 0.0,
               f"quality/{kind}_flag_rate": sum(1 for f in flags if not f["dropped"]) / n if n else 0.0}
    metrics.update({f"quality/{kind}_{issue}": rate for issue, rate in rates.items()})
    section = {
        "rows": n, "kept": len(kept), "issues": {issue: round(rate, 5) for issue, rate in rates.items()},
        "actions": {issue: getattr(p.checks, issue) for issue in rates}, "examples": examples,
    }
    return metrics, section, flags, path


def rewrite(row: dict[str, Any], fix) -> dict[str, Any]:
    """The row with `fix` applied to every piece of its text."""
    if "messages" in row:
        return {**row, "messages": [{**m, "content": fix(m["content"])} for m in row["messages"]]}
    if "text" in row:
        return {**row, "text": fix(row["text"])}
    return {**row, **{k: [{**m, "content": fix(m["content"])} for m in row[k]] for k in ("prompt", "chosen", "rejected")}}


@register("llm.check_data")
class CheckData(Stage):
    """Check training data for defects, duplicates, leaks and private data; drop or flag rows; report."""

    Params = CheckParams

    def run(self, ctx: StageContext) -> StageResult:
        p: CheckParams = self.params
        if not (p.chat or p.text or p.preference):
            raise ValueError("check_data needs at least one of chat, text, preference")
        count, exact, special = token_counter(p.tokenizer)
        metrics: dict[str, float] = {}
        report: dict[str, Any] = {"token_counts": "exact" if exact else "estimated as characters / 4"}
        artifacts: dict[str, Any] = {}
        flags: list[dict[str, Any]] = []

        if p.chat:
            rows = [r for r in read_jsonl(p.chat) if r.get("messages")]
            held_out = [r for path in p.held_out for r in read_jsonl(path) if r.get("messages")]
            audit = audit_chat(rows, held_out, p, count, special)
            if p.judge is not None:
                metrics.update({f"quality/chat_{k}": v for k, v in grade(rows, audit, p.judge).items()})
            m, section, f, artifacts["chat"] = finish_kind(ctx, "chat", rows, audit, p, lambda r: chat_text(r["messages"]))
            answers = [r["messages"][-1]["content"] for r in rows]
            prompts = [prompt_of(r["messages"]) for r in rows]
            opening, share = q.opening_share(answers)
            lengths = [count(chat_text(r["messages"])) for r in rows]
            section.update({
                # how varied the data is: low values mean a few templates wrote most of it
                "distinct_2_answers": q.distinct_n(answers), "distinct_2_prompts": q.distinct_n(prompts),
                "top_answer_opening": {"opening": opening, "share": share},
                "tokens": q.percentiles(lengths),
                "sources": dict(Counter(str(r.get("source", "unknown")) for r in rows).most_common(20)),
            })
            m.update({
                "quality/chat_distinct_2": section["distinct_2_answers"], "quality/chat_top_opening_share": share,
                "quality/chat_tokens_p95": section["tokens"].get("p95", 0.0),
            })
            metrics.update(m)
            report["chat"] = section
            flags += f
        if p.text:
            rows = [r for r in read_jsonl(p.text) if isinstance(r.get("text"), str)]
            audit = Auditor(p.checks, len(rows))
            audit_texts(audit, [r["text"] for r in rows], p, count, special)
            for i, row in enumerate(rows):
                if q.repetition(row["text"]) > p.checks.max_repetition:
                    audit.add(i, "repetition")
            m, section, f, artifacts["text"] = finish_kind(ctx, "text", rows, audit, p, lambda r: r["text"])
            section["tokens"] = q.percentiles([count(r["text"]) for r in rows])
            metrics.update(m)
            report["text"] = section
            flags += f
        if p.preference:
            rows = [r for r in read_jsonl(p.preference) if r.get("chosen") and r.get("rejected")]
            audit = audit_preference(rows, p, count, special)
            m, section, f, artifacts["preference"] = finish_kind(
                ctx, "preference", rows, audit, p, lambda r: f"{chat_text(r['chosen'])} | {chat_text(r['rejected'])}"
            )
            longer = [len(chat_text(r["chosen"])) > len(chat_text(r["rejected"])) for r in rows]
            # near 0.5 is healthy; near 1 the model can learn "longer is better" instead of "better"
            section["chosen_is_longer"] = sum(longer) / len(longer) if longer else 0.0
            m["quality/preference_chosen_is_longer"] = section["chosen_is_longer"]
            metrics.update(m)
            report["preference"] = section
            flags += f

        artifacts["flags"] = ctx.path("flags.jsonl")
        write_jsonl(artifacts["flags"], flags)
        artifacts["report"] = write_json(ctx.path("report.json"), report)
        for kind in ("chat", "text", "preference"):
            if kind in report and not report[kind]["kept"]:
                raise ValueError(f"check_data dropped every {kind} row; see {artifacts['report']}")
        return StageResult(artifacts=artifacts, metrics=metrics, notes={k: v for k, v in report.items() if k != "token_counts"})
