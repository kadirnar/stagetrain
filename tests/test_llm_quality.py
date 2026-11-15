"""Data-quality checks and the `llm.check_data` stage. No model, no network."""

import json

import pytest
from llm_fakes import FakeBackend

from stagetrain.core import Pipeline, PipelineError, build_config, read_jsonl, write_jsonl
from stagetrain.tracks.llm import quality as q
from stagetrain.tracks.llm import set_backend
from stagetrain.tracks.llm.check import Checks


def chat(user, assistant, **extra):
    return {"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}], **extra}


# -------------------------------------------------------------------- checks


def test_text_defects():
    assert q.text_defects("A clean sentence.") == []
    assert q.text_defects("  ") == ["empty"]
    assert q.text_defects("Sure!<|im_end|>\n<|im_start|>user") == ["template_marker"]
    assert q.text_defects("[INST] do it [/INST]") == ["template_marker"]
    assert q.text_defects("the cafÃ© is closed") == ["mojibake"] and q.text_defects("donâ€™t") == ["mojibake"]
    assert q.text_defects("bell\x07 rings") == ["control_character"]
    assert q.text_defects("<div><p>Hi</p><br></div>") == ["html"] and q.text_defects("use the <b> tag") == []
    assert q.text_defects("café and naïve are fine") == []


def test_answer_defects():
    assert q.answer_defects("Paris.") == []
    assert q.answer_defects("As an AI language model, I cannot browse.") == ["refusal"]
    assert q.answer_defects("I'm sorry, but I can't help with that.") == ["refusal"]
    assert q.answer_defects("I cannot stress enough how good this is.") == []  # "cannot" alone is not a refusal
    assert q.answer_defects("Dear [insert name], lorem ipsum.") == ["placeholder"]
    loop = "the answer is yes " * 30
    assert "repetition" in q.answer_defects(loop) and q.repetition(loop) > 0.8
    assert q.repetition("Every word in this sentence appears exactly one time only here.") == 0.0
    long_cut = (
        "Photosynthesis turns light into chemical energy. Plants absorb sunlight with chlorophyll, split water, "
        "release oxygen, and build sugars out of carbon dioxide in a cycle named after Calvin. The first stage "
        "happens in the thylakoid membranes and the second one in the"
    )
    assert q.answer_defects(long_cut) == ["truncated"]
    assert q.answer_defects("Short and no full stop") == []  # short answers end how they like
    worked = (
        "Natalia sold 48 clips to her friends in April. In May she sold half as many, which makes 24 clips for that "
        "month. Adding the two months together gives 48 + 24 = 72 clips, and no other month is mentioned in the task.\n"
    )
    assert q.answer_defects(worked + "#### 72") == []  # a closing line of its own is not a cut-off sentence
    assert q.answer_defects(worked + "So the total number of clips that she sold in both of the") == ["truncated"]
    assert q.answer_defects("```python\nprint('hi')") == ["truncated"]  # an open code block
    assert q.answer_defects("```python\nprint('hi')\n```") == []
    prompt = "Please summarise the following long paragraph about the history of the bicycle in Europe."
    assert q.answer_defects(prompt, prompt) == ["echoes_prompt"]
    assert q.answer_defects("", min_chars=1) == ["too_short"]


def test_pii_and_redaction():
    text = "Mail jane.doe@example.com or call +1 415-555-2671, card 4111 1111 1111 1111, key sk-abcdefghijklmnopqrstuvwx"
    assert set(q.find_pii(text)) == {"email", "phone", "credit_card", "secret"}
    assert q.find_pii("Order 1234 5678 9012 3456 shipped") == {}  # sixteen digits that fail the card checksum
    assert q.find_pii("version 1.2.3 and 2026-11-15 and 3.14159") == {}
    assert set(q.find_pii("server at 192.168.1.20")) == {"ip_address"} and q.find_pii("localhost 127.0.0.1") == {}
    redacted = q.redact_pii(text)
    assert "jane.doe" not in redacted and "[EMAIL]" in redacted and "[CREDIT_CARD]" in redacted and "[SECRET]" in redacted
    assert q.luhn("4111111111111111") and not q.luhn("1234567890123456")


def test_scripts():
    assert next(iter(q.script_shares("Merhaba dünya, nasılsın?"))) == "latin"
    assert next(iter(q.script_shares("Привет, мир"))) == "cyrillic"
    assert next(iter(q.script_shares("こんにちは世界"))) in ("hiragana", "han")
    mixed = q.script_shares("Hello мир")
    assert mixed["latin"] == pytest.approx(5 / 8) and mixed["cyrillic"] == pytest.approx(3 / 8)
    assert q.script_shares("12345 !!!") == {}


def test_near_duplicates_and_leaks():
    base = "The quick brown fox jumps over the lazy dog near the river bank every single morning before sunrise"
    texts = [
        base,
        "A completely different sentence about training language models on several graphics cards at once",
        base.replace("morning", "evening"),  # one word changed
        base.upper() + "!!!",  # case and punctuation changed
        "Yet another unrelated line that shares no five word sequence with any of the other rows here",
    ]
    found = q.near_duplicates(texts, threshold=0.5)
    assert set(found) == {2, 3} and found[2][:2] == ("self", 0) and found[3][:2] == ("self", 0)
    assert found[3][2] == 1.0 and 0.5 < found[2][2] < 1.0
    assert q.near_duplicates(texts, threshold=0.99).keys() == {3}  # only the exact repeat
    leaks = q.near_duplicates(texts, threshold=0.4, against=[base.replace("quick", "fast")])
    assert leaks[0][0] == "against" and leaks[2][0] == "against"  # a leak outranks a repeat in the same file
    assert q.jaccard(q.shingles("a b c d e f"), q.shingles("a b c d e f")) == 1.0 and q.jaccard(set(), {"x"}) == 0.0
    assert q.shingles("too short") == {"too short"}
    hasher = q.MinHasher(seed=1)
    assert hasher.signature({"a", "b"}) == hasher.signature({"b", "a"}) and len(hasher.band_keys(hasher.signature({"a"}))) == 16
    with pytest.raises(ValueError, match="multiple of bands"):
        q.MinHasher(num_perm=64, bands=10)


def test_leaks_by_shared_ngrams():
    held_out = ["Write a function that returns the n-th Fibonacci number using dynamic programming and explain its complexity"]
    prompts = [
        "write a function that returns the n-th fibonacci number using dynamic programming and explain its complexity!",  # exact
        "Please write a function that returns the n-th Fibonacci number using dynamic programming, in Rust",  # most of it
        "Write a function that reverses a linked list and explain its complexity",  # a few shared words only
        "Fibonacci",
    ]
    found = q.ngram_leaks(prompts, held_out)
    assert set(found) == {0, 1} and found[0] == (0, 1.0) and 0.5 < found[1][1] < 1.0
    assert q.ngram_leaks(prompts, held_out, coverage=0.99).keys() == {0}
    assert q.ngram_leaks(["What is 2+2?"], ["what is 2 + 2"]) == {0: (0, 1.0)}  # too short for 8-grams: exact match
    assert q.ngram_leaks(prompts, []) == {}
    assert q.text_defects("stop here <|custom_end|>", special_tokens=("<|custom_end|>",)) == ["template_marker"]
    assert q.repair_text("plain text") == "plain text"


def test_set_level_measures():
    assert q.distinct_n(["a b c", "a b d"]) == pytest.approx(3 / 4) and q.distinct_n([]) == 0.0
    assert q.opening_share(["Sure, here is one", "Sure, here is two", "No way"]) == ("sure here is", pytest.approx(2 / 3))
    assert q.conflicting_answers(["What is 2+2?", "what is 2+2", "Capital?"], ["4", "5", "Paris"]) == {0, 1}
    assert q.percentiles([1, 2, 3, 4, 100])["max"] == 100.0 and q.percentiles([]) == {}
    assert q.summarise_issues([{"issues": ["a", "b"]}, {"issues": ["a"]}], 4) == {"a": 0.5, "b": 0.25}


# --------------------------------------------------------------------- stage


@pytest.fixture
def files(tmp_path):
    good = [chat(f"What is the capital of country number {i}?", f"The capital of country number {i} is city {i}.") for i in range(40)]
    bad = [
        chat("What is the capital of country number 3?", "The capital of country number 3 is city 3!"),  # near-duplicate
        chat("Explain photosynthesis in one line.", "As an AI language model, I cannot explain that."),  # refusal: flag
        chat("Write a poem about the sea.", "wave after wave " * 40),  # repetition: drop
        chat("Say hi.", "Hi!<|im_end|>"),  # template marker: drop
        chat("Email the customer.", "Sent to jane.doe@example.com as asked."),  # pii: flag
        chat("What is the tallest mountain on planet number 7?", "Mount Seven."),  # leaks the held-out question
        chat("How many legs does a spider have?", "Eight."),
        chat("How many legs does a spider have?", "Six."),  # conflict: flag both
        chat("Fill the form.", "Dear [insert name], thank you."),  # placeholder: drop
    ]
    write_jsonl(tmp_path / "train.jsonl", good + bad)
    write_jsonl(tmp_path / "eval.jsonl", [chat("What is the tallest mountain on planet number 7", "Mount Seven.")])
    write_jsonl(tmp_path / "text.jsonl", [{"text": f"Document {i} talks about a topic of its own, number {i}, at some length."} for i in range(10)]
                + [{"text": "spam spam spam spam " * 30}, {"text": "<div><p>nav</p><br><span>menu</span></div>"}])
    pref = lambda p, a, b: {"prompt": [{"role": "user", "content": p}], "chosen": [{"role": "assistant", "content": a}],  # noqa: E731
                            "rejected": [{"role": "assistant", "content": b}]}
    write_jsonl(tmp_path / "pref.jsonl", [pref(f"Question {i}", f"A careful and complete answer number {i}.", f"no {i}") for i in range(10)]
                + [pref("Same", "identical answer", "Identical answer")])
    return tmp_path


def pipeline(tmp_path, **params):
    base = {"chat": str(tmp_path / "train.jsonl"), "held_out": [str(tmp_path / "eval.jsonl")],
            "text": str(tmp_path / "text.jsonl"), "preference": str(tmp_path / "pref.jsonl")}
    return Pipeline(build_config({"track": "llm", "name": "q", "output_dir": str(tmp_path / "runs"), "stages": [
        {"name": "quality", "kind": "llm.check_data", "params": {**base, **params}}]}))


def test_check_data_drops_flags_and_reports(files):
    p = pipeline(files)
    m = p.run()[0]
    run = p.run_dir / "quality"
    flags = {f["row"]: f for f in read_jsonl(run / "flags.jsonl") if f["kind"] == "chat"}
    # a near-duplicate inside the training set is flagged, not dropped
    assert flags[40]["issues"] == ["near_duplicate"] and not flags[40]["dropped"] and flags[40]["details"]["near_duplicate"]["of"] == 3
    assert flags[41]["issues"] == ["refusal"] and not flags[41]["dropped"]
    assert "repetition" in flags[42]["issues"] and flags[42]["dropped"]
    assert flags[43]["issues"] == ["template_marker"] and flags[43]["dropped"]
    assert flags[44]["issues"] == ["pii"] and flags[44]["details"]["pii"]["kinds"] == ["email"] and not flags[44]["dropped"]
    assert flags[45]["issues"] == ["leak"] and flags[45]["dropped"] and flags[45]["details"]["leak"]["covered"] == 1.0
    assert flags[46]["issues"] == flags[47]["issues"] == ["conflict"] and not flags[46]["dropped"]
    assert flags[48]["issues"] == ["placeholder"] and not flags[48]["dropped"]
    assert not any(row < 40 for row in flags)  # no clean row is touched

    # dropped: the loop, the template marker and the leak; everything else is kept and listed
    kept = list(read_jsonl(run / "chat.jsonl"))
    assert len(kept) == 49 - 3 and m.metrics["quality/chat_kept"] == 46 and m.metrics["quality/chat_rows"] == 49
    assert m.metrics["quality/chat_drop_rate"] == pytest.approx(3 / 49) and m.metrics["quality/chat_flag_rate"] == pytest.approx(6 / 49)
    assert m.metrics["quality/chat_leak"] == pytest.approx(1 / 49) and m.metrics["quality/chat_refusal"] == pytest.approx(1 / 49)
    assert 0 < m.metrics["quality/chat_distinct_2"] < 1 and m.metrics["quality/chat_top_opening_share"] > 0.7  # one template

    # of twelve documents the looping one is dropped; the one that is mostly HTML is kept and flagged
    assert m.metrics["quality/text_kept"] == 11 and m.metrics["quality/text_repetition"] == pytest.approx(1 / 12)
    assert m.metrics["quality/text_html"] == pytest.approx(1 / 12)
    assert m.metrics["quality/preference_identical_pair"] == pytest.approx(1 / 11) and m.metrics["quality/preference_kept"] == 10
    assert m.metrics["quality/preference_chosen_is_longer"] > 0.9  # here the better answer is always the longer one

    report = json.loads((run / "report.json").read_text())
    assert report["token_counts"].startswith("estimated") and report["chat"]["actions"]["refusal"] == "flag"
    assert report["chat"]["examples"]["template_marker"] == ["Say hi.\nHi!<|im_end|>"]
    assert report["chat"]["top_answer_opening"]["opening"] == "the capital of"
    assert set(m.artifacts) == {"chat", "text", "preference", "flags", "report"}


def test_actions_are_configurable(files):
    strict = pipeline(files, checks={"refusal": "drop", "pii": "redact", "conflict": "drop", "near_duplicate": "off",
                                     "low_quality": "drop"})
    m = strict.run()[0]
    kept = list(read_jsonl(strict.run_dir / "quality" / "chat.jsonl"))
    texts = [r["messages"][-1]["content"] for r in kept]
    assert not any("As an AI" in t for t in texts) and "Sent to [EMAIL] as asked." in texts
    assert not any(r["messages"][0]["content"].startswith("How many legs") for r in kept)
    assert "quality/chat_near_duplicate" not in m.metrics and any("city 3!" in t for t in texts)

    scored = pipeline(files, text=None, preference=None, scorer={"name": "numeric"}, checks={"near_duplicate": "off"})
    m = scored.run()[0]
    # a numeric scorer cannot check "Eight." or "Mount Seven.": those rows cannot be rewarded, so they go
    assert m.metrics["quality/chat_unverifiable"] > 0
    assert all(any(ch.isdigit() for ch in r["messages"][-1]["content"]) for r in read_jsonl(scored.run_dir / "quality" / "chat.jsonl"))

    script = pipeline(files, text=None, preference=None, checks={"script": "drop", "expected_scripts": ["cyrillic"]})
    with pytest.raises(ValueError, match="dropped every chat row"):
        script.run()
    with pytest.raises(PipelineError, match="colour"):
        pipeline(files, checks={"colour": "drop"}).validate()
    assert Checks().near_duplicate == "flag" and Checks().refusal == "flag" and Checks().leak == "drop"


def test_a_judge_grades_what_the_rules_kept(files):
    class Judge(FakeBackend):
        def generate(self, model, tokenizer, conversations, **kwargs):
            self.calls.append(("generate", len(conversations)))
            # low grades for answers about country numbers ending in 0, an unreadable one, good otherwise
            out = []
            for c in conversations:
                text = c[0]["content"]
                out.append("Grade: 2" if "number 0?" in text or "number 10?" in text else "no idea" if "number 5?" in text else "5")
            return out

    fake = Judge()
    set_backend(fake)
    try:
        p = pipeline(files, text=None, preference=None, judge={"init_from": "org/judge", "min_score": 3})
        m = p.run()[0]
    finally:
        set_backend(None)
    assert fake.of("generate") == [("generate", 46)]  # rows the rules already dropped are not graded
    # a low grade flags a row; it is dropped only when asked for
    assert m.metrics["quality/chat_low_quality"] == pytest.approx(2 / 49) and m.metrics["quality/chat_kept"] == 46
    assert m.metrics["quality/chat_judged"] == 46 and m.metrics["quality/chat_judge_unreadable"] == pytest.approx(1 / 46)
    assert 4.5 < m.metrics["quality/chat_judge_mean"] < 5
    assert len(fake.of("release")) == len(fake.of("load")) == 1


def test_check_data_needs_an_input(tmp_path):
    with pytest.raises(ValueError, match="at least one of chat, text, preference"):
        pipeline(tmp_path, chat=None, text=None, preference=None, held_out=[]).run()
