import json

import pytest

from stagetrain.tracks.llm.data import (
    Collector,
    Rejected,
    chunk_text,
    normalise_chat,
    normalise_preference,
    split_rows,
    validate_messages,
)

U, A = {"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}


def test_every_chat_format_becomes_messages():
    assert normalise_chat({"messages": [U, A]}) == [U, A]
    sharegpt = {"conversations": [
        {"from": "system", "value": "be brief"}, {"from": "human", "value": " hi "}, {"from": "gpt", "value": "hello"},
    ]}
    assert normalise_chat(sharegpt, "sharegpt") == [{"role": "system", "content": "be brief"}, U, A]
    alpaca = {"instruction": "Translate", "input": "merhaba", "output": "hello"}
    assert normalise_chat(alpaca, "alpaca") == [{"role": "user", "content": "Translate\n\nmerhaba"}, A]
    assert normalise_chat({"instruction": "hi", "input": "", "output": "hello"}, "alpaca") == [U, A]
    assert normalise_chat({"prompt": "hi", "completion": "hello"}, "prompt_completion") == [U, A]
    assert normalise_chat({"messages": json.dumps([U, A])}) == [U, A]  # a JSON string of the list
    with pytest.raises(ValueError, match="unknown chat format"):
        normalise_chat({}, "xml")


def test_system_prompt_comes_from_the_row_then_the_source_and_never_twice():
    assert normalise_chat({"messages": [U, A]}, system="S")[0] == {"role": "system", "content": "S"}
    assert normalise_chat({"messages": [U, A], "system": "row"}, system="S")[0]["content"] == "row"
    own = [{"role": "system", "content": "own"}, U, A]
    assert normalise_chat({"messages": own}, system="S") == own


@pytest.mark.parametrize(
    "messages, reason",
    [
        ([], "not a list"),
        ("nonsense", "not a list"),
        ([A], "does not start with a user"),
        ([U], "does not end with an assistant"),
        ([U, U, A], "do not alternate"),
        ([U, A, {"role": "system", "content": "late"}, U, A], "system message is not first"),
        ([U, {"role": "assistant", "content": "  "}], "empty assistant message"),
        ([{"role": "narrator", "content": "x"}, A], "unknown role"),
        ([U, "oops"], "not a mapping"),
    ],
)
def test_unusable_conversations_are_rejected_with_a_reason(messages, reason):
    with pytest.raises(Rejected, match=reason):
        normalise_chat({"messages": messages})


def test_tool_results_may_sit_between_turns():
    tool = {"role": "tool", "content": "42"}
    call, answer = {"role": "assistant", "content": "calling"}, {"role": "assistant", "content": "it is 42"}
    assert validate_messages([U, call, tool, answer])  # call, result, answer
    assert normalise_chat({"conversations": [
        {"from": "human", "value": "hi"}, {"from": "gpt", "value": "calling"},
        {"from": "observation", "value": "42"}, {"from": "gpt", "value": "it is 42"},
    ]}, "sharegpt")[2] == tool
    with pytest.raises(Rejected, match="do not alternate"):
        validate_messages([U, call, answer])


def test_preference_rows():
    want = {"prompt": [U], "chosen": [{"role": "assistant", "content": "good"}],
            "rejected": [{"role": "assistant", "content": "bad"}]}
    assert normalise_preference({"prompt": "hi", "chosen": "good", "rejected": "bad"}) == want
    # TRL's own layout: no prompt column, chosen and rejected are whole conversations
    whole = {"chosen": [U, {"role": "assistant", "content": "good"}], "rejected": [U, {"role": "assistant", "content": "bad"}]}
    assert normalise_preference(whole) == want
    with_system = normalise_preference({"prompt": [U], "chosen": "good", "rejected": "bad"}, system="S")
    assert with_system["prompt"][0] == {"role": "system", "content": "S"}
    with pytest.raises(Rejected, match="the same"):
        normalise_preference({"prompt": "hi", "chosen": "x", "rejected": "x"})
    with pytest.raises(Rejected, match="empty rejected"):
        normalise_preference({"prompt": "hi", "chosen": "x", "rejected": ""})
    with pytest.raises(Rejected, match="no prompt"):
        normalise_preference({"chosen": [A], "rejected": [A]})


def test_chunk_text_cuts_at_breaks_and_keeps_everything():
    text = "\n\n".join(f"paragraph {i} " + "x" * 40 for i in range(20))
    pieces = chunk_text(text, 200)
    assert len(pieces) > 1 and all(len(p) <= 200 for p in pieces)
    assert all(p.startswith("paragraph") for p in pieces)  # cut between paragraphs, not inside one
    assert "".join(pieces).replace("\n", "") == text.replace("\n", "")
    assert chunk_text("short", 200) == ["short"] and chunk_text("  ", 200) == [] and chunk_text("abc") == ["abc"]
    overlapped = chunk_text("a" * 500, 200, overlap=50)
    assert sum(len(p) for p in overlapped) > 500 and all(len(p) <= 200 for p in overlapped)
    with pytest.raises(ValueError, match="chunk_overlap"):
        chunk_text("a" * 500, 200, overlap=150)


def test_split_depends_on_content_not_on_position():
    rows = [{"text": f"row {i}"} for i in range(400)]
    train, held_out = split_rows(rows, 0.1, None, seed=1)
    assert len(train) + len(held_out) == 400 and 15 < len(held_out) < 70
    assert not {r["text"] for r in train} & {r["text"] for r in held_out}
    # shuffled and extended input: every old row stays on its side
    again_train, again_eval = split_rows(rows[::-1] + [{"text": f"new {i}"} for i in range(100)], 0.1, None, seed=1)
    assert {r["text"] for r in held_out} <= {r["text"] for r in again_eval}
    assert {r["text"] for r in train} <= {r["text"] for r in again_train}
    assert {r["text"] for r in split_rows(rows, 0.1, None, seed=2)[1]} != {r["text"] for r in held_out}


def test_split_edges():
    rows = [{"text": f"row {i}"} for i in range(50)]
    assert split_rows(rows, 0.0, None, 0) == (rows, [])
    assert split_rows(rows[:1], 0.5, None, 0) == (rows[:1], [])
    assert len(split_rows(rows, 0.5, 3, 0)[1]) == 3  # capped
    assert len(split_rows(rows, 1e-9, None, 0)[1]) == 1  # a positive fraction always yields an eval row
    train, held_out = split_rows(rows[:2], 0.5, None, 0)
    assert len(train) >= 1 and len(train) + len(held_out) == 2


def test_collector_dedupes_and_counts_reasons():
    c = Collector(dedupe=True)

    def convert(raw):
        yield {"messages": normalise_chat(raw)}

    c.extend([{"messages": [U, A]}, {"messages": [U, {"role": "assistant", "content": "HELLO "}]}, {"messages": [A]}, {}], convert)
    assert len(c.rows) == 1 and c.duplicates == 1  # case and whitespace do not make a new row
    assert c.rejected == {"conversation does not start with a user message": 1, "messages are not a list": 1}
    keep = Collector(dedupe=False)
    keep.extend([{"messages": [U, A]}] * 3, convert)
    assert len(keep.rows) == 3 and keep.stats()["duplicates"] == 0
