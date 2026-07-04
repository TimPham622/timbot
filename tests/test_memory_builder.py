from pathlib import Path

from timbot.memory_builder import (
    Batch,
    extract_text,
    message_id,
    metadata_for_message,
    normalize_document_text,
    normalize_for_dedupe,
    parse_jsonl_line,
)


def test_extract_text_prefers_text_then_content():
    assert extract_text({"text": "from text", "content": "from content"}) == "from text"
    assert extract_text({"content": "from content"}) == "from content"
    assert extract_text({}) is None


def test_dedupe_normalization_collapses_case_and_space():
    assert normalize_for_dedupe("  LOL \n") == normalize_for_dedupe("lol")


def test_message_id_is_deterministic_from_normalized_text():
    assert message_id("  Hello ") == message_id("hello")
    assert message_id("hello") != message_id("goodbye")


def test_metadata_removes_none_and_preserves_chroma_safe_values():
    metadata = metadata_for_message(
        {
            "message_id": "1",
            "channel_id": None,
            "timestamp": "2026-01-01T00:00:00+00:00",
            "source_file": Path("x"),
            "merged_count": 3,
            "ignored": "nope",
        }
    )
    assert metadata == {
        "message_id": "1",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "source_file": "x",
        "merged_count": 3,
    }


def test_parse_jsonl_line_requires_object():
    row = parse_jsonl_line('{"content": "hello"}', Path("messages.jsonl"), 1)
    assert row == {"content": "hello"}


def test_batch_tracks_parallel_lists():
    batch = Batch()
    batch.add("hello", "id1", {"x": 1})
    assert len(batch) == 1
    assert batch.ids == ["id1"]
    assert batch.documents == ["hello"]
    assert batch.metadatas == [{"x": 1}]
    batch.clear()
    assert not batch


def test_empty_contents_normalize_to_empty_string():
    assert normalize_document_text(None) == ""
