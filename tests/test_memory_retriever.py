from timbot.memory_retriever import (
    build_augmented_prompt,
    query_memories,
    sanitize_memory,
    strip_augmented_prompt,
)


class FakeCollection:
    def query(self, query_texts, n_results):
        assert query_texts == ["hello"]
        assert n_results == 3
        return {"documents": [["one\none", "two", "three"]]}


def test_sanitize_memory_collapses_whitespace_and_truncates():
    memory = "a\n  b\t" + "c" * 400
    sanitized = sanitize_memory(memory, max_chars=20)
    assert "\n" not in sanitized
    assert len(sanitized) <= 20
    assert sanitized.endswith("...")


def test_query_memories_returns_sanitized_documents():
    assert query_memories(FakeCollection(), " hello ", 3) == ["one one", "two", "three"]


def test_build_augmented_prompt_exact_shape():
    prompt = build_augmented_prompt("User: hi", ["memory 1", "memory 2"])
    assert prompt == (
        "You are a digital clone of Tim. Here are some real things Tim has said in the past that might relate to the current conversation:\n"
        "- memory 1\n"
        "- memory 2\n\n"
        "User: hi\n"
        "Tim: "
    )


def test_strip_augmented_prompt_removes_prompt_prefix():
    prompt = build_augmented_prompt("hi", ["memory"])
    assert strip_augmented_prompt(prompt + " hello", prompt) == "hello"
