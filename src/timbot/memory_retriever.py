from __future__ import annotations

from pathlib import Path
import re
from typing import Any


MAX_MEMORY_CHARS = 300
DEFAULT_COLLECTION = "timbot_memories"
DEFAULT_DB_PATH = Path("chroma_db")
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_memory_collection(
    db_path: Path = DEFAULT_DB_PATH,
    collection_name: str = DEFAULT_COLLECTION,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
):
    if not db_path.exists():
        raise RuntimeError(
            f"Missing ChromaDB directory: {db_path}. "
            "Run `timbot build-memory data/messages.clean.jsonl` before starting the Discord bot."
        )

    try:
        import chromadb
        from chromadb.errors import NotFoundError
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    except ImportError as exc:
        raise RuntimeError("Install memory dependencies with: pip install -e .") from exc

    client = chromadb.PersistentClient(path=str(db_path))
    embedding_function = SentenceTransformerEmbeddingFunction(model_name=embedding_model)
    try:
        return client.get_collection(
            name=collection_name,
            embedding_function=embedding_function,
        )
    except NotFoundError as exc:
        raise RuntimeError(
            f"Missing ChromaDB collection `{collection_name}` in {db_path}. "
            "Run `timbot build-memory data/messages.clean.jsonl` before starting the Discord bot."
        ) from exc


def query_memories(collection: Any, query: str, n_results: int = 3) -> list[str]:
    query = query.strip()
    if not query:
        return []
    results = collection.query(query_texts=[query], n_results=n_results)
    documents = results.get("documents") or []
    if not documents:
        return []
    return [sanitize_memory(document) for document in documents[0] if document]


def sanitize_memory(memory: str, max_chars: int = MAX_MEMORY_CHARS) -> str:
    text = re.sub(r"\s+", " ", str(memory)).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def build_augmented_prompt(incoming_message: str, memories: list[str]) -> str:
    memory_lines = "\n".join(f"- {memory}" for memory in memories)
    if not memory_lines:
        memory_lines = "-"
    return (
        "You are a digital clone of Tim. Here are some real things Tim has said in the past that might relate to the current conversation:\n"
        f"{memory_lines}\n\n"
        f"User: {incoming_message.strip()}\n"
        "Tim:"
    )


def strip_augmented_prompt(output: str, prompt: str) -> str:
    text = str(output)
    if text.startswith(prompt):
        text = text[len(prompt) :]
    if "Tim:" in text:
        text = text.split("Tim:", 1)[-1]
    return text.strip()
