from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from tqdm import tqdm


DEFAULT_COLLECTION = "timbot_memories"
DEFAULT_DB_PATH = Path("chroma_db")
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_JSONL_PATH = Path("data/messages.clean.jsonl")
METADATA_KEYS = ("message_id", "channel_id", "timestamp", "source_file", "merged_count")


@dataclass(frozen=True)
class MemoryBuildStats:
    rows_read: int
    missing_text_rows: int
    short_rows: int
    duplicate_rows: int
    added_rows: int
    batch_size: int
    db_path: str
    collection: str
    source_path: str


def build_memory_db(
    jsonl_path: Path = DEFAULT_JSONL_PATH,
    db_path: Path = DEFAULT_DB_PATH,
    collection_name: str = DEFAULT_COLLECTION,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 200,
) -> MemoryBuildStats:
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    collection = create_collection(db_path, collection_name, embedding_model)
    seen_hashes: set[str] = set()
    batch = Batch()
    rows_read = 0
    missing_text_rows = 0
    short_rows = 0
    duplicate_rows = 0
    added_rows = 0
    total_rows = estimate_jsonl_rows(jsonl_path)

    with jsonl_path.open("r", encoding="utf-8") as handle:
        with tqdm(total=total_rows, desc="Indexing memories", unit="msg") as progress:
            for line_no, line in enumerate(handle, start=1):
                progress.update(1)
                rows_read += 1
                row = parse_jsonl_line(line, jsonl_path, line_no)
                raw_text = extract_text(row)
                if raw_text is None:
                    missing_text_rows += 1
                    continue

                text = normalize_document_text(raw_text)
                if len(text) < 3:
                    short_rows += 1
                    continue

                record_id = message_id(text)
                if record_id in seen_hashes:
                    duplicate_rows += 1
                    continue
                seen_hashes.add(record_id)

                batch.add(
                    document=text,
                    record_id=record_id,
                    metadata=metadata_for_message(row),
                )
                if len(batch) >= batch_size:
                    added_rows += flush_batch(collection, batch)

    added_rows += flush_batch(collection, batch)
    return MemoryBuildStats(
        rows_read=rows_read,
        missing_text_rows=missing_text_rows,
        short_rows=short_rows,
        duplicate_rows=duplicate_rows,
        added_rows=added_rows,
        batch_size=batch_size,
        db_path=str(db_path),
        collection=collection_name,
        source_path=str(jsonl_path),
    )


def create_collection(db_path: Path, collection_name: str, embedding_model: str):
    try:
        import chromadb
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    except ImportError as exc:
        raise RuntimeError("Install memory dependencies with: pip install -e .") from exc

    db_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(db_path))
    embedding_function = SentenceTransformerEmbeddingFunction(model_name=embedding_model)
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function,
    )


def estimate_jsonl_rows(jsonl_path: Path) -> int:
    return sum(1 for _ in jsonl_path.open("r", encoding="utf-8"))


def parse_jsonl_line(line: str, path: Path, line_no: int) -> dict[str, Any]:
    try:
        row = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
    if not isinstance(row, dict):
        raise ValueError(f"{path}:{line_no}: expected JSON object")
    return row


def extract_text(row: dict[str, Any]) -> object | None:
    return row.get("text", row.get("content"))


def normalize_document_text(value: object) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def normalize_for_dedupe(text: str) -> str:
    return normalize_document_text(text).lower()


def message_id(text: str) -> str:
    return hashlib.sha256(normalize_for_dedupe(text).encode("utf-8")).hexdigest()


def metadata_for_message(row: dict[str, Any]) -> dict[str, str | int | float | bool]:
    metadata: dict[str, str | int | float | bool] = {}
    for key in METADATA_KEYS:
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, bool | int | float | str):
            metadata[key] = value
        else:
            metadata[key] = str(value)
    return metadata


def flush_batch(collection, batch: "Batch") -> int:
    if not batch:
        return 0
    collection.add(
        ids=batch.ids,
        documents=batch.documents,
        metadatas=batch.metadatas,
    )
    count = len(batch)
    batch.clear()
    return count


class Batch:
    def __init__(self) -> None:
        self.ids: list[str] = []
        self.documents: list[str] = []
        self.metadatas: list[dict] = []

    def __bool__(self) -> bool:
        return bool(self.ids)

    def __len__(self) -> int:
        return len(self.ids)

    def add(self, document: str, record_id: str, metadata: dict) -> None:
        self.ids.append(record_id)
        self.documents.append(document)
        self.metadatas.append(metadata)

    def clear(self) -> None:
        self.ids.clear()
        self.documents.clear()
        self.metadatas.clear()
