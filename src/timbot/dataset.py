from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

from .io_utils import read_jsonl, write_jsonl


def prepare_mlx_jsonl(
    rows: Iterable[dict],
    out_path: Path,
    validation_path: Path | None = None,
    max_examples: int | None = None,
    validation_fraction: float = 0.1,
    seed: int = 7,
) -> dict:
    examples = [row_to_mlx_text(row) for row in rows if usable_for_training(row)]
    rng = random.Random(seed)
    rng.shuffle(examples)
    if max_examples is not None:
        examples = examples[:max_examples]

    validation: list[dict] = []
    if validation_path and len(examples) >= 20 and validation_fraction > 0:
        validation_size = max(1, int(len(examples) * validation_fraction))
        validation = examples[:validation_size]
        examples = examples[validation_size:]

    train_count = write_jsonl(out_path, examples)
    validation_count = write_jsonl(validation_path, validation) if validation_path else 0
    return {"training_examples": train_count, "validation_examples": validation_count}


def convert_chat_jsonl_to_mlx(source_path: Path, mlx_path: Path) -> int:
    return write_jsonl(mlx_path, iter_chat_examples_as_mlx(source_path))


def convert_legacy_pair_to_mlx(
    train_source_path: Path = Path("data/openai_train.jsonl"),
    valid_source_path: Path = Path("data/openai_valid.jsonl"),
    train_out_path: Path = Path("data/train.jsonl"),
    valid_out_path: Path = Path("data/valid.jsonl"),
) -> dict:
    return {
        "train_examples": convert_chat_jsonl_to_mlx(train_source_path, train_out_path),
        "valid_examples": convert_chat_jsonl_to_mlx(valid_source_path, valid_out_path),
        "train_path": str(train_out_path),
        "valid_path": str(valid_out_path),
    }


def iter_chat_examples_as_mlx(source_path: Path):
    for row in read_jsonl(source_path):
        text = extract_assistant_text(row)
        if text:
            yield {"text": text}


def extract_assistant_text(row: dict) -> str | None:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return None
    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        if message.get("role") != "assistant":
            continue
        content = str(message.get("content", "")).strip()
        return content or None
    return None


def usable_for_training(row: dict) -> bool:
    content = str(row.get("content", "")).strip()
    if len(content) < 2 or len(content) > 500:
        return False
    return any(char.isalnum() for char in content)


def row_to_mlx_text(row: dict) -> dict:
    return {"text": str(row["content"]).strip()}
