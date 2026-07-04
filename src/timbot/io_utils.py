from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Any


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    ensure_parent(path)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL line") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            yield obj


def write_json(path: Path, data: Any) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
