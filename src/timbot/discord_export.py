from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Iterable, Iterator


CONTENT_KEYS = ("content", "contents", "Content", "Contents", "message", "Message", "text", "Text")
TIMESTAMP_KEYS = (
    "timestamp",
    "Timestamp",
    "created_at",
    "createdAt",
    "Date",
    "date",
    "sent_at",
    "SentAt",
)
ID_KEYS = ("id", "ID", "message_id", "Message ID", "messageId")
CHANNEL_KEYS = ("channel_id", "Channel ID", "channelId", "guild_channel_id")
AUTHOR_KEYS = ("author", "Author", "author_id", "Author ID", "user_id", "User ID", "username")


@dataclass(frozen=True)
class DiscordMessage:
    message_id: str | None
    channel_id: str | None
    timestamp: str | None
    content: str
    source_file: str
    author: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def iter_export_messages(export_path: Path) -> Iterator[DiscordMessage]:
    if export_path.is_file():
        files = [export_path]
    else:
        files = sorted(export_path.rglob("*.json"))

    for json_path in files:
        for record in _load_message_records(json_path):
            message = _record_to_message(record, json_path)
            if message is not None:
                yield message


def sorted_messages(messages: Iterable[DiscordMessage]) -> list[DiscordMessage]:
    return sorted(messages, key=lambda msg: (_timestamp_sort_key(msg.timestamp), msg.channel_id or "", msg.message_id or ""))


def _load_message_records(path: Path) -> Iterator[dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return

    yield from _walk_for_message_dicts(data)


def _walk_for_message_dicts(value: Any) -> Iterator[dict[str, Any]]:
    if isinstance(value, list):
        for item in value:
            yield from _walk_for_message_dicts(item)
        return

    if not isinstance(value, dict):
        return

    if any(key in value for key in CONTENT_KEYS):
        yield value
        return

    for child in value.values():
        yield from _walk_for_message_dicts(child)


def _record_to_message(record: dict[str, Any], path: Path) -> DiscordMessage | None:
    content = _first_string(record, CONTENT_KEYS)
    if content is None:
        return None

    content = _coerce_content(content)
    if content is None:
        return None

    timestamp = _first_string(record, TIMESTAMP_KEYS)
    channel_id = _first_string(record, CHANNEL_KEYS) or _channel_from_path(path)
    author = _author_to_string(_first_present(record, AUTHOR_KEYS))
    message_id = _first_string(record, ID_KEYS)

    return DiscordMessage(
        message_id=message_id,
        channel_id=channel_id,
        timestamp=_normalize_timestamp(timestamp),
        content=content,
        source_file=str(path),
        author=author,
    )


def _first_present(record: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return None


def _first_string(record: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    value = _first_present(record, keys)
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return None


def _author_to_string(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, dict):
        for key in ("username", "name", "id", "global_name"):
            if key in value and value[key]:
                return str(value[key])
    return str(value)


def _coerce_content(value: str) -> str | None:
    text = re.sub(r"\s+", " ", value).strip()
    return text or None


def _channel_from_path(path: Path) -> str | None:
    for part in reversed(path.parts):
        match = re.search(r"(\d{12,})", part)
        if match:
            return match.group(1)
    return path.parent.name if path.parent.name else None


def _normalize_timestamp(value: str | None) -> str | None:
    if not value:
        return None

    text = value.strip()
    if not text:
        return None

    candidates = [text, text.replace("Z", "+00:00")]
    for candidate in candidates:
        try:
            parsed = datetime.fromisoformat(candidate)
        except ValueError:
            continue
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).isoformat()

    return text


def _timestamp_sort_key(value: str | None) -> tuple[int, str]:
    if value is None:
        return (1, "")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return (0, value)
    return (0, parsed.isoformat())
