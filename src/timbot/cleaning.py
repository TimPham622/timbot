from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import re
from typing import Iterable

from .discord_export import DiscordMessage


URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_RE = re.compile(r"<@!?\d+>|<@&\d+>|<#\d+>")
CUSTOM_EMOJI_RE = re.compile(r"<a?:([A-Za-z0-9_]+):\d+>")
COMMAND_RE = re.compile(r"^\s*[!/?\\.-]\S+")
SYSTEMISH_RE = re.compile(
    r"\b(joined the server|started a call|missed a call|pinned a message|changed the channel)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CleanConfig:
    min_chars: int = 2
    max_chars: int = 600
    merge_window_seconds: int = 60
    keep_urls_as_token: bool = False
    drop_commands: bool = True


def clean_messages(messages: Iterable[DiscordMessage], config: CleanConfig = CleanConfig()) -> list[dict]:
    rows: list[dict] = []
    for msg in messages:
        cleaned = clean_text(msg.content, config)
        if cleaned is None:
            continue
        rows.append(
            {
                "message_id": msg.message_id,
                "channel_id": msg.channel_id,
                "timestamp": msg.timestamp,
                "content": cleaned,
                "source_file": msg.source_file,
                "author": msg.author,
            }
        )
    return merge_consecutive(rows, config.merge_window_seconds)


def clean_text(text: str, config: CleanConfig = CleanConfig()) -> str | None:
    if SYSTEMISH_RE.search(text):
        return None

    text = CUSTOM_EMOJI_RE.sub(r":\1:", text)
    text = MENTION_RE.sub("", text)
    text = URL_RE.sub(" URL " if config.keep_urls_as_token else " ", text)
    text = text.replace("\u200b", "")
    text = re.sub(r"\s+", " ", text).strip()

    if config.drop_commands and COMMAND_RE.match(text):
        return None
    if len(text) < config.min_chars or len(text) > config.max_chars:
        return None
    if not re.search(r"[A-Za-z0-9]", text):
        return None
    return text


def merge_consecutive(rows: list[dict], window_seconds: int) -> list[dict]:
    def sort_key(row: dict) -> tuple[str, str, str]:
        return (row.get("channel_id") or "", row.get("timestamp") or "", row.get("message_id") or "")

    merged: list[dict] = []
    for row in sorted(rows, key=sort_key):
        if not merged or not _can_merge(merged[-1], row, window_seconds):
            item = dict(row)
            item["merged_count"] = 1
            merged.append(item)
            continue

        merged[-1]["content"] = f"{merged[-1]['content']} {row['content']}"
        merged[-1]["merged_count"] += 1
        if row.get("timestamp"):
            merged[-1]["last_timestamp"] = row["timestamp"]
    return merged


def _can_merge(prev: dict, row: dict, window_seconds: int) -> bool:
    if prev.get("channel_id") != row.get("channel_id"):
        return False
    prev_ts = _parse_ts(prev.get("last_timestamp") or prev.get("timestamp"))
    row_ts = _parse_ts(row.get("timestamp"))
    if prev_ts is None or row_ts is None:
        return False
    delta = (row_ts - prev_ts).total_seconds()
    return 0 <= delta <= window_seconds


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None
