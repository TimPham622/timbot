from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from html import escape
from pathlib import Path
import math
import re
from statistics import mean, median
from typing import Iterable

from .io_utils import write_json


WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9']*")
SENTENCE_RE = re.compile(r"[.!?]+")
EMOTE_RE = re.compile(r"(:[A-Za-z0-9_+-]+:|[xX][dD]+|lmao+|lmfao+|haha+|hehe+)", re.IGNORECASE)

STOP_WORDS = {
    "a", "about", "after", "all", "am", "an", "and", "are", "as", "at", "be", "been", "but",
    "by", "can", "do", "for", "from", "get", "got", "had", "has", "have", "he", "her", "here",
    "him", "his", "i", "if", "im", "in", "is", "it", "its", "just", "like", "me", "my", "no",
    "not", "of", "on", "or", "our", "out", "she", "so", "that", "the", "their", "them", "then",
    "there", "they", "this", "to", "too", "up", "was", "we", "were", "what", "when", "with",
    "you", "your", "youre",
}

POSITIVE = {
    "good", "great", "nice", "love", "loved", "best", "fun", "funny", "cool", "happy", "win",
    "winning", "perfect", "amazing", "awesome", "based", "glad", "yes", "yeah", "yep",
}
NEGATIVE = {
    "bad", "awful", "hate", "hated", "sad", "mad", "angry", "annoying", "tired", "dead",
    "lose", "losing", "lost", "worst", "terrible", "pain", "nope", "nah", "wrong",
}
PROFANITY = {
    "af", "ass", "bitch", "bullshit", "crap", "damn", "fuck", "fucked", "fucking", "hell", "shit",
}


def build_report(rows: Iterable[dict], out_dir: Path) -> dict:
    messages = list(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(messages)
    write_json(out_dir / "summary.json", summary)
    (out_dir / "report.html").write_text(render_html(summary), encoding="utf-8")
    return summary


def summarize(messages: list[dict]) -> dict:
    texts = [str(row.get("content", "")) for row in messages if row.get("content")]
    tokens = [token.lower() for text in texts for token in WORD_RE.findall(text)]
    meaningful_tokens = [token for token in tokens if token not in STOP_WORDS and len(token) > 1]
    timestamps = [_parse_ts(row.get("timestamp")) for row in messages]
    timestamps = [ts for ts in timestamps if ts is not None]

    word_counts = Counter(meaningful_tokens)
    bigrams = Counter(_ngrams(meaningful_tokens, 2))
    trigrams = Counter(_ngrams(meaningful_tokens, 3))
    lengths = [len(text) for text in texts]
    word_lengths = [len(WORD_RE.findall(text)) for text in texts]
    readability = [_flesch_kincaid(text) for text in texts if len(WORD_RE.findall(text)) >= 3]
    sentiment_scores = [_sentiment(text) for text in texts]
    profanity_hits = Counter(token for token in tokens if token in PROFANITY)
    emotes = Counter(match.group(0).lower() for text in texts for match in EMOTE_RE.finditer(text))
    punctuation = Counter(char for text in texts for char in text if char in "!?.,;:")

    return {
        "message_count": len(texts),
        "channel_count": len({row.get("channel_id") for row in messages if row.get("channel_id")}),
        "date_range": _date_range(timestamps),
        "length": {
            "characters_mean": _round(mean(lengths)) if lengths else 0,
            "characters_median": _round(median(lengths)) if lengths else 0,
            "words_mean": _round(mean(word_lengths)) if word_lengths else 0,
            "words_median": _round(median(word_lengths)) if word_lengths else 0,
        },
        "readability": {
            "flesch_kincaid_grade_mean": _round(mean(readability)) if readability else None,
        },
        "sentiment": {
            "mean": _round(mean(sentiment_scores)) if sentiment_scores else 0,
            "positive_messages": sum(score > 0 for score in sentiment_scores),
            "negative_messages": sum(score < 0 for score in sentiment_scores),
            "neutral_messages": sum(score == 0 for score in sentiment_scores),
        },
        "profanity": {
            "total": sum(profanity_hits.values()),
            "top": profanity_hits.most_common(25),
            "per_100_messages": _round(sum(profanity_hits.values()) / max(len(texts), 1) * 100),
        },
        "top_words": word_counts.most_common(100),
        "top_bigrams": [(" ".join(key), value) for key, value in bigrams.most_common(50)],
        "top_trigrams": [(" ".join(key), value) for key, value in trigrams.most_common(50)],
        "top_emotes": emotes.most_common(50),
        "punctuation": punctuation.most_common(),
        "by_hour": _by_hour(messages),
        "by_month": _by_month(messages),
        "sample_messages": _sample_messages(texts, 30),
    }


def render_html(summary: dict) -> str:
    by_hour = summary["by_hour"]
    by_month = summary["by_month"]
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Timbot Linguistic Report</title>
  <style>
    :root {{ color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    body {{ margin: 0; background: #f7f7f4; color: #1d2528; }}
    main {{ max-width: 1120px; margin: 0 auto; padding: 32px 20px 56px; }}
    h1 {{ font-size: 38px; margin: 0 0 8px; letter-spacing: 0; }}
    h2 {{ font-size: 20px; margin: 32px 0 12px; }}
    .muted {{ color: #586368; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }}
    .card {{ background: #ffffff; border: 1px solid #dde1df; border-radius: 8px; padding: 16px; }}
    .metric {{ font-size: 30px; font-weight: 750; }}
    table {{ width: 100%; border-collapse: collapse; background: #ffffff; border-radius: 8px; overflow: hidden; }}
    th, td {{ border-bottom: 1px solid #e7e9e7; padding: 9px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #ebf1ef; font-size: 13px; }}
    .cols {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 18px; }}
    .bars {{ display: flex; align-items: end; gap: 3px; height: 160px; padding: 12px 0; }}
    .bar {{ flex: 1; min-width: 4px; background: #2f6f73; border-radius: 3px 3px 0 0; }}
    .samples {{ columns: 2 360px; column-gap: 22px; }}
    .sample {{ break-inside: avoid; background: #ffffff; border: 1px solid #dde1df; border-radius: 8px; padding: 10px 12px; margin: 0 0 10px; }}
  </style>
</head>
<body>
<main>
  <h1>Timbot Linguistic Report</h1>
  <p class="muted">A compact read on your Discord texting style.</p>
  <section class="grid">
    {_metric("Messages", summary["message_count"])}
    {_metric("Channels", summary["channel_count"])}
    {_metric("Mean Words", summary["length"]["words_mean"])}
    {_metric("Mean Sentiment", summary["sentiment"]["mean"])}
    {_metric("Profanity / 100", summary["profanity"]["per_100_messages"])}
    {_metric("FK Grade", summary["readability"]["flesch_kincaid_grade_mean"])}
  </section>
  <div class="cols">
    <section>
      <h2>Top Words</h2>
      {_table(["Word", "Count"], summary["top_words"][:25])}
    </section>
    <section>
      <h2>Top Phrases</h2>
      {_table(["Phrase", "Count"], summary["top_bigrams"][:25])}
    </section>
  </div>
  <div class="cols">
    <section>
      <h2>Messages By Hour</h2>
      {_bar_chart([by_hour.get(str(hour), 0) for hour in range(24)])}
    </section>
    <section>
      <h2>Recent Monthly Volume</h2>
      {_bar_chart([count for _, count in list(by_month.items())[-24:]])}
    </section>
  </div>
  <div class="cols">
    <section>
      <h2>Emotes</h2>
      {_table(["Emote", "Count"], summary["top_emotes"][:20])}
    </section>
    <section>
      <h2>Profanity Index</h2>
      {_table(["Term", "Count"], summary["profanity"]["top"][:20])}
    </section>
  </div>
  <section>
    <h2>Sample Clean Messages</h2>
    <div class="samples">{''.join(f'<p class="sample">{escape(item)}</p>' for item in summary["sample_messages"])}</div>
  </section>
</main>
</body>
</html>
"""


def _metric(label: str, value: object) -> str:
    return f'<div class="card"><div class="muted">{escape(label)}</div><div class="metric">{escape(str(value))}</div></div>'


def _table(headers: list[str], rows: list[tuple]) -> str:
    body = "".join(
        "<tr>" + "".join(f"<td>{escape(str(cell))}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    head = "".join(f"<th>{escape(header)}</th>" for header in headers)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _bar_chart(values: list[int]) -> str:
    maximum = max(values or [1]) or 1
    bars = "".join(f'<div class="bar" style="height:{max(2, int(value / maximum * 150))}px" title="{value}"></div>' for value in values)
    return f'<div class="card"><div class="bars">{bars}</div></div>'


def _ngrams(tokens: list[str], n: int):
    for index in range(len(tokens) - n + 1):
        yield tuple(tokens[index : index + n])


def _sentiment(text: str) -> int:
    tokens = [token.lower() for token in WORD_RE.findall(text)]
    return sum(1 for token in tokens if token in POSITIVE) - sum(1 for token in tokens if token in NEGATIVE)


def _flesch_kincaid(text: str) -> float:
    words = WORD_RE.findall(text)
    sentences = max(1, len(SENTENCE_RE.findall(text)) or 1)
    syllables = sum(_syllables(word) for word in words)
    return 0.39 * (len(words) / sentences) + 11.8 * (syllables / max(len(words), 1)) - 15.59


def _syllables(word: str) -> int:
    word = word.lower()
    groups = re.findall(r"[aeiouy]+", word)
    count = len(groups)
    if word.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _date_range(timestamps: list[datetime]) -> dict[str, str | None]:
    if not timestamps:
        return {"start": None, "end": None}
    return {"start": min(timestamps).date().isoformat(), "end": max(timestamps).date().isoformat()}


def _by_hour(messages: list[dict]) -> dict[str, int]:
    counts = {str(hour): 0 for hour in range(24)}
    for row in messages:
        ts = _parse_ts(row.get("timestamp"))
        if ts is not None:
            counts[str(ts.hour)] += 1
    return counts


def _by_month(messages: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in messages:
        ts = _parse_ts(row.get("timestamp"))
        if ts is not None:
            counts[ts.strftime("%Y-%m")] += 1
    return dict(sorted(counts.items()))


def _sample_messages(texts: list[str], limit: int) -> list[str]:
    if len(texts) <= limit:
        return texts
    step = max(1, math.floor(len(texts) / limit))
    return [texts[index] for index in range(0, len(texts), step)][:limit]


def _round(value: float | int, places: int = 2) -> float:
    return round(float(value), places)
