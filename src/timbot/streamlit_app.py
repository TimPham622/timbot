from __future__ import annotations

from collections import Counter
from datetime import datetime
import argparse
import os
from pathlib import Path
import re
import sys

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".cache" / "matplotlib"))

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from timbot.analysis import STOP_WORDS, WORD_RE
from timbot.io_utils import read_jsonl


EXTRA_STOP_WORDS = {
    "lol",
    "lmao",
    "lmfao",
    "yeah",
    "yea",
    "yep",
    "nah",
    "u",
    "ur",
    "idk",
    "ive",
    "dont",
    "didnt",
    "cant",
    "wont",
    "thats",
}


def main() -> None:
    args = parse_args()
    st.set_page_config(page_title="Timbot Linguistic Dashboard", layout="wide")
    st.title("Timbot Linguistic Dashboard")

    with st.sidebar:
        st.header("Data")
        train_path = Path(st.text_input("MLX train JSONL", str(args.train)))
        messages_path = Path(st.text_input("Timestamped messages JSONL", str(args.messages)))
        max_rows = st.number_input("Max rows to read", min_value=1000, max_value=500000, value=150000, step=5000)
        custom_stop_words = st.text_area("Extra stop words", value="", help="Comma or newline separated.")

    train_df = load_train_jsonl(train_path, int(max_rows))
    if train_df.empty:
        st.error(f"No text rows found in {train_path}")
        return

    timestamp_df = load_timestamped_messages(messages_path, int(max_rows))
    stop_words = build_stop_words(custom_stop_words)
    top_words = count_top_words(train_df["text"].dropna().tolist(), stop_words, limit=100)

    metric_cols = st.columns(4)
    metric_cols[0].metric("Training Messages", f"{len(train_df):,}")
    metric_cols[1].metric("Timestamped Messages", f"{len(timestamp_df):,}")
    metric_cols[2].metric("Unique Top Words", f"{len(top_words):,}")
    metric_cols[3].metric("Mean Words / Message", f"{train_df['word_count'].mean():.1f}")

    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Top 100 Word Cloud")
        fig = make_wordcloud(top_words)
        st.pyplot(fig, clear_figure=True)
    with right:
        st.subheader("Top Words")
        st.dataframe(pd.DataFrame(top_words, columns=["word", "count"]), use_container_width=True, hide_index=True)

    st.subheader("Most Active Chatting Hours")
    if timestamp_df.empty:
        st.info("`train.jsonl` only contains text. Add `data/messages.clean.jsonl` to show hourly activity.")
    else:
        hourly = active_hours(timestamp_df)
        st.bar_chart(hourly, x="hour", y="messages", use_container_width=True)

    st.subheader("VADER Sentiment Timeline")
    if timestamp_df.empty:
        st.info("Sentiment timeline needs timestamps. The app uses `data/messages.clean.jsonl` when available.")
    else:
        timeline = vader_timeline(timestamp_df)
        if timeline.empty:
            st.info("No timestamped text rows were available for VADER sentiment.")
        else:
            st.line_chart(timeline, x="date", y=["compound", "rolling_7d"], use_container_width=True)
            st.dataframe(timeline.tail(30), use_container_width=True, hide_index=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--train", type=Path, default=Path("data/train.jsonl"))
    parser.add_argument("--messages", type=Path, default=Path("data/messages.clean.jsonl"))
    args, _ = parser.parse_known_args()
    return args


@st.cache_data(show_spinner=False)
def load_train_jsonl(path: Path, max_rows: int) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["text", "word_count"])
    rows = []
    for index, row in enumerate(read_jsonl(path)):
        if index >= max_rows:
            break
        text = str(row.get("text") or row.get("content") or "").strip()
        if text:
            rows.append({"text": text, "word_count": len(WORD_RE.findall(text))})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def load_timestamped_messages(path: Path, max_rows: int) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["text", "timestamp"])
    rows = []
    for index, row in enumerate(read_jsonl(path)):
        if index >= max_rows:
            break
        text = str(row.get("content") or row.get("text") or "").strip()
        timestamp = parse_timestamp(row.get("timestamp"))
        if text and timestamp is not None:
            rows.append({"text": text, "timestamp": timestamp})
    return pd.DataFrame(rows)


def build_stop_words(custom_stop_words: str) -> set[str]:
    custom = {
        item.strip().lower()
        for item in re.split(r"[,\n]+", custom_stop_words)
        if item.strip()
    }
    return STOP_WORDS | EXTRA_STOP_WORDS | custom


def count_top_words(texts: list[str], stop_words: set[str], limit: int = 100) -> list[tuple[str, int]]:
    counter: Counter[str] = Counter()
    for text in texts:
        for token in WORD_RE.findall(text.lower()):
            if len(token) > 1 and token not in stop_words:
                counter[token] += 1
    return counter.most_common(limit)


def make_wordcloud(top_words: list[tuple[str, int]]):
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.axis("off")
    if not top_words:
        ax.text(0.5, 0.5, "No words available", ha="center", va="center")
        return fig
    cloud = WordCloud(
        width=1400,
        height=760,
        background_color="white",
        colormap="viridis",
        max_words=100,
        prefer_horizontal=0.92,
        collocations=False,
    ).generate_from_frequencies(dict(top_words))
    ax.imshow(cloud, interpolation="bilinear")
    return fig


def active_hours(df: pd.DataFrame) -> pd.DataFrame:
    hourly = (
        df.assign(hour=df["timestamp"].dt.hour)
        .groupby("hour")
        .size()
        .reindex(range(24), fill_value=0)
        .rename("messages")
        .reset_index()
    )
    hourly["hour"] = hourly["hour"].map(lambda hour: f"{hour:02d}:00")
    return hourly


def vader_timeline(df: pd.DataFrame) -> pd.DataFrame:
    analyzer = SentimentIntensityAnalyzer()
    scored = df.copy()
    scored["date"] = scored["timestamp"].dt.date
    scored["compound"] = scored["text"].map(lambda text: analyzer.polarity_scores(text)["compound"])
    daily = scored.groupby("date", as_index=False)["compound"].mean()
    daily["rolling_7d"] = daily["compound"].rolling(7, min_periods=1).mean()
    return daily


def parse_timestamp(value: object) -> pd.Timestamp | None:
    if value is None:
        return None
    try:
        timestamp = pd.to_datetime(value, utc=True)
    except (TypeError, ValueError):
        return None
    if pd.isna(timestamp):
        return None
    return timestamp


if __name__ == "__main__":
    main()
