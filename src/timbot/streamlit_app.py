from __future__ import annotations

from collections import Counter
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

PHRASE_CATEGORIES = {
    "Laughter": {"lol", "lmao", "lmfao", "haha", "hehe", "xd"},
    "Questions": {"what", "why", "how", "when", "where", "who", "which", "can", "do", "does", "did"},
    "Negation": {"no", "not", "dont", "don't", "didnt", "didn't", "cant", "can't", "wont", "won't", "nah"},
    "Emphasis": {"very", "really", "actually", "literally", "so", "too", "af"},
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

    train_df = load_train_jsonl(train_path, int(max_rows))
    if train_df.empty:
        st.error(f"No text rows found in {train_path}")
        return

    timestamp_df = load_timestamped_messages(messages_path, int(max_rows))
    controls = render_controls(train_df, timestamp_df)
    filtered_df = filter_timestamped_messages(timestamp_df, controls)
    text_source = select_text_source(train_df, filtered_df, controls)
    top_terms = count_top_terms(
        text_source,
        build_stop_words(controls["custom_stop_words"]),
        limit=controls["top_terms_limit"],
        min_word_len=controls["min_word_len"],
        ngram_size=controls["ngram_size"],
    )

    metric_cols = st.columns(4)
    metric_cols[0].metric("Training Messages", f"{len(train_df):,}")
    metric_cols[1].metric("Filtered Messages", f"{len(filtered_df):,}" if not timestamp_df.empty else "n/a")
    metric_cols[2].metric("Top Terms", f"{len(top_terms):,}")
    metric_cols[3].metric("Mean Words / Message", f"{mean_words(text_source):.1f}")

    left, right = st.columns([1.2, 1])
    with left:
        st.subheader(f"Top {controls['top_terms_limit']} Term Word Cloud")
        fig = make_wordcloud(
            top_terms,
            max_words=controls["top_terms_limit"],
            colormap=controls["colormap"],
            background_color=controls["background_color"],
            prefer_horizontal=controls["prefer_horizontal"],
        )
        st.pyplot(fig, clear_figure=True)
    with right:
        st.subheader("Top Terms")
        st.dataframe(pd.DataFrame(top_terms, columns=["term", "count"]), use_container_width=True, hide_index=True)

    if timestamp_df.empty:
        st.info("`train.jsonl` only contains text. Add `data/messages.clean.jsonl` to unlock time range, channel, hour, and sentiment filters.")
    else:
        activity_tab, sentiment_tab, phrases_tab, channel_tab, samples_tab = st.tabs(
            ["Activity", "Sentiment", "Phrases", "Channels", "Samples"]
        )

        with activity_tab:
            st.subheader("Most Active Chatting Hours")
            hourly = active_hours(filtered_df)
            st.bar_chart(hourly, x="hour", y="messages", use_container_width=True)

            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader("Weekday Activity")
                st.bar_chart(weekday_activity(filtered_df), x="weekday", y="messages", use_container_width=True)
            with col_b:
                st.subheader("Message Length Over Time")
                length_timeline = message_length_timeline(filtered_df, controls["sentiment_granularity"])
                if length_timeline.empty:
                    st.info("No messages in the selected range.")
                else:
                    st.line_chart(length_timeline, x="period", y="mean_words", use_container_width=True)

        with sentiment_tab:
            st.subheader("VADER Sentiment Timeline")
            timeline = vader_timeline(filtered_df, controls["sentiment_granularity"], controls["smoothing_window"])
            if timeline.empty:
                st.info("No timestamped text rows were available for VADER sentiment.")
            else:
                st.line_chart(timeline, x="period", y=["compound", "smoothed"], use_container_width=True)
                st.dataframe(timeline.tail(controls["timeline_rows"]), use_container_width=True, hide_index=True)

            st.subheader("Sentiment Mix")
            mix = sentiment_mix(filtered_df)
            if mix.empty:
                st.info("No messages in the selected range.")
            else:
                st.bar_chart(mix, x="sentiment", y="messages", use_container_width=True)

        with phrases_tab:
            render_phrase_tab(filtered_df, controls)

        with channel_tab:
            st.subheader("Most Active Channels")
            channels = top_channels(filtered_df, controls["channel_chart_limit"])
            if channels.empty:
                st.info("No channels in the selected range.")
            else:
                st.bar_chart(channels, x="channel_id", y="messages", use_container_width=True)

        with samples_tab:
            st.subheader("Filtered Message Samples")
            samples = sample_messages(filtered_df, controls["sample_count"])
            if samples.empty:
                st.info("No sample messages match the current filters.")
            else:
                st.dataframe(samples, use_container_width=True, hide_index=True)


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
    return pd.DataFrame(rows, columns=["text", "word_count"])


@st.cache_data(show_spinner=False)
def load_timestamped_messages(path: Path, max_rows: int) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["text", "timestamp", "channel_id", "word_count", "char_count"])
    rows = []
    for index, row in enumerate(read_jsonl(path)):
        if index >= max_rows:
            break
        text = str(row.get("content") or row.get("text") or "").strip()
        timestamp = parse_timestamp(row.get("timestamp"))
        if text and timestamp is not None:
            rows.append(
                {
                    "text": text,
                    "timestamp": timestamp,
                    "channel_id": str(row.get("channel_id") or "unknown"),
                    "word_count": len(WORD_RE.findall(text)),
                    "char_count": len(text),
                }
            )
    frame = pd.DataFrame(rows, columns=["text", "timestamp", "channel_id", "word_count", "char_count"])
    if not frame.empty:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame


def render_controls(train_df: pd.DataFrame, timestamp_df: pd.DataFrame) -> dict:
    with st.sidebar:
        st.header("Filters")
        if timestamp_df.empty:
            date_range = ()
            selected_channels: list[str] = []
        else:
            min_date = timestamp_df["timestamp"].dt.date.min()
            max_date = timestamp_df["timestamp"].dt.date.max()
            date_range = st.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
            channel_options = sorted(timestamp_df["channel_id"].dropna().unique().tolist())
            selected_channels = st.multiselect("Channels", channel_options, help="Leave empty to include every channel.")

        hour_range = st.slider("Hour range", 0, 23, (0, 23), format="%d:00")
        search_query = st.text_input("Text contains", "")
        min_chars, max_chars = st.slider("Message length", 1, 1000, (1, 600))

        st.header("Word Cloud")
        word_source = st.radio(
            "Word source",
            ["Filtered timestamped messages", "Full train.jsonl"],
            index=0 if not timestamp_df.empty else 1,
            disabled=timestamp_df.empty,
        )
        top_terms_limit = st.slider("Top terms", 25, 200, 100, step=25)
        min_word_len = st.slider("Minimum word length", 1, 10, 2)
        ngram_label = st.selectbox("Term type", ["Single words", "Two-word phrases", "Three-word phrases"])
        custom_stop_words = st.text_area("Extra stop words", value="", help="Comma or newline separated.")
        colormap = st.selectbox("Word cloud palette", ["viridis", "plasma", "cividis", "magma", "tab20", "Set2"])
        background_color = st.selectbox("Background", ["white", "black"])
        prefer_horizontal = st.slider("Horizontal word bias", 0.5, 1.0, 0.92, step=0.01)

        st.header("Sentiment")
        sentiment_granularity = st.selectbox("Timeline grouping", ["Daily", "Weekly", "Monthly"])
        smoothing_window = st.slider("Smoothing window", 1, 30, 7)
        timeline_rows = st.slider("Rows shown", 10, 100, 30, step=10)

        st.header("Tables")
        channel_chart_limit = st.slider("Channels shown", 5, 50, 15, step=5)
        sample_count = st.slider("Sample messages", 5, 100, 25, step=5)

        st.header("Phrases")
        phrase_min_n, phrase_max_n = st.slider("Phrase length", 2, 8, (2, 3))
        phrase_limit = st.slider("Top phrases", 10, 200, 50, step=10)
        phrase_min_count = st.slider("Minimum phrase count", 1, 50, 2)
        phrase_search = st.text_input("Phrase search", "")
        phrase_timeline_target = st.text_input("Timeline phrase", help="Leave blank to chart the top phrase.")
        phrase_context_count = st.slider("Phrase examples", 3, 50, 10)
        phrase_category = st.selectbox("Phrase category", ["All", *PHRASE_CATEGORIES.keys()])
        phrase_include_single_message_repeats = st.checkbox("Count repeated phrases within one message", value=True)

    return {
        "date_range": date_range,
        "selected_channels": selected_channels,
        "hour_range": hour_range,
        "search_query": search_query,
        "min_chars": min_chars,
        "max_chars": max_chars,
        "word_source": word_source,
        "top_terms_limit": top_terms_limit,
        "min_word_len": min_word_len,
        "ngram_size": {"Single words": 1, "Two-word phrases": 2, "Three-word phrases": 3}[ngram_label],
        "custom_stop_words": custom_stop_words,
        "colormap": colormap,
        "background_color": background_color,
        "prefer_horizontal": prefer_horizontal,
        "sentiment_granularity": sentiment_granularity,
        "smoothing_window": smoothing_window,
        "timeline_rows": timeline_rows,
        "channel_chart_limit": channel_chart_limit,
        "sample_count": sample_count,
        "phrase_min_n": phrase_min_n,
        "phrase_max_n": phrase_max_n,
        "phrase_limit": phrase_limit,
        "phrase_min_count": phrase_min_count,
        "phrase_search": phrase_search,
        "phrase_timeline_target": phrase_timeline_target,
        "phrase_context_count": phrase_context_count,
        "phrase_category": phrase_category,
        "phrase_include_single_message_repeats": phrase_include_single_message_repeats,
    }


def filter_timestamped_messages(df: pd.DataFrame, controls: dict) -> pd.DataFrame:
    if df.empty:
        return df

    filtered = df.copy()
    start_date, end_date = normalize_date_range(controls["date_range"], filtered)
    filtered = filtered[
        (filtered["timestamp"].dt.date >= start_date)
        & (filtered["timestamp"].dt.date <= end_date)
    ]

    start_hour, end_hour = controls["hour_range"]
    filtered = filtered[
        (filtered["timestamp"].dt.hour >= start_hour)
        & (filtered["timestamp"].dt.hour <= end_hour)
    ]

    if controls["selected_channels"]:
        filtered = filtered[filtered["channel_id"].isin(controls["selected_channels"])]

    query = controls["search_query"].strip()
    if query:
        filtered = filtered[filtered["text"].str.contains(re.escape(query), case=False, na=False)]

    filtered = filtered[
        (filtered["char_count"] >= controls["min_chars"])
        & (filtered["char_count"] <= controls["max_chars"])
    ]
    return filtered


def select_text_source(train_df: pd.DataFrame, filtered_df: pd.DataFrame, controls: dict) -> list[str]:
    if controls["word_source"] == "Filtered timestamped messages" and not filtered_df.empty:
        return filtered_df["text"].dropna().tolist()
    texts = train_df["text"].dropna().tolist()
    query = controls["search_query"].strip()
    if query:
        texts = [text for text in texts if query.lower() in text.lower()]
    return [text for text in texts if controls["min_chars"] <= len(text) <= controls["max_chars"]]


def normalize_date_range(value: object, df: pd.DataFrame):
    min_date = df["timestamp"].dt.date.min()
    max_date = df["timestamp"].dt.date.max()
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return value[0], value[1]
    if isinstance(value, (tuple, list)) and len(value) == 1:
        return value[0], value[0]
    return min_date, max_date


def build_stop_words(custom_stop_words: str) -> set[str]:
    custom = {
        item.strip().lower()
        for item in re.split(r"[,\n]+", custom_stop_words)
        if item.strip()
    }
    return STOP_WORDS | EXTRA_STOP_WORDS | custom


def count_top_terms(
    texts: list[str],
    stop_words: set[str],
    limit: int = 100,
    min_word_len: int = 2,
    ngram_size: int = 1,
) -> list[tuple[str, int]]:
    counter: Counter[str] = Counter()
    for text in texts:
        tokens = [
            token
            for token in WORD_RE.findall(text.lower())
            if len(token) >= min_word_len and token not in stop_words
        ]
        for term in iter_terms(tokens, ngram_size):
            counter[term] += 1
    return counter.most_common(limit)


def iter_terms(tokens: list[str], ngram_size: int):
    if ngram_size <= 1:
        yield from tokens
        return
    for index in range(len(tokens) - ngram_size + 1):
        yield " ".join(tokens[index : index + ngram_size])


def make_wordcloud(
    top_words: list[tuple[str, int]],
    max_words: int = 100,
    colormap: str = "viridis",
    background_color: str = "white",
    prefer_horizontal: float = 0.92,
):
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.axis("off")
    if not top_words:
        ax.text(0.5, 0.5, "No words available", ha="center", va="center")
        return fig
    cloud = WordCloud(
        width=1400,
        height=760,
        background_color=background_color,
        colormap=colormap,
        max_words=max_words,
        prefer_horizontal=prefer_horizontal,
        collocations=False,
    ).generate_from_frequencies(dict(top_words))
    ax.imshow(cloud, interpolation="bilinear")
    return fig


def active_hours(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame({"hour": [f"{hour:02d}:00" for hour in range(24)], "messages": [0] * 24})
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


def weekday_activity(df: pd.DataFrame) -> pd.DataFrame:
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    if df.empty:
        return pd.DataFrame({"weekday": names, "messages": [0] * 7})
    weekday = (
        df.assign(weekday=df["timestamp"].dt.dayofweek)
        .groupby("weekday")
        .size()
        .reindex(range(7), fill_value=0)
        .rename("messages")
        .reset_index()
    )
    weekday["weekday"] = weekday["weekday"].map(lambda index: names[index])
    return weekday


def vader_timeline(df: pd.DataFrame, granularity: str = "Daily", smoothing_window: int = 7) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["period", "compound", "smoothed"])
    analyzer = SentimentIntensityAnalyzer()
    scored = df.copy()
    scored["period"] = period_series(scored["timestamp"], granularity)
    scored["compound"] = scored["text"].map(lambda text: analyzer.polarity_scores(text)["compound"])
    timeline = scored.groupby("period", as_index=False)["compound"].mean()
    timeline["smoothed"] = timeline["compound"].rolling(smoothing_window, min_periods=1).mean()
    return timeline


def sentiment_mix(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["sentiment", "messages"])
    analyzer = SentimentIntensityAnalyzer()
    scores = df["text"].map(lambda text: analyzer.polarity_scores(text)["compound"])
    labels = scores.map(sentiment_label)
    mix = labels.value_counts().reindex(["positive", "neutral", "negative"], fill_value=0)
    frame = mix.rename("messages").reset_index()
    return frame.rename(columns={"index": "sentiment", "text": "sentiment"})


def sentiment_label(score: float) -> str:
    if score >= 0.05:
        return "positive"
    if score <= -0.05:
        return "negative"
    return "neutral"


def render_phrase_tab(df: pd.DataFrame, controls: dict) -> None:
    st.subheader("Natural Phrases")
    if df.empty:
        st.info("No messages match the current filters.")
        return

    phrases = phrase_table(
        df,
        min_n=controls["phrase_min_n"],
        max_n=controls["phrase_max_n"],
        min_count=controls["phrase_min_count"],
        search=controls["phrase_search"],
        category=controls["phrase_category"],
        include_single_message_repeats=controls["phrase_include_single_message_repeats"],
        limit=controls["phrase_limit"],
    )

    if phrases.empty:
        st.info("No phrases match the current phrase settings.")
        return

    total_occurrences = int(phrases["count"].sum())
    metric_cols = st.columns(4)
    metric_cols[0].metric("Shown Phrases", f"{len(phrases):,}")
    metric_cols[1].metric("Occurrences", f"{total_occurrences:,}")
    metric_cols[2].metric("Length Range", f"{controls['phrase_min_n']}-{controls['phrase_max_n']}")
    metric_cols[3].metric("Top Phrase", str(phrases.iloc[0]["phrase"]))

    chart_df = phrases.head(min(len(phrases), 30))
    st.bar_chart(chart_df, x="phrase", y="count", use_container_width=True)
    st.dataframe(phrases, use_container_width=True, hide_index=True)

    phrase_options = phrases["phrase"].tolist()
    requested = controls["phrase_timeline_target"].strip().lower()
    if requested and requested not in phrase_options:
        phrase_options = [requested, *phrase_options]
    selected_phrase = st.selectbox("Phrase to inspect", phrase_options)

    col_a, col_b = st.columns([1.1, 0.9])
    with col_a:
        st.subheader("Phrase Timeline")
        timeline = phrase_timeline(df, selected_phrase, controls["sentiment_granularity"])
        if timeline.empty:
            st.info("That phrase does not occur in the filtered range.")
        else:
            st.line_chart(timeline, x="period", y="occurrences", use_container_width=True)

    with col_b:
        st.subheader("Phrase Categories")
        category_counts = phrase_category_counts(phrases)
        st.bar_chart(category_counts, x="category", y="phrases", use_container_width=True)

    st.subheader("Examples In Context")
    examples = phrase_examples(df, selected_phrase, controls["phrase_context_count"])
    if examples.empty:
        st.info("No examples found for that phrase.")
    else:
        st.dataframe(examples, use_container_width=True, hide_index=True)


def phrase_table(
    df: pd.DataFrame,
    min_n: int,
    max_n: int,
    min_count: int,
    search: str,
    category: str,
    include_single_message_repeats: bool,
    limit: int,
) -> pd.DataFrame:
    rows = []
    message_count = max(len(df), 1)
    phrase_counts: Counter[str] = Counter()
    phrase_message_counts: Counter[str] = Counter()

    for text in df["text"].dropna():
        message_phrases = list(iter_message_phrases(str(text), min_n, max_n))
        counted_for_message = set(message_phrases)
        phrase_message_counts.update(counted_for_message)
        if include_single_message_repeats:
            phrase_counts.update(message_phrases)
        else:
            phrase_counts.update(counted_for_message)

    search = search.strip().lower()
    for phrase, count in phrase_counts.most_common():
        if count < min_count:
            continue
        if search and search not in phrase:
            continue
        phrase_category = classify_phrase(phrase)
        if category != "All" and phrase_category != category:
            continue
        rows.append(
            {
                "phrase": phrase,
                "count": count,
                "messages": phrase_message_counts[phrase],
                "message_share": round(phrase_message_counts[phrase] / message_count * 100, 2),
                "length": len(phrase.split()),
                "category": phrase_category,
            }
        )
        if len(rows) >= limit:
            break

    return pd.DataFrame(rows, columns=["phrase", "count", "messages", "message_share", "length", "category"])


def iter_message_phrases(text: str, min_n: int, max_n: int):
    tokens = [token.lower() for token in WORD_RE.findall(text)]
    max_n = min(max_n, len(tokens))
    for ngram_size in range(min_n, max_n + 1):
        for index in range(len(tokens) - ngram_size + 1):
            yield " ".join(tokens[index : index + ngram_size])


def classify_phrase(phrase: str) -> str:
    tokens = set(phrase.split())
    for category, markers in PHRASE_CATEGORIES.items():
        if tokens & markers:
            return category
    return "Other"


def phrase_category_counts(phrases: pd.DataFrame) -> pd.DataFrame:
    if phrases.empty:
        return pd.DataFrame(columns=["category", "phrases"])
    return phrases["category"].value_counts().rename("phrases").reset_index().rename(columns={"index": "category"})


def phrase_timeline(df: pd.DataFrame, phrase: str, granularity: str) -> pd.DataFrame:
    phrase = phrase.strip().lower()
    if not phrase:
        return pd.DataFrame(columns=["period", "occurrences"])
    rows = []
    for _, row in df.iterrows():
        occurrences = count_phrase_occurrences(str(row["text"]), phrase)
        if occurrences:
            rows.append({"timestamp": row["timestamp"], "occurrences": occurrences})
    if not rows:
        return pd.DataFrame(columns=["period", "occurrences"])
    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["period"] = period_series(frame["timestamp"], granularity)
    return frame.groupby("period", as_index=False)["occurrences"].sum()


def phrase_examples(df: pd.DataFrame, phrase: str, limit: int) -> pd.DataFrame:
    phrase = phrase.strip().lower()
    if not phrase:
        return pd.DataFrame(columns=["timestamp", "channel_id", "text"])
    rows = []
    for _, row in df.sort_values("timestamp", ascending=False).iterrows():
        if count_phrase_occurrences(str(row["text"]), phrase):
            rows.append(
                {
                    "timestamp": row["timestamp"],
                    "channel_id": row["channel_id"],
                    "text": row["text"],
                }
            )
        if len(rows) >= limit:
            break
    return pd.DataFrame(rows, columns=["timestamp", "channel_id", "text"])


def count_phrase_occurrences(text: str, phrase: str) -> int:
    phrase_tokens = phrase.split()
    if not phrase_tokens:
        return 0
    tokens = [token.lower() for token in WORD_RE.findall(text)]
    size = len(phrase_tokens)
    count = 0
    for index in range(len(tokens) - size + 1):
        if tokens[index : index + size] == phrase_tokens:
            count += 1
    return count


def period_series(series: pd.Series, granularity: str) -> pd.Series:
    series = series.dt.tz_convert(None) if getattr(series.dt, "tz", None) is not None else series
    if granularity == "Weekly":
        return series.dt.to_period("W").astype(str)
    if granularity == "Monthly":
        return series.dt.to_period("M").astype(str)
    return series.dt.date.astype(str)


def message_length_timeline(df: pd.DataFrame, granularity: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["period", "mean_words"])
    grouped = df.copy()
    grouped["period"] = period_series(grouped["timestamp"], granularity)
    return grouped.groupby("period", as_index=False)["word_count"].mean().rename(columns={"word_count": "mean_words"})


def top_channels(df: pd.DataFrame, limit: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["channel_id", "messages"])
    frame = df["channel_id"].value_counts().head(limit).rename("messages").reset_index()
    return frame.rename(columns={"index": "channel_id"})


def sample_messages(df: pd.DataFrame, count: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "channel_id", "text"])
    columns = ["timestamp", "channel_id", "text", "word_count"]
    return df.sort_values("timestamp", ascending=False)[columns].head(count)


def mean_words(texts: list[str]) -> float:
    if not texts:
        return 0.0
    return sum(len(WORD_RE.findall(text)) for text in texts) / len(texts)


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
