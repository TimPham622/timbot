# timbot

`timbot` is a local-first Discord language project. It takes a Discord data export, cleans and analyzes the messages, prepares a fine-tuning dataset, trains a local MLX LoRA adapter, builds a local ChromaDB memory index, and can run a Discord bot that replies in a style learned from historical messages.

The project is designed as both a practical personal tool and a portfolio project: it combines data cleaning, linguistic analysis, local model fine-tuning, Streamlit visualization, retrieval-augmented generation, and Discord bot deployment without relying on hosted model APIs.

## What It Does

- Parses a Discord export into clean JSONL datasets.
- Generates linguistic analysis reports and an interactive Streamlit dashboard.
- Prepares MLX-compatible LoRA training data.
- Fine-tunes a local model with Apple `mlx-lm`.
- Builds a local ChromaDB memory index from cleaned messages.
- Runs a local Discord bot with always-on RAG memory retrieval.

This project assumes you only have your own Discord export messages. That is enough to learn style, vocabulary, pacing, phrasing, and recurring topics. It is not enough to learn perfect conversational reply behavior, because Discord's privacy export contains your messages but not the full surrounding conversation from other people.

## Privacy

The repository intentionally does not track private Discord data or generated model artifacts. The following are ignored by Git:

- `Messages/`
- `data/`
- `reports/`
- `adapters/`
- `chroma_db/`
- `.cache/`
- `*.jsonl`
- `*.safetensors`

That means the code can live on GitHub while the raw messages, cleaned datasets, training files, LoRA adapters, reports, and vector database remain local.

## Hardware Requirements

`timbot` is heavily optimized for Apple Silicon Macs through Apple's [`mlx-lm`](https://github.com/ml-explore/mlx-lm) framework.

Recommended hardware:

- Apple Silicon Mac, preferably M-series Pro/Max/Ultra.
- Roughly 16GB-24GB+ of Unified Memory depending on the model and batch size.
- Enough disk space for model weights, adapters, ChromaDB files, and intermediate datasets.

The default model used in the examples is:

```text
meta-llama/Meta-Llama-3-8B-Instruct
```

An 8B model is realistic on higher-memory Apple Silicon machines, but training settings may need to be adjusted for smaller systems.

## External Setup

Some setup happens outside this repository.

### Discord Bot Token

Create a bot in the [Discord Developer Portal](https://discord.com/developers/applications), copy its bot token, and enable **Message Content Intent** so the bot can read messages it is mentioned in.

Set the token in your shell or `.env`:

```bash
export DISCORD_BOT_TOKEN="..."
```

### Hugging Face Access

Create a [Hugging Face](https://huggingface.co/) account, accept the Meta Llama 3 license for the model you plan to use, then authenticate locally:

```bash
huggingface-cli login
```

`meta-llama/Meta-Llama-3-8B-Instruct` may require Hugging Face access approval before MLX can download it.

## Installation

Create a virtual environment, install the project, and copy the example environment file:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
```

Optional environment values used by the bot:

```bash
export DISCORD_BOT_TOKEN="..."
export TIMBOT_MLX_MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
export TIMBOT_ADAPTER_PATH="adapters"
```

## Workflow

The project is easiest to use as a pipeline. Each stage creates files used by later stages.

### Step 1: Data Prep

Start by parsing the Discord export into a clean dataset. This reads the local `Messages/` folder, filters obvious noise, merges nearby consecutive messages, and writes both a raw and cleaned JSONL file.

```bash
timbot parse Messages --out data/messages.clean.jsonl --raw-out data/messages.raw.jsonl
```

The cleaned dataset is the main source for analysis, fine-tuning preparation, and memory indexing.

### Step 2: Analysis & Dashboard

Generate a static report first. This gives a quick overview of word usage, phrases, activity, sentiment, profanity index, readability, and sample messages.

```bash
timbot analyze data/messages.clean.jsonl --out-dir reports
```

For deeper exploration, run the Streamlit dashboard. It reads `data/train.jsonl` for word-cloud-style training text and uses `data/messages.clean.jsonl` for timestamp-based charts.

```bash
timbot dashboard --train data/train.jsonl --messages data/messages.clean.jsonl
```

The dashboard includes date range, hour range, channel filtering, text search, message length filtering, custom stop words, single-word or phrase word clouds, palette/background controls, sentiment timeline grouping, smoothing, channel volume, weekday activity, and filtered sample messages.

Timestamp charts and filters default to `Australia/Adelaide`, with a sidebar timezone selector for changing the view.

The **Phrases** tab supports custom n-gram ranges, natural phrase counting without stop-word removal, phrase search, top phrase tables, phrase timelines, simple phrase categories, and example messages containing the selected phrase.

### Step 3: Fine-Tuning

Prepare MLX LoRA data from the cleaned messages. The output uses simple `{"text": ...}` JSONL records suitable for `mlx_lm.lora`.

```bash
timbot prepare-mlx data/messages.clean.jsonl --out data/train.jsonl --validation-out data/valid.jsonl
```

If you previously generated old chat-style JSONL files, convert them into MLX format instead of regenerating from scratch.

```bash
timbot convert-legacy --train-source data/openai_train.jsonl --valid-source data/openai_valid.jsonl --train-out data/train.jsonl --valid-out data/valid.jsonl
```

Run local LoRA fine-tuning with MLX. This trains adapters into the `adapters/` directory.

```bash
timbot fine-tune --model meta-llama/Meta-Llama-3-8B-Instruct --data data --iters 1000 --batch-size 4 --adapter-path adapters
```

Because this project does not fetch surrounding context from other users, the fine-tuned model is best at autocomplete-style texting voice rather than perfectly deciding how to answer every possible message.

### Step 4: RAG Memory

Build a local ChromaDB memory index from the cleaned messages. This lets the bot retrieve relevant historical messages before generating a reply.

```bash
timbot build-memory data/messages.clean.jsonl --db-path chroma_db --batch-size 200
```

The memory builder reads `text` or `content`, skips messages shorter than 3 characters, stores raw message text as Chroma documents, and preserves metadata such as Discord message ID, timestamp, source file, channel ID, and merged count.

It deduplicates by deterministic SHA256 IDs derived from normalized message text before ingestion. The ChromaDB collection is named:

```text
timbot_memories
```

The vector database uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

### Step 5: Local Chat & Deployment

You can test the fine-tuned model locally from the terminal. This command uses the MLX model and adapters directly, without RAG.

```bash
timbot chat --model meta-llama/Meta-Llama-3-8B-Instruct --adapter-path adapters
```

Run the Discord bot when the adapters and ChromaDB memory index are ready.

```bash
timbot discord-bot --model meta-llama/Meta-Llama-3-8B-Instruct --adapter-path adapters
```

The Discord bot uses always-on local RAG. On startup it loads the MLX model, LoRA adapters, and the local `chroma_db` collection. For each incoming Discord message, it strips the bot mention, retrieves the top 3 related memories from ChromaDB, builds an augmented prompt, generates locally with MLX, and sends only the new reply back to Discord.

Startup fails fast if `chroma_db` or the `timbot_memories` collection is missing. If that happens, run:

```bash
timbot build-memory data/messages.clean.jsonl
```

## Useful Commands

Show all available commands:

```bash
timbot --help
```

Show options for a specific command:

```bash
timbot build-memory --help
timbot dashboard --help
timbot discord-bot --help
```

## Project Layout

```text
src/timbot/
  analysis.py          Static linguistic report generation
  cleaning.py          Message filtering and consecutive-message merging
  cli.py               Command-line entry point
  dataset.py           MLX JSONL preparation and legacy conversion
  discord_bot.py       Discord bot runtime
  discord_export.py    Discord export parsing
  local_mlx_client.py  MLX training/inference helpers
  memory_builder.py    ChromaDB memory ingestion
  memory_retriever.py  ChromaDB retrieval for RAG prompts
  streamlit_app.py     Interactive dashboard
```

## Quality Notes

For the highest quality, review the dashboard and report before training. If the top phrases, sample messages, or profanity index include junk, adjust the parse filters and regenerate the dataset.

This project is intentionally local-first. Model quality depends heavily on the quality and representativeness of the Discord export, the amount of cleaned training text, the LoRA settings, and whether the retrieved memories are relevant to the incoming Discord message.
