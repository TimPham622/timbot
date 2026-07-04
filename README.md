# timbot

Local tooling for turning a Discord data export into:

- a cleaned message dataset
- a linguistic analysis report
- MLX LoRA fine-tuning JSONL
- a small chatbot CLI
- an optional Discord bot wrapper

This project assumes you only have your own Discord export messages. That is enough to learn style, vocabulary, pacing, and phrasing. It is not enough to learn perfect reply behavior, because the export does not include what other people said.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
```

For the optional Discord bot, also set:

```bash
export DISCORD_BOT_TOKEN="..."
export TIMBOT_MLX_MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
export TIMBOT_ADAPTER_PATH="adapters"
```

## Workflow

Parse and clean your Discord export folder:

```bash
timbot parse Messages --out data/messages.clean.jsonl --raw-out data/messages.raw.jsonl
```

Generate the linguistic report:

```bash
timbot analyze data/messages.clean.jsonl --out-dir reports
```

Prepare MLX LoRA fine-tuning data:

```bash
timbot prepare-mlx data/messages.clean.jsonl --out data/train.jsonl --validation-out data/valid.jsonl
```

If you already generated the old chat JSONL files, convert them:

```bash
timbot convert-legacy --train-source data/openai_train.jsonl --valid-source data/openai_valid.jsonl --train-out data/train.jsonl --valid-out data/valid.jsonl
```

Run local MLX LoRA training:

```bash
timbot fine-tune --model meta-llama/Meta-Llama-3-8B-Instruct --data data --iters 1000 --batch-size 4 --adapter-path adapters
```

Chat with the model:

```bash
timbot chat --model meta-llama/Meta-Llama-3-8B-Instruct --adapter-path adapters
```

Run the Discord bot:

```bash
timbot discord-bot --model meta-llama/Meta-Llama-3-8B-Instruct --adapter-path adapters
```

Run the Streamlit linguistic dashboard:

```bash
timbot dashboard --train data/train.jsonl --messages data/messages.clean.jsonl
```

The dashboard reads `data/train.jsonl` for the word cloud. Since MLX `train.jsonl` only stores `{"text": ...}`, it uses `data/messages.clean.jsonl` for timestamp-based charts when that file is available.

Dashboard customizations include date range, hour range, channel filtering, text search, message length filtering, custom stop words, single-word or phrase word clouds, palette/background controls, sentiment timeline grouping, smoothing, channel volume, weekday activity, and filtered sample messages.

The Phrases tab supports custom n-gram ranges, natural phrase counting without stop-word removal, phrase search, top phrase tables, phrase timelines, simple phrase categories, and example messages containing the selected phrase.

## Quality Notes

For the highest quality, review `reports/report.html` before fine-tuning. If the top phrases, sample messages, or profanity index include junk, adjust the parse filters and regenerate the dataset.

Because this version does not fetch surrounding context, the fine-tuned model is best at autocomplete-style texting voice, not at knowing exactly how you would answer a specific friend's message.

`meta-llama/Meta-Llama-3-8B-Instruct` may require Hugging Face access approval and local Hugging Face authentication before MLX can download it.
