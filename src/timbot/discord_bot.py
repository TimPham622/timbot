from __future__ import annotations

import os
import re

from .env import load_env
from .local_mlx_client import DEFAULT_ADAPTER_PATH, DEFAULT_MLX_MODEL, clean_generation, load_local_model
from .memory_retriever import build_augmented_prompt, load_memory_collection, query_memories, strip_augmented_prompt


def run_discord_bot(
    model_name: str | None = None,
    adapter_path: str | None = None,
    mention_only: bool = True,
    max_tokens: int = 100,
) -> None:
    load_env()
    try:
        import discord
    except ImportError as exc:
        raise RuntimeError('Install Discord support with: pip install -e ".[discord]"') from exc

    token = os.getenv("DISCORD_BOT_TOKEN")
    if not token:
        raise RuntimeError("Set DISCORD_BOT_TOKEN before running the bot.")

    intents = discord.Intents.default()
    intents.message_content = True
    client = discord.Client(intents=intents)
    model_name = model_name or os.getenv("TIMBOT_MLX_MODEL") or DEFAULT_MLX_MODEL
    adapter_path = adapter_path or os.getenv("TIMBOT_ADAPTER_PATH") or str(DEFAULT_ADAPTER_PATH)
    model, tokenizer = load_local_model(model_name, adapter_path)
    memory_collection = load_memory_collection()

    @client.event
    async def on_ready():
        print(f"Logged in as {client.user}")
        print(f"Loaded {model_name} with adapters from {adapter_path}")
        print("Loaded ChromaDB collection timbot_memories from chroma_db")

    @client.event
    async def on_message(message):
        if message.author.bot:
            return

        mentioned = client.user in message.mentions if client.user else False
        if mention_only and not mentioned:
            return

        prompt_message = strip_bot_mention(message.content, client.user.id if client.user else None)
        async with message.channel.typing():
            conversation_log = await build_conversation_log(message.channel, limit=6, current_message=message)
            memories = query_memories(memory_collection, conversation_log or prompt_message, n_results=3)
            prompt = build_augmented_prompt(conversation_log or f"User: {prompt_message}", memories)
            try:
                from mlx_lm import generate
            except ImportError as exc:
                raise RuntimeError('Install MLX support with: pip install "mlx-lm[train]"') from exc
            output = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens)
            reply = clean_generation(strip_augmented_prompt(output, prompt))
        if reply:
            await message.reply(reply[:1800], mention_author=False)

    client.run(token)


def strip_bot_mention(content: str, bot_user_id: int | None) -> str:
    if bot_user_id is None:
        return strip_discord_mentions(content)
    text = content.replace(f"<@{bot_user_id}>", "").replace(f"<@!{bot_user_id}>", "")
    return strip_discord_mentions(text)


async def build_conversation_log(channel, limit: int = 6, current_message=None) -> str:
    messages = []
    seen_ids = set()
    async for history_message in channel.history(limit=limit):
        message_id = getattr(history_message, "id", id(history_message))
        if message_id in seen_ids:
            continue
        seen_ids.add(message_id)
        messages.append(history_message)

    if current_message is not None:
        current_id = getattr(current_message, "id", id(current_message))
        if current_id not in seen_ids:
            messages.append(current_message)

    messages.sort(key=lambda item: getattr(item, "created_at", None) or getattr(item, "id", 0))
    return "\n".join(format_message_for_log(message) for message in messages if clean_message_content(message.content))


def format_message_for_log(message) -> str:
    author = getattr(message, "author", None)
    label = "Tim" if getattr(author, "bot", False) else "User"
    return f"{label}: {clean_message_content(message.content)}"


def clean_message_content(content: str) -> str:
    return re.sub(r"\s+", " ", strip_discord_mentions(content)).strip()


def strip_discord_mentions(content: str) -> str:
    return re.sub(r"<@!?\d+>", "", content).strip()
