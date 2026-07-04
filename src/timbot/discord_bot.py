from __future__ import annotations

import os

from .env import load_env
from .local_mlx_client import DEFAULT_ADAPTER_PATH, DEFAULT_MLX_MODEL, generate_local_text, load_local_model


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

    @client.event
    async def on_ready():
        print(f"Logged in as {client.user}")
        print(f"Loaded {model_name} with adapters from {adapter_path}")

    @client.event
    async def on_message(message):
        if message.author.bot:
            return

        mentioned = client.user in message.mentions if client.user else False
        if mention_only and not mentioned:
            return

        prompt_message = message.content
        if client.user:
            prompt_message = prompt_message.replace(f"<@{client.user.id}>", "").replace(f"<@!{client.user.id}>", "").strip()
        async with message.channel.typing():
            reply = generate_local_text(model, tokenizer, prompt_message, max_tokens=max_tokens)
        if reply:
            await message.reply(reply[:1800], mention_author=False)

    client.run(token)
