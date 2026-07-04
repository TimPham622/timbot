from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any

from .env import load_env


DEFAULT_MLX_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
DEFAULT_ADAPTER_PATH = Path("adapters")
DEFAULT_DATA_DIR = Path("data")

SPECIAL_TOKEN_RE = re.compile(r"<\|[^|]+?\|>|</?s>")


@dataclass(frozen=True)
class MlxTrainingConfig:
    model: str = DEFAULT_MLX_MODEL
    data_dir: Path = DEFAULT_DATA_DIR
    iters: int = 1000
    batch_size: int = 4
    adapter_path: Path = DEFAULT_ADAPTER_PATH


def run_lora_training(config: MlxTrainingConfig) -> None:
    if not (config.data_dir / "train.jsonl").exists():
        raise FileNotFoundError(f"Missing MLX training file: {config.data_dir / 'train.jsonl'}")

    command_name = shutil.which("mlx_lm.lora")
    if command_name is None:
        raise RuntimeError('Could not find `mlx_lm.lora`. Install dependencies with: pip install -e .')

    command = [
        command_name,
        "--model",
        config.model,
        "--train",
        "--data",
        str(config.data_dir),
        "--iters",
        str(config.iters),
        "--batch-size",
        str(config.batch_size),
        "--adapter-path",
        str(config.adapter_path),
    ]
    subprocess.run(command, check=True)


def load_local_model(model_name: str | None = None, adapter_path: str | Path | None = None) -> tuple[Any, Any]:
    load_env()
    try:
        from mlx_lm import load
    except ImportError as exc:
        raise RuntimeError('Install MLX support with: pip install "mlx-lm[train]"') from exc

    model_name = model_name or os.getenv("TIMBOT_MLX_MODEL") or DEFAULT_MLX_MODEL
    adapter_path = adapter_path or os.getenv("TIMBOT_ADAPTER_PATH") or DEFAULT_ADAPTER_PATH
    return load(model_name, adapter_path=str(adapter_path))


def generate_local_text(
    model: Any,
    tokenizer: Any,
    incoming_message: str,
    max_tokens: int = 100,
) -> str:
    try:
        from mlx_lm import generate
    except ImportError as exc:
        raise RuntimeError('Install MLX support with: pip install "mlx-lm[train]"') from exc

    prompt = build_prompt(incoming_message)
    output = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens)
    return clean_generation(output)


def build_prompt(incoming_message: str) -> str:
    return (
        "Continue this Discord chat in my texting style. "
        "Keep it short, casual, and natural.\n\n"
        f"Message: {incoming_message.strip()}\n"
        "Reply:"
    )


def clean_generation(output: str) -> str:
    text = SPECIAL_TOKEN_RE.sub("", output)
    if "Reply:" in text:
        text = text.split("Reply:", 1)[-1]
    lines = text.strip().splitlines()
    return lines[0].strip() if lines else ""
