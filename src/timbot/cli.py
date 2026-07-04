from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from .analysis import build_report
from .cleaning import CleanConfig, clean_messages
from .dataset import convert_legacy_pair_to_mlx, prepare_mlx_jsonl
from .discord_export import iter_export_messages, sorted_messages
from .env import load_env
from .io_utils import read_jsonl, write_jsonl
from .local_mlx_client import DEFAULT_ADAPTER_PATH, DEFAULT_DATA_DIR, DEFAULT_MLX_MODEL, MlxTrainingConfig, generate_local_text, load_local_model, run_lora_training


def main(argv: list[str] | None = None) -> int:
    load_env()
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="timbot", description="Discord export analysis and texting-style chatbot tooling.")
    sub = parser.add_subparsers(required=True)

    parse_cmd = sub.add_parser("parse", help="Parse and clean a Discord JSON export folder.")
    parse_cmd.add_argument("export_path", type=Path)
    parse_cmd.add_argument("--out", type=Path, default=Path("data/messages.clean.jsonl"))
    parse_cmd.add_argument("--raw-out", type=Path, default=None)
    parse_cmd.add_argument("--min-chars", type=int, default=2)
    parse_cmd.add_argument("--max-chars", type=int, default=600)
    parse_cmd.add_argument("--merge-window-seconds", type=int, default=60)
    parse_cmd.add_argument("--keep-urls-as-token", action="store_true")
    parse_cmd.set_defaults(func=cmd_parse)

    analyze = sub.add_parser("analyze", help="Generate a linguistic HTML report and JSON summary.")
    analyze.add_argument("messages_jsonl", type=Path)
    analyze.add_argument("--out-dir", type=Path, default=Path("reports"))
    analyze.set_defaults(func=cmd_analyze)

    prepare = sub.add_parser("prepare-mlx", help="Create MLX LoRA text JSONL from cleaned messages.")
    prepare.add_argument("messages_jsonl", type=Path)
    prepare.add_argument("--out", type=Path, default=Path("data/train.jsonl"))
    prepare.add_argument("--validation-out", type=Path, default=Path("data/valid.jsonl"))
    prepare.add_argument("--max-examples", type=int, default=None)
    prepare.add_argument("--validation-fraction", type=float, default=0.1)
    prepare.set_defaults(func=cmd_prepare_mlx)

    convert = sub.add_parser("convert-legacy", help="Convert existing chat JSONL to MLX text JSONL.")
    convert.add_argument("--train-source", type=Path, default=Path("data/openai_train.jsonl"))
    convert.add_argument("--valid-source", type=Path, default=Path("data/openai_valid.jsonl"))
    convert.add_argument("--train-out", type=Path, default=Path("data/train.jsonl"))
    convert.add_argument("--valid-out", type=Path, default=Path("data/valid.jsonl"))
    convert.set_defaults(func=cmd_convert_legacy)

    fine_tune = sub.add_parser("fine-tune", help="Run a local MLX LoRA fine-tuning job.")
    fine_tune.add_argument("--model", default=DEFAULT_MLX_MODEL)
    fine_tune.add_argument("--data", type=Path, default=DEFAULT_DATA_DIR)
    fine_tune.add_argument("--iters", type=int, default=1000)
    fine_tune.add_argument("--batch-size", type=int, default=4)
    fine_tune.add_argument("--adapter-path", type=Path, default=DEFAULT_ADAPTER_PATH)
    fine_tune.set_defaults(func=cmd_fine_tune)

    chat = sub.add_parser("chat", help="Chat locally with the MLX model and LoRA adapters.")
    chat.add_argument("--model", default=DEFAULT_MLX_MODEL)
    chat.add_argument("--adapter-path", default=str(DEFAULT_ADAPTER_PATH))
    chat.add_argument("--max-tokens", type=int, default=100)
    chat.set_defaults(func=cmd_chat)

    bot = sub.add_parser("discord-bot", help="Run the optional Discord bot.")
    bot.add_argument("--model", default=DEFAULT_MLX_MODEL)
    bot.add_argument("--adapter-path", default=str(DEFAULT_ADAPTER_PATH))
    bot.add_argument("--max-tokens", type=int, default=100)
    bot.add_argument("--reply-to-all", action="store_true")
    bot.set_defaults(func=cmd_discord_bot)

    dashboard = sub.add_parser("dashboard", help="Run the Streamlit linguistic dashboard.")
    dashboard.add_argument("--train", type=Path, default=Path("data/train.jsonl"))
    dashboard.add_argument("--messages", type=Path, default=Path("data/messages.clean.jsonl"))
    dashboard.add_argument("--server-port", type=int, default=8501)
    dashboard.set_defaults(func=cmd_dashboard)

    return parser


def cmd_parse(args: argparse.Namespace) -> int:
    raw = sorted_messages(iter_export_messages(args.export_path))
    if args.raw_out:
        write_jsonl(args.raw_out, (message.to_dict() for message in raw))

    config = CleanConfig(
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        merge_window_seconds=args.merge_window_seconds,
        keep_urls_as_token=args.keep_urls_as_token,
    )
    cleaned = clean_messages(raw, config)
    count = write_jsonl(args.out, cleaned)
    print(f"Parsed {len(raw)} raw messages and wrote {count} clean messages to {args.out}")
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    rows = list(read_jsonl(args.messages_jsonl))
    summary = build_report(rows, args.out_dir)
    print(f"Wrote {args.out_dir / 'report.html'}")
    print(f"Messages analyzed: {summary['message_count']}")
    return 0


def cmd_prepare_mlx(args: argparse.Namespace) -> int:
    stats = prepare_mlx_jsonl(
        read_jsonl(args.messages_jsonl),
        out_path=args.out,
        validation_path=args.validation_out,
        max_examples=args.max_examples,
        validation_fraction=args.validation_fraction,
    )
    print(json.dumps(stats, indent=2))
    return 0


def cmd_convert_legacy(args: argparse.Namespace) -> int:
    result = convert_legacy_pair_to_mlx(args.train_source, args.valid_source, args.train_out, args.valid_out)
    print(json.dumps(result, indent=2))
    return 0


def cmd_fine_tune(args: argparse.Namespace) -> int:
    config = MlxTrainingConfig(
        model=args.model,
        data_dir=args.data,
        iters=args.iters,
        batch_size=args.batch_size,
        adapter_path=args.adapter_path,
    )
    run_lora_training(config)
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    model, tokenizer = load_local_model(args.model, args.adapter_path)
    print("Type a message. Ctrl-D exits.")
    while True:
        try:
            prompt = input("> ").strip()
        except EOFError:
            print()
            return 0
        if not prompt:
            continue
        reply = generate_local_text(model, tokenizer, prompt, max_tokens=args.max_tokens)
        print(reply)


def cmd_discord_bot(args: argparse.Namespace) -> int:
    from .discord_bot import run_discord_bot

    run_discord_bot(
        model_name=args.model,
        adapter_path=args.adapter_path,
        mention_only=not args.reply_to_all,
        max_tokens=args.max_tokens,
    )
    return 0


def cmd_dashboard(args: argparse.Namespace) -> int:
    app_path = Path(__file__).with_name("streamlit_app.py")
    src_root = str(app_path.parents[1])
    env = os.environ.copy()
    env["PYTHONPATH"] = src_root + os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else src_root
    env.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".cache" / "matplotlib"))
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(args.server_port),
        "--",
        "--train",
        str(args.train),
        "--messages",
        str(args.messages),
    ]
    subprocess.run(command, check=True, env=env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
