"""User-facing command entry point for NCRT."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
from typing import Any

from . import __version__


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_DIR / "config.json"

BANNER = r"""
 _   _  ____ ____ _____
| \ | |/ ___|  _ \_   _|
|  \| | |   | |_) || |
| |\  | |___|  _ < | |
|_| \_|\____|_| \_\|_|
"""

INTRO = (
    "NCRT (Next-generation Cognitive Red-Teaming) is an authorized LLM "
    "security-evaluation platform. It orchestrates multiple red-teaming "
    "strategies, model endpoints, automated judging, and reproducible outputs."
)


def print_header() -> None:
    print(BANNER.rstrip())
    print(f"NCRT v{__version__} - {INTRO}")
    print("Use only with authorized models and approved evaluation data.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="NCRT",
        add_help=False,
        description="NCRT command-line launcher",
    )
    parser.add_argument("-run", action="store_true", help="Run an evaluation using config.json.")
    parser.add_argument("-client", action="store_true", help="With -run, open the desktop client.")
    parser.add_argument("-nolog", action="store_true", help="With -run, hide detailed console logs.")
    parser.add_argument("-v", "--version", action="store_true", help="Show the NCRT version.")
    parser.add_argument("-help", "--help", action="store_true", help="Show this help message.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        metavar="PATH",
        help="Path to a JSON configuration file (default: project config.json).",
    )
    return parser


def print_help(parser: argparse.ArgumentParser) -> None:
    print("\nUsage")
    print("  NCRT                 Show the introduction")
    print("  NCRT -v              Show the version")
    print("  NCRT -help           Show this help")
    print("  NCRT -run            Run the evaluation configured in config.json")
    print("  NCRT -run -client    Open the NCRT desktop client")
    print("  NCRT -run -nolog     Run with an animated progress indicator only")
    print("  NCRT -run --config PATH  Use another configuration file")
    print("\nConfiguration")
    print("  Start from config.json in the project root. Endpoint keys may also use")
    print("  *_env fields so secrets remain outside the configuration file.")
    print("\nArguments accepted by the launcher")
    parser.print_help()


def _required_object(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"'{key}' must be a JSON object.")
    return value


def load_config(path_text: str) -> dict[str, Any]:
    path = Path(path_text).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Configuration file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as file:
            config = json.load(file)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}: {error.msg}") from error
    if not isinstance(config, dict):
        raise ValueError("The configuration root must be a JSON object.")

    for section in ("attacker", "victim", "judge", "run"):
        _required_object(config, section)
    return config


def _credential(role: dict[str, Any], role_name: str) -> str:
    env_name = role.get("api_key_env", "")
    if env_name:
        if not isinstance(env_name, str):
            raise ValueError("api_key_env must be a string.")
        # Temporary compatibility: API keys commonly contain '-' and therefore
        # cannot be valid environment-variable names. Prefer `api_key` for this.
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", env_name):
            return env_name
        value = os.environ.get(env_name, "")
        if value:
            return value
    value = role.get("api_key", "")
    if not isinstance(value, str):
        raise ValueError("api_key must be a string.")
    if value:
        return value
    if env_name:
        raise ValueError(
            f"{role_name}.api_key_env is configured, but the environment variable "
            "is not available to this NCRT process."
        )
    return value


def command_from_config(config: dict[str, Any]) -> list[str]:
    run = _required_object(config, "run")
    attacker = _required_object(config, "attacker")
    victim = _required_object(config, "victim")
    judge = _required_object(config, "judge")

    command = [sys.executable, str(PROJECT_DIR / "run.py")]
    field_map = (
        ("planner", "--planner"),
        ("goal", "--goal"),
        ("scale", "--scale"),
        ("rounds", "--rounds"),
        ("beam", "--beam"),
        ("branch", "--branch"),
        ("threshold", "--threshold"),
        ("seed", "--seed"),
        ("workers", "--workers"),
        ("output", "--output"),
    )
    for key, flag in field_map:
        value = run.get(key)
        if value not in (None, ""):
            command.extend((flag, str(value)))
    if run.get("compare", False):
        command.append("--compare")

    for section, prefix, role_name in (
        (attacker, "attack", "attacker"),
        (victim, "victim", "victim"),
        (judge, "judge", "judge"),
    ):
        model = section.get("model", "")
        base_url = section.get("base_url", "")
        if model:
            command.extend((f"--{prefix}-model", str(model)))
        if base_url:
            command.extend((f"--{prefix}-base-url", str(base_url)))
        key = _credential(section, role_name)
        key_flag = "--judge-key" if prefix == "judge" else f"--{prefix}-api-key"
        if key:
            command.extend((key_flag, key))
    return command


def _print_runner_line(line: str) -> None:
    """Forward child output without allowing a legacy encoding to end the run."""
    safe_line = line.replace("\ufffd", "?")
    try:
        print(safe_line, end="")
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or "utf-8"
        sys.stdout.buffer.write(safe_line.encode(encoding, errors="replace"))
        sys.stdout.flush()


def run_with_progress(command: list[str], quiet: bool) -> int:
    print("\nEach completed conversation is saved as an individual JSON file in output/.")

    process = subprocess.Popen(
        command,
        cwd=PROJECT_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None

    if not quiet:
        for line in process.stdout:
            _print_runner_line(line)
    else:
        # The legacy runner does not expose total work for every mode, so show
        # an indeterminate progress bar while conversation JSON files are written.
        stop = threading.Event()

        def animate() -> None:
            frames = ("[=       ]", "[==      ]", "[===     ]", "[ ====   ]", "[  ===== ]", "[   =====]", "[    ====]", "[     ===]", "[      ==]", "[       =]")
            index = 0
            while not stop.is_set():
                print(f"\r  Running evaluation {frames[index % len(frames)]}", end="", flush=True)
                index += 1
                stop.wait(0.15)

        worker = threading.Thread(target=animate, daemon=True)
        worker.start()
        for _ in process.stdout:
            pass
        stop.set()
        worker.join(timeout=1)
        print("\r  Running evaluation [complete]", flush=True)

    return process.wait()


def launch_client() -> int:
    print("\nStarting the NCRT desktop client...")
    return subprocess.call([sys.executable, str(PROJECT_DIR / "Client" / "launcher.py")], cwd=PROJECT_DIR)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    parsed = parser.parse_args(argv)
    print_header()

    if parsed.version:
        print(f"\nNCRT version {__version__}")
        return 0
    if parsed.help:
        print_help(parser)
        return 0
    if not parsed.run:
        if parsed.client or parsed.nolog:
            print("\n-client and -nolog require -run. See NCRT -help.", file=sys.stderr)
            return 2
        print("\nRun 'NCRT -help' to view commands, or 'NCRT -run' to start an evaluation.")
        return 0
    if parsed.client:
        return launch_client()

    try:
        config = load_config(parsed.config)
        command = command_from_config(config)
    except ValueError as error:
        print(f"\nConfiguration error: {error}", file=sys.stderr)
        print("Copy and adapt config.json before running NCRT -run.", file=sys.stderr)
        return 2

    print(f"\nUsing configuration: {Path(parsed.config).expanduser().resolve()}")
    return run_with_progress(command, quiet=parsed.nolog)
