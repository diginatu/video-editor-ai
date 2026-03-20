"""Stage 1.5 CLI: LLM text filter for WhisperX transcriptions."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

from nagare_clip.config import get_effective_config
from nagare_clip.stage1_5.llm_filter import filter_transcript
from nagare_clip.stage1_5.sync_json import sync_text_to_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLM text filter for WhisperX transcriptions."
    )
    parser.add_argument(
        "--txt", required=True, dest="txt_path", help="WhisperX .txt path"
    )
    parser.add_argument(
        "--json", required=True, dest="json_path", help="WhisperX .json path"
    )
    parser.add_argument(
        "--output-txt",
        required=True,
        dest="output_txt",
        help="Output filtered .txt path",
    )
    parser.add_argument(
        "--output-json",
        required=True,
        dest="output_json",
        help="Output filtered .json path",
    )
    parser.add_argument(
        "--config",
        dest="config_path",
        default=None,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity (default: INFO)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cli_overrides: dict = {}
    if args.log_level is not None:
        cli_overrides.setdefault("general", {})["log_level"] = args.log_level

    config_path = Path(args.config_path) if args.config_path else None
    cfg = get_effective_config(config_path, cli_overrides)

    logging.basicConfig(
        level=cfg["general"]["log_level"], format="%(levelname)s: %(message)s"
    )

    s15 = cfg["stage1_5"]

    txt_path = Path(args.txt_path)
    json_path = Path(args.json_path)
    output_txt = Path(args.output_txt)
    output_json = Path(args.output_json)

    if not s15.get("enabled", False):
        logging.info("Stage 1.5 disabled, copying inputs to outputs")
        output_txt.parent.mkdir(parents=True, exist_ok=True)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(txt_path, output_txt)
        shutil.copy2(json_path, output_json)
        return

    # Read input files
    lines = txt_path.read_text(encoding="utf-8").splitlines()
    with json_path.open("r", encoding="utf-8") as f:
        json_data = json.load(f)

    logging.info("Stage 1.5: filtering %d lines with LLM", len(lines))

    # Filter text
    corrected_lines = filter_transcript(lines, s15)

    # Count changes
    changes = sum(1 for o, c in zip(lines, corrected_lines) if o != c)
    logging.info("Stage 1.5: %d/%d lines modified", changes, len(lines))

    # Sync to JSON
    filtered_json = sync_text_to_json(json_data, corrected_lines)

    # Write outputs
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    output_txt.write_text(
        "\n".join(corrected_lines) + "\n", encoding="utf-8"
    )

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(filtered_json, f, ensure_ascii=False, indent=2)
        f.write("\n")

    logging.info("Stage 1.5: wrote %s and %s", output_txt, output_json)


if __name__ == "__main__":
    main()
