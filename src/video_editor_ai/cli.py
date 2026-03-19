"""Stage 2 CLI entry point: compute keep intervals from WhisperX JSON."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Tuple

import spacy

from video_editor_ai.stage2.bunsetu import build_bunsetu_times
from video_editor_ai.stage2.captions import collect_captions
from video_editor_ai.stage2.intervals import (
    apply_margins,
    enforce_min_keep_duration,
    ensure_keep_covers_captions,
    invert_intervals,
    merge_intervals,
)
from video_editor_ai.stage2.io import infer_source_file
from video_editor_ai.stage2.speech import build_speech_spans, get_duration_sec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute keep intervals from WhisperX word-level JSON output."
    )
    parser.add_argument(
        "--json", required=True, dest="json_path", help="WhisperX JSON path"
    )
    parser.add_argument(
        "--silence_threshold",
        type=float,
        default=1.5,
        help="Silence gap threshold in seconds",
    )
    parser.add_argument(
        "--min_keep",
        type=float,
        default=1.0,
        help="Minimum keep interval length in seconds",
    )
    parser.add_argument(
        "--pre_margin",
        type=float,
        default=1.0,
        help="Seconds to extend each keep interval before its start (default: 1.0)",
    )
    parser.add_argument(
        "--post_margin",
        type=float,
        default=1.0,
        help="Seconds to extend each keep interval after its end (default: 1.0)",
    )
    parser.add_argument(
        "--caption_max_bunsetu",
        type=int,
        default=12,
        help="Maximum bunsetsu units per caption chunk (default: 12)",
    )
    parser.add_argument(
        "--caption_max_duration",
        type=float,
        default=4.0,
        help="Maximum seconds per caption chunk (default: 4.0)",
    )
    parser.add_argument(
        "--caption_min_bunsetu",
        type=int,
        default=3,
        help="Minimum bunsetsu units before a chunk can be flushed (default: 3)",
    )
    parser.add_argument(
        "--caption_min_duration",
        type=float,
        default=1.5,
        help="Minimum seconds of speech before flushing a caption chunk (default: 1.5)",
    )
    parser.add_argument(
        "--caption_silence_flush",
        type=float,
        default=1.5,
        help="Silence duration that forces flushing the current caption chunk (default: 1.5)",
    )
    parser.add_argument(
        "--caption_bunsetu_separator",
        type=str,
        default=" ",
        help="Separator inserted between bunsetsu units in caption text; use empty string to disable (default: ' ')",
    )
    parser.add_argument(
        "--output", required=True, dest="output_path", help="Output JSON path"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity (default: INFO)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log_level, format="%(levelname)s: %(message)s")

    json_path = Path(args.json_path)
    output_path = Path(args.output_path)

    with json_path.open("r", encoding="utf-8") as f:
        whisperx_data = json.load(f)

    logging.info(
        "Loaded %d segment(s) from %s",
        len(whisperx_data.get("segments", [])),
        json_path.name,
    )

    nlp = spacy.load("ja_ginza")
    all_bunsetu_times = build_bunsetu_times(whisperx_data, nlp)

    words = all_bunsetu_times
    speech_spans = build_speech_spans(whisperx_data)
    duration_sec = get_duration_sec(whisperx_data, words)
    logging.info("Duration: %.1fs, bunsetsu: %d", duration_sec, len(all_bunsetu_times))

    excludes: List[Tuple[float, float]] = []

    silence_excludes = 0
    for idx in range(len(speech_spans) - 1):
        current_end = speech_spans[idx][1]
        next_start = speech_spans[idx + 1][0]
        gap = next_start - current_end
        if gap > args.silence_threshold:
            logging.debug(
                "Silence gap: %.3f-%.3f (%.3fs)", current_end, next_start, gap
            )
            excludes.append((current_end, next_start))
            silence_excludes += 1

    if speech_spans and speech_spans[0][0] > args.silence_threshold:
        logging.debug(
            "Silence gap: 0.000-%.3f (%.3fs) [leading]",
            speech_spans[0][0],
            speech_spans[0][0],
        )
        excludes.append((0.0, speech_spans[0][0]))
        silence_excludes += 1

    if speech_spans and (duration_sec - speech_spans[-1][1]) > args.silence_threshold:
        logging.debug(
            "Silence gap: %.3f-%.3f (%.3fs) [trailing]",
            speech_spans[-1][1],
            duration_sec,
            duration_sec - speech_spans[-1][1],
        )
        excludes.append((speech_spans[-1][1], duration_sec))
        silence_excludes += 1

    logging.info("Silence excluded: %d interval(s)", silence_excludes)

    bounded_excludes = [
        (max(0.0, start), min(duration_sec, end))
        for start, end in excludes
        if end > start
    ]
    merged_excludes = merge_intervals(bounded_excludes)
    keep_intervals = invert_intervals(merged_excludes, duration_sec)
    filtered_keep = [
        {"start": round(start, 3), "end": round(end, 3)}
        for start, end in keep_intervals
        if (end - start) >= args.min_keep
    ]
    logging.info("Keep intervals before margins: %d", len(filtered_keep))

    keep_intervals_dicts = apply_margins(
        filtered_keep,
        pre_margin=args.pre_margin,
        post_margin=args.post_margin,
        duration_sec=duration_sec,
    )
    logging.info(
        "After margins (pre=%.2fs post=%.2fs): %d interval(s)",
        args.pre_margin,
        args.post_margin,
        len(keep_intervals_dicts),
    )

    captions = collect_captions(
        all_bunsetu_times,
        keep_intervals_dicts,
        max_duration=args.caption_max_duration,
        max_bunsetu=args.caption_max_bunsetu,
        min_bunsetu=args.caption_min_bunsetu,
        min_duration=args.caption_min_duration,
        silence_flush=args.caption_silence_flush,
        duration_sec=duration_sec,
        bunsetu_separator=args.caption_bunsetu_separator,
    )
    logging.info("Captions: %d chunk(s)", len(captions))

    keep_intervals_dicts = ensure_keep_covers_captions(
        keep_intervals_dicts,
        captions,
        duration_sec,
    )
    logging.info("After caption expansion: %d interval(s)", len(keep_intervals_dicts))

    keep_intervals_dicts = enforce_min_keep_duration(
        keep_intervals_dicts,
        args.min_keep,
        duration_sec,
    )
    logging.info(
        "After min_keep enforcement: %d interval(s)", len(keep_intervals_dicts)
    )

    output_data = {
        "source_file": infer_source_file(whisperx_data, json_path),
        "duration_sec": round(duration_sec, 3),
        "keep_intervals": keep_intervals_dicts,
        "captions": captions,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    logging.info("Writing output to %s", output_path)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
