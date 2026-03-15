#!/usr/bin/env python3

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import yaml
from fugashi import Tagger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute keep intervals from WhisperX word-level JSON output."
    )
    parser.add_argument(
        "--json", required=True, dest="json_path", help="WhisperX JSON path"
    )
    parser.add_argument(
        "--config",
        required=True,
        dest="config_path",
        help="Filler words YAML path",
    )
    parser.add_argument(
        "--language", required=True, help="Language key in filler config"
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
        "--caption_max_morphemes",
        type=int,
        default=12,
        help="Maximum morphemes per caption chunk (default: 12)",
    )
    parser.add_argument(
        "--caption_max_duration",
        type=float,
        default=4.0,
        help="Maximum seconds per caption chunk (default: 4.0)",
    )
    parser.add_argument(
        "--caption_min_morphemes",
        type=int,
        default=3,
        help="Minimum morphemes before a chunk can be flushed (default: 3)",
    )
    parser.add_argument(
        "--output", required=True, dest="output_path", help="Output JSON path"
    )
    parser.add_argument(
        "--word_padding",
        type=float,
        default=0.1,
        help="Padding seconds before/after excluded filler words",
    )
    return parser.parse_args()


def normalize_word(text: str) -> str:
    lowered = text.lower().strip()
    cleaned_chars = []
    for ch in lowered:
        category = unicodedata.category(ch)
        if category.startswith("L") or category.startswith("N") or ch.isspace():
            cleaned_chars.append(ch)
    cleaned = "".join(cleaned_chars)
    return re.sub(r"\s+", " ", cleaned).strip()


def load_filler_set(config_path: Path, language: str) -> set[str]:
    with config_path.open("r", encoding="utf-8") as f:
        config: Dict[str, Sequence[str]] = yaml.safe_load(f) or {}

    if language not in config:
        available = ", ".join(sorted(config.keys())) or "(none)"
        raise ValueError(
            f"Language '{language}' not found in {config_path}. Available: {available}"
        )

    return {normalize_word(w) for w in config[language] if normalize_word(w)}


def flatten_words(whisperx_data: dict) -> List[Tuple[float, float, str]]:
    words: List[Tuple[float, float, str]] = []

    for segment in whisperx_data.get("segments", []):
        seg_end = float(segment.get("end") or 0.0)
        raw_seg: List[Tuple[float, str]] = []
        for word in segment.get("words", []):
            start = word.get("start")
            token = word.get("word", "")
            if start is None:
                continue
            raw_seg.append((float(start), token))
        raw_seg.sort(key=lambda item: item[0])

        for i, (start, token) in enumerate(raw_seg):
            if i + 1 < len(raw_seg):
                # Within segment: end = next char's start (no intra-segment gaps)
                end = min(start + 0.02, raw_seg[i + 1][0])
            else:
                # Last char of segment: use reliable segment-level end
                end = max(seg_end, start + 0.02)
            if end <= start:
                end = start + 0.02
            words.append((start, end, token))

    words.sort(key=lambda item: item[0])
    return words


def get_duration_sec(
    whisperx_data: dict, words: Sequence[Tuple[float, float, str]]
) -> float:
    max_end = 0.0
    if words:
        max_end = max(max_end, max(end for _, end, _ in words))

    for segment in whisperx_data.get("segments", []):
        end = segment.get("end")
        if end is not None:
            max_end = max(max_end, float(end))

    if isinstance(whisperx_data.get("duration"), (int, float)):
        max_end = max(max_end, float(whisperx_data["duration"]))

    return max_end


def merge_intervals(
    intervals: Iterable[Tuple[float, float]], epsilon: float = 1e-6
) -> List[List[float]]:
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    if not sorted_intervals:
        return []

    merged: List[List[float]] = [[sorted_intervals[0][0], sorted_intervals[0][1]]]
    for start, end in sorted_intervals[1:]:
        last = merged[-1]
        if start <= last[1] + epsilon:
            last[1] = max(last[1], end)
        else:
            merged.append([start, end])
    return merged


def invert_intervals(
    excludes: Sequence[Sequence[float]], duration_sec: float
) -> List[List[float]]:
    keeps: List[List[float]] = []
    cursor = 0.0
    for start, end in excludes:
        start_f = max(0.0, min(float(start), duration_sec))
        end_f = max(0.0, min(float(end), duration_sec))
        if start_f > cursor:
            keeps.append([cursor, start_f])
        cursor = max(cursor, end_f)
    if cursor < duration_sec:
        keeps.append([cursor, duration_sec])
    return keeps


def infer_source_file(whisperx_data: dict, json_path: Path) -> str:
    for key in ("source_file", "source", "audio", "audio_path", "file", "input_file"):
        value = whisperx_data.get(key)
        if isinstance(value, str) and value.strip():
            return Path(value).name
    return json_path.stem


def collect_captions(
    whisperx_data: dict,
    keep_intervals: List[dict],
    max_duration: float = 4.0,
    max_morphemes: int = 12,
    min_morphemes: int = 3,
) -> List[dict]:
    tagger = Tagger("-Owakati")
    captions = []

    for segment in whisperx_data.get("segments", []):
        seg_text = segment.get("text", "").strip()
        char_entries = segment.get("words", [])
        if not seg_text or not char_entries:
            continue

        # Build flat list of char start times aligned to seg_text characters.
        # Inherit last valid start for score=0.0 placeholder entries.
        char_starts: List[float] = []
        last_valid = float(char_entries[0].get("start") or 0.0)
        for entry in char_entries:
            s = entry.get("start")
            if s is not None:
                last_valid = float(s)
            char_starts.append(last_valid)

        # Align to seg_text length
        while len(char_starts) < len(seg_text):
            char_starts.append(char_starts[-1] if char_starts else 0.0)
        char_starts = char_starts[: len(seg_text)]

        # Map each morpheme to (surface, start, end) using char_starts.
        # morpheme start = char_starts[first char index]
        # morpheme end   = char_starts[last char index + 1] if available,
        #                  else char_starts[last char index] + 0.02
        morphemes: List[str] = [w.surface for w in tagger(seg_text)]
        morpheme_times: List[Tuple[str, float, float]] = []
        char_cursor = 0
        for morpheme in morphemes:
            m_len = len(morpheme)
            start_idx = min(char_cursor, len(char_starts) - 1)
            next_idx = min(char_cursor + m_len, len(char_starts))
            m_start = char_starts[start_idx]
            m_end = (
                char_starts[next_idx]
                if next_idx < len(char_starts)
                else char_starts[-1] + 0.02
            )
            if m_end <= m_start:
                m_end = m_start + 0.02
            morpheme_times.append((morpheme, m_start, m_end))
            char_cursor += m_len

        # Group morphemes into caption chunks
        chunk: List[str] = []
        chunk_start = 0.0
        chunk_end = 0.0

        for morpheme, m_start, m_end in morpheme_times:
            if chunk:
                speech_duration = chunk_end - chunk_start
                silence_gap = m_start - chunk_end
                should_flush = (
                    (speech_duration > max_duration or len(chunk) >= max_morphemes)
                    and len(chunk) >= min_morphemes
                ) or silence_gap > max_duration

                if should_flush:
                    captions.append(
                        {
                            "start": round(chunk_start, 3),
                            "end": round(chunk_end, 3),
                            "text": "".join(chunk),
                        }
                    )
                    chunk = []

            if not chunk:
                chunk_start = m_start
            chunk.append(morpheme)
            chunk_end = m_end

        if chunk:
            captions.append(
                {
                    "start": round(chunk_start, 3),
                    "end": round(chunk_end, 3),
                    "text": "".join(chunk),
                }
            )

    # Retain only captions that overlap at least one keep interval
    filtered = []
    for cap in captions:
        for iv in keep_intervals:
            if cap["start"] < iv["end"] and cap["end"] > iv["start"]:
                filtered.append(cap)
                break
    return filtered


def main() -> None:
    args = parse_args()

    json_path = Path(args.json_path)
    config_path = Path(args.config_path)
    output_path = Path(args.output_path)

    with json_path.open("r", encoding="utf-8") as f:
        whisperx_data = json.load(f)

    filler_set = load_filler_set(config_path, args.language)
    words = flatten_words(whisperx_data)
    duration_sec = get_duration_sec(whisperx_data, words)

    excludes: List[Tuple[float, float]] = []

    for start, end, token in words:
        normalized = normalize_word(token)
        if normalized in filler_set:
            excludes.append((start - args.word_padding, end + args.word_padding))

    for idx in range(len(words) - 1):
        current_end = words[idx][1]
        next_start = words[idx + 1][0]
        gap = next_start - current_end
        # print(f"[debug] gap between '{words[idx][2]}' and '{words[idx + 1][2]}': {gap:.3f}s")
        if gap > args.silence_threshold:
            excludes.append((current_end, next_start))

    if words and words[0][0] > args.silence_threshold:
        excludes.append((0.0, words[0][0]))

    if words and (duration_sec - words[-1][1]) > args.silence_threshold:
        excludes.append((words[-1][1], duration_sec))

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

    captions = collect_captions(
        whisperx_data,
        filtered_keep,
        max_duration=args.caption_max_duration,
        max_morphemes=args.caption_max_morphemes,
        min_morphemes=args.caption_min_morphemes,
    )

    output_data = {
        "source_file": infer_source_file(whisperx_data, json_path),
        "duration_sec": round(duration_sec, 3),
        "keep_intervals": filtered_keep,
        "captions": captions,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
