#!/usr/bin/env python3

import argparse
import json
import logging
import sys
from pathlib import Path

import bpy


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def parse_blender_args(argv: list[str]) -> argparse.Namespace:
    if "--" not in argv:
        raise ValueError("Expected '--' before script arguments.")

    user_args = argv[argv.index("--") + 1 :]
    parser = argparse.ArgumentParser(
        description="Build rough-cut VSE layout from keep intervals."
    )
    parser.add_argument("--source", required=True, help="Source video file path")
    parser.add_argument("--intervals", required=True, help="Intervals JSON path")
    parser.add_argument("--output", required=True, help="Output .blend path")
    return parser.parse_args(user_args)


def sec_to_frames(seconds: float, fps: float) -> int:
    return int(round(seconds * fps))


def reset_scene() -> bpy.types.Scene:
    bpy.ops.wm.read_factory_settings(use_empty=False)
    bpy.ops.wm.read_homefile(app_template="Video_Editing")
    scene = bpy.context.scene
    scene.sequence_editor_create()
    return scene


def load_source_metadata(source_path: Path) -> tuple[float, int, int]:
    clip = bpy.data.movieclips.load(str(source_path))
    fps = float(clip.fps) if clip.fps and clip.fps > 0 else 30.0
    width, height = clip.size
    bpy.data.movieclips.remove(clip)
    return fps, int(width), int(height)


def main() -> None:
    args = parse_blender_args(sys.argv)

    source_path = Path(args.source).expanduser().resolve()
    intervals_path = Path(args.intervals).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    with intervals_path.open("r", encoding="utf-8") as f:
        intervals_data = json.load(f)

    keep_intervals = intervals_data.get("keep_intervals", [])
    scene = reset_scene()

    source_fps, source_width, source_height = load_source_metadata(source_path)
    fps_int = max(1, int(round(source_fps)))
    fps_base = fps_int / source_fps

    scene.render.fps = fps_int
    scene.render.fps_base = fps_base
    scene.render.resolution_x = source_width
    scene.render.resolution_y = source_height
    scene.frame_start = 1

    timeline_cursor = 1
    sequence_editor = scene.sequence_editor
    sequence_collection = getattr(sequence_editor, "sequences", None)
    if sequence_collection is None:
        sequence_collection = sequence_editor.strips
    effective_fps = scene.render.fps / scene.render.fps_base

    for idx, interval in enumerate(keep_intervals, start=1):
        start_sec = float(interval["start"])
        end_sec = float(interval["end"])
        if end_sec <= start_sec:
            continue

        src_start_frame = max(0, sec_to_frames(start_sec, effective_fps))
        src_end_frame = max(src_start_frame + 1, sec_to_frames(end_sec, effective_fps))

        logging.info(
            "Strip %d: source %.3fs–%.3fs → frames %d–%d",
            idx,
            start_sec,
            end_sec,
            src_start_frame,
            src_end_frame,
        )

        strip = sequence_collection.new_movie(
            name=f"keep_{idx:04d}",
            filepath=str(source_path),
            channel=1,
            frame_start=timeline_cursor,
        )

        full_duration = max(1, int(strip.frame_duration))
        bounded_start = min(src_start_frame, full_duration - 1)
        bounded_end = min(max(src_end_frame, bounded_start + 1), full_duration)
        keep_frame_count = bounded_end - bounded_start

        if bounded_start != src_start_frame or bounded_end != src_end_frame:
            logging.warning(
                "Strip %d: interval clamped to clip duration (%d frames). "
                "Requested frames %d–%d, applied %d–%d",
                idx,
                full_duration,
                src_start_frame,
                src_end_frame,
                bounded_start,
                bounded_end,
            )

        strip.animation_offset_start = bounded_start
        strip.animation_offset_end = full_duration - bounded_end

        sound_strip = sequence_collection.new_sound(
            name=f"keep_{idx:04d}_audio",
            filepath=str(source_path),
            channel=2,
            frame_start=timeline_cursor,
        )
        sound_strip.animation_offset_start = bounded_start
        sound_strip.animation_offset_end = full_duration - bounded_end

        if strip.frame_final_duration != keep_frame_count:
            logging.warning(
                "Strip %d: frame_final_duration=%d differs from keep_frame_count=%d",
                idx,
                strip.frame_final_duration,
                keep_frame_count,
            )

        logging.info(
            "Strip %d: animation_offset_start=%d animation_offset_end=%d "
            "keep_frames=%d timeline_cursor=%d",
            idx,
            bounded_start,
            full_duration - bounded_end,
            keep_frame_count,
            timeline_cursor,
        )

        timeline_cursor += keep_frame_count

    scene.frame_end = max(scene.frame_start, timeline_cursor - 1)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logging.info(
        "Done: %d strip(s) placed, scene ends at frame %d",
        len(keep_intervals),
        scene.frame_end,
    )
    bpy.ops.wm.save_as_mainfile(filepath=str(output_path))


if __name__ == "__main__":
    main()
