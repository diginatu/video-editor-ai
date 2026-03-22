"""Tests for caption frame computation (no Blender required)."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

# Stub bpy so we can import timeline without Blender
sys.modules.setdefault("bpy", MagicMock())

from nagare_clip.stage4.timeline import build_timeline_map, place_captions


def test_adjacent_captions_no_frame_overlap():
    """Captions sharing a boundary must not produce overlapping frames."""
    fps = 59.94  # NTSC fps triggers rounding overlap at boundary 3.46s

    tl_map = build_timeline_map(
        [{"start": 0.0, "end": 10.0}],
        effective_fps=fps,
        source_fps=fps,
    )

    captions = [
        {"start": 1.46, "end": 3.46, "text": "A"},
        {"start": 3.46, "end": 5.46, "text": "B"},
    ]

    # Collect frame_start and length from mock strips
    placed = []
    seq = MagicMock()

    def capture_effect(**kwargs):
        strip = MagicMock()
        placed.append(kwargs)
        return strip

    seq.new_effect = capture_effect

    place_captions(captions, tl_map, fps, seq)

    assert len(placed) == 2, f"Expected 2 caption strips, got {len(placed)}"

    end_a = placed[0]["frame_start"] + placed[0]["length"]
    start_b = placed[1]["frame_start"]
    assert end_a <= start_b, (
        f"Caption overlap: A ends at frame {end_a}, B starts at frame {start_b}"
    )
