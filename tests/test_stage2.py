import pytest
from unittest.mock import MagicMock, patch

import stage2_intervals as s


def make_tagger(morpheme_lists):
    """
    morpheme_lists: list of lists of surface strings, one per tagger call.
    """

    call_iter = iter(morpheme_lists)

    def side_effect(text):
        surfaces = next(call_iter)
        return [MagicMock(surface=surface) for surface in surfaces]

    return MagicMock(side_effect=side_effect)


# apply_margins


def test_apply_margins_basic_expansion():
    intervals = [{"start": 5.0, "end": 10.0}]
    result = s.apply_margins(intervals, 1.0, 1.0, 20.0)
    assert result == [{"start": 4.0, "end": 11.0}]


def test_apply_margins_clamps_bounds():
    intervals = [{"start": 0.5, "end": 19.5}]
    result = s.apply_margins(intervals, 1.0, 1.0, 20.0)
    assert result == [{"start": 0.0, "end": 20.0}]


def test_apply_margins_no_merge_after_expansion():
    intervals = [{"start": 5.0, "end": 10.0}, {"start": 13.0, "end": 18.0}]
    result = s.apply_margins(intervals, 1.0, 1.0, 30.0)
    assert result == [{"start": 4.0, "end": 11.0}, {"start": 12.0, "end": 19.0}]


def test_apply_margins_merges_after_expansion():
    intervals = [{"start": 5.0, "end": 10.0}, {"start": 11.5, "end": 18.0}]
    result = s.apply_margins(intervals, 1.0, 1.0, 30.0)
    assert result == [{"start": 4.0, "end": 19.0}]


def test_apply_margins_empty_input():
    assert s.apply_margins([], 1.0, 1.0, 20.0) == []


def test_apply_margins_zero_margins_no_change():
    intervals = [{"start": 5.0, "end": 10.0}]
    result = s.apply_margins(intervals, 0.0, 0.0, 20.0)
    assert result == intervals


def test_ensure_keep_covers_captions_adds_missing_ranges():
    keep_intervals = [{"start": 0.0, "end": 2.0}]
    captions = [
        {"start": 1.2, "end": 1.6, "text": "inside"},
        {"start": 3.0, "end": 3.4, "text": "outside"},
    ]

    result = s.ensure_keep_covers_captions(keep_intervals, captions, duration_sec=5.0)

    assert result == [{"start": 0.0, "end": 2.0}, {"start": 3.0, "end": 3.4}]


def test_ensure_keep_covers_captions_merges_overlaps():
    keep_intervals = [{"start": 5.0, "end": 6.0}]
    captions = [
        {"start": 5.5, "end": 6.5, "text": "overlap"},
        {"start": 6.5, "end": 7.0, "text": "touch"},
    ]

    result = s.ensure_keep_covers_captions(keep_intervals, captions, duration_sec=10.0)

    assert result == [{"start": 5.0, "end": 7.0}]


def test_enforce_min_keep_duration_expands_short_intervals():
    keep_intervals = [{"start": 5.0, "end": 5.2}, {"start": 9.8, "end": 10.0}]

    result = s.enforce_min_keep_duration(
        keep_intervals, min_keep=1.0, duration_sec=10.0
    )

    assert result == [{"start": 4.6, "end": 5.6}, {"start": 9.0, "end": 10.0}]


def test_enforce_min_keep_duration_merges_after_expansion():
    keep_intervals = [{"start": 5.0, "end": 5.2}, {"start": 5.9, "end": 6.1}]

    result = s.enforce_min_keep_duration(
        keep_intervals, min_keep=1.0, duration_sec=10.0
    )

    assert result == [{"start": 4.6, "end": 6.5}]


# flatten_words


def test_flatten_words_basic_two_morphemes():
    whisperx_data = {
        "segments": [
            {
                "start": 0.0,
                "end": 5.0,
                "text": "AB",
                "words": [
                    {"word": "A", "start": 0.0, "score": 0.9},
                    {"word": "B", "start": 3.0, "score": 0.9},
                ],
            }
        ]
    }
    tagger = make_tagger([["A", "B"]])
    with patch("fugashi.Tagger", return_value=tagger):
        words = s.flatten_words(whisperx_data)

    assert len(words) == 2
    assert words[0] == pytest.approx((0.0, 0.02, "A"), abs=1e-3)
    assert words[1] == pytest.approx((3.0, 3.02, "B"), abs=1e-3)


def test_flatten_words_intra_segment_no_silence():
    whisperx_data = {
        "segments": [
            {
                "start": 0.0,
                "end": 2.0,
                "text": "AB",
                "words": [
                    {"word": "A", "start": 0.0, "score": 0.9},
                    {"word": "B", "start": 0.02, "score": 0.9},
                ],
            }
        ]
    }
    tagger = make_tagger([["A", "B"]])
    with patch("fugashi.Tagger", return_value=tagger):
        words = s.flatten_words(whisperx_data)

    assert len(words) == 2
    assert words[0] == pytest.approx((0.0, 0.02, "A"), abs=1e-3)
    assert words[1] == pytest.approx((0.02, 0.04, "B"), abs=1e-3)


def test_flatten_words_inter_segment_silence_preserved():
    whisperx_data = {
        "segments": [
            {
                "start": 0.0,
                "end": 1.0,
                "text": "A",
                "words": [{"word": "A", "start": 0.0, "score": 0.9}],
            },
            {
                "start": 3.0,
                "end": 4.0,
                "text": "B",
                "words": [{"word": "B", "start": 3.0, "score": 0.9}],
            },
        ]
    }
    tagger = make_tagger([["A"], ["B"]])
    with patch("fugashi.Tagger", return_value=tagger):
        words = s.flatten_words(whisperx_data)

    assert len(words) == 2
    # NOTE: inter-segment end is last_char_start + _CHAR_EPS, NOT segment["end"].
    # The silence gap is detected via next_segment.first_char.start - this_segment.last_char.end,
    # which equals 3.0 - 0.02 = 2.98s here. segment["end"] is not used by build_morpheme_times.
    assert words[0][1] == pytest.approx(0.02, abs=1e-3)
    assert words[1][0] == pytest.approx(3.0, abs=1e-3)
    gap = words[1][0] - words[0][1]
    assert gap == pytest.approx(2.98, abs=1e-3)
    assert gap > 1.5


def test_flatten_words_placeholder_inherits_start():
    whisperx_data = {
        "segments": [
            {
                "start": 0.0,
                "end": 5.0,
                "text": "AB",
                "words": [
                    {"word": "A", "start": 1.0, "score": 0.9},
                    {"word": "B", "start": None, "score": 0.0},
                ],
            }
        ]
    }
    tagger = make_tagger([["AB"]])
    with patch("fugashi.Tagger", return_value=tagger):
        words = s.flatten_words(whisperx_data)

    assert len(words) == 1
    assert words[0] == pytest.approx((1.0, 1.02, "AB"), abs=1e-3)


# build_morpheme_times


def test_build_morpheme_times_ignores_inflated_end():
    whisperx_data = {
        "segments": [
            {
                "start": 5.0,
                "end": 20.0,
                "text": "はこ",
                "words": [
                    {"word": "は", "start": 12.856, "score": 0.983},
                    {"word": "こ", "start": 18.206, "score": 0.0},
                ],
            }
        ]
    }
    tagger = make_tagger([["は", "こ"]])
    morphemes = s.build_morpheme_times(whisperx_data, tagger)

    assert morphemes[0][1] == pytest.approx(12.876, abs=0.001)
    assert morphemes[1][0] == pytest.approx(18.206, abs=0.001)


def test_build_morpheme_times_real_silence_gap():
    whisperx_data = {
        "segments": [
            {
                "start": 30.0,
                "end": 55.0,
                "text": "はいちょっと",
                "words": [
                    {"word": "は", "start": 34.496, "score": 0.781},
                    {"word": "い", "start": 34.596, "score": 0.991},
                    {"word": "ち", "start": 53.545, "score": 0.0},
                    {"word": "ょ", "start": 53.565, "score": 0.0},
                    {"word": "っ", "start": 53.585, "score": 0.0},
                    {"word": "と", "start": 53.605, "score": 0.0},
                ],
            }
        ]
    }
    tagger = make_tagger([["はい", "ちょっと"]])
    morphemes = s.build_morpheme_times(whisperx_data, tagger)

    gap = morphemes[1][0] - morphemes[0][1]
    assert gap > 1.5


# collect_captions


def test_collect_captions_flush_and_preserve_silence_split_chunks():
    morphemes = [
        (0.0, 0.5, "あ"),
        (0.6, 1.0, "い"),
        (3.3, 3.5, "う"),
    ]
    keep_intervals = [{"start": 0.0, "end": 2.0}]

    captions = s.collect_captions(
        morphemes,
        keep_intervals,
        max_duration=4.0,
        max_morphemes=12,
        min_morphemes=1,
        min_duration=0.0,
        silence_flush=1.5,
    )

    assert len(captions) == 2
    assert captions[0]["text"] == "あい"
    assert captions[0]["start"] == pytest.approx(0.0)
    assert captions[0]["end"] == pytest.approx(1.0)
    assert captions[1]["text"] == "う"
    assert captions[1]["start"] == pytest.approx(3.3)
    assert captions[1]["end"] == pytest.approx(3.5)


def test_build_speech_spans_caps_inflated_word_end_for_gap_detection():
    whisperx_data = {
        "segments": [
            {
                "text": "ICに",
                "words": [
                    {"word": "I", "start": 75.053, "end": 75.073},
                    {"word": "C", "start": 75.073, "end": 80.06},
                    {"word": "に", "start": 80.06, "end": 80.08},
                ],
            }
        ]
    }

    spans = s.build_speech_spans(whisperx_data)

    ic_idx = next(
        i
        for i, (start, _) in enumerate(spans)
        if start == pytest.approx(75.073, abs=1e-3)
    )
    assert spans[ic_idx][1] == pytest.approx(75.673, abs=1e-3)
    gap = spans[ic_idx + 1][0] - spans[ic_idx][1]
    assert gap == pytest.approx(4.387, abs=1e-3)


def test_collect_captions_splits_on_keep_boundary_without_silence():
    morphemes = [
        (0.0, 0.3, "あ"),
        (0.3, 0.6, "い"),
        (0.6, 0.9, "う"),
    ]
    keep_intervals = [{"start": 0.0, "end": 0.6}]

    captions = s.collect_captions(
        morphemes,
        keep_intervals,
        max_duration=10.0,
        max_morphemes=12,
        min_morphemes=1,
        min_duration=0.0,
        silence_flush=10.0,
    )

    assert [cap["text"] for cap in captions] == ["あい", "う"]


# apply_margins integration


def test_apply_margins_real_intervals_no_merge():
    intervals = [
        {"start": 5.903, "end": 87.43},
        {"start": 89.857, "end": 109.502},
    ]
    result = s.apply_margins(intervals, 1.0, 1.0, 109.502)

    assert len(result) == 2
    assert result[0] == pytest.approx({"start": 4.903, "end": 88.43})
    assert result[1] == pytest.approx({"start": 88.857, "end": 109.502})


def test_apply_margins_real_intervals_merge_with_larger_padding():
    intervals = [
        {"start": 5.903, "end": 87.43},
        {"start": 89.857, "end": 109.502},
    ]
    result = s.apply_margins(intervals, 2.0, 2.0, 109.502)

    assert len(result) == 1
    assert result[0] == pytest.approx({"start": 3.903, "end": 109.502})


# Regression: multi-char morpheme hiding silence gap


def test_build_morpheme_times_multichar_morpheme_preserves_silence_gap():
    """
    Repro for real data: WhisperX char "あ" at 101.031 with inflated end 107.74,
    followed by "と" at 107.74.  fugashi groups them into a single morpheme "あと".
    build_morpheme_times must NOT produce a morpheme spanning (101.031, 107.76)
    because the 6.7 s silence between the two characters would be hidden.

    WhisperX misaligned the first char "あ" to 101.031; the real utterance of
    "あと" is at ~107.74 (the "と" cluster).  The fix detects the large
    intra-morpheme gap and snaps the morpheme start forward to the later
    cluster, so the silence gap appears *before* "あと" instead of being
    hidden inside it.
    """
    whisperx_data = {
        "segments": [
            {
                "start": 100.0,
                "end": 110.0,
                "text": "ねあとは",
                "words": [
                    {"word": "ね", "start": 101.011, "end": 101.031, "score": 0.0},
                    {"word": "あ", "start": 101.031, "end": 107.74, "score": 0.978},
                    {"word": "と", "start": 107.74, "end": 107.84, "score": 0.601},
                    {"word": "は", "start": 107.84, "end": 108.541, "score": 0.679},
                ],
            }
        ]
    }
    # fugashi tokenizes "ねあとは" -> ["ね", "あと", "は"]
    tagger = make_tagger([["ね", "あと", "は"]])
    morphemes = s.build_morpheme_times(whisperx_data, tagger)

    assert len(morphemes) == 3
    ne, ato, ha = morphemes

    assert ne[2] == "ね"
    assert ato[2] == "あと"
    assert ha[2] == "は"

    # "あと" should snap forward to the later cluster ("と" at 107.74),
    # because the gap between "あ" (101.031) and "と" (107.74) exceeds
    # _SILENCE_MAX_WORD_SPAN and "あ" is the misaligned character.
    assert ato[0] == pytest.approx(107.74, abs=1e-3)
    assert ato[1] == pytest.approx(107.76, abs=1e-3)

    # The silence gap should now appear between "ね" and "あと"
    gap = ato[0] - ne[1]
    assert gap > 1.5, (
        f"gap between 'ね' and 'あと' is only {gap:.3f} s; "
        f"silence is hidden inside the morpheme"
    )
