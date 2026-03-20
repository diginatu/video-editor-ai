"""Tests for Stage 1.5 LLM filter."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from nagare_clip.stage1_5.llm_filter import (
    _apply_patches,
    _batch_lines,
    _format_batch,
    _parse_response,
    filter_transcript,
)


class TestBatchLines:
    def test_single_batch(self):
        lines = ["a", "b", "c"]
        batches = _batch_lines(lines, 10)
        assert len(batches) == 1
        assert batches[0] == [(0, "a"), (1, "b"), (2, "c")]

    def test_multiple_batches(self):
        lines = ["a", "b", "c", "d", "e"]
        batches = _batch_lines(lines, 2)
        assert len(batches) == 3
        assert batches[0] == [(0, "a"), (1, "b")]
        assert batches[1] == [(2, "c"), (3, "d")]
        assert batches[2] == [(4, "e")]

    def test_empty(self):
        assert _batch_lines([], 5) == []


class TestFormatBatch:
    def test_basic(self):
        batch = [(0, "hello"), (1, "world")]
        result = _format_batch(batch)
        assert result == "1: hello\n2: world"

    def test_preserves_content(self):
        batch = [(4, "line five")]
        result = _format_batch(batch)
        assert result == "5: line five"


class TestApplyPatches:
    def test_correction(self):
        result = _apply_patches(
            "{{急->今日}}はいい天気ですね",
            "急はいい天気ですね",
        )
        assert result == "今日はいい天気ですね"

    def test_deletion(self):
        result = _apply_patches(
            "{{えーと->}}はい",
            "えーとはい",
        )
        assert result == "はい"

    def test_whole_line_delete(self):
        result = _apply_patches(
            "{{(雑音)->}}",
            "(雑音)",
        )
        assert result == ""

    def test_multiple_patches(self):
        result = _apply_patches(
            "{{えーと->}}回り始めるようになっ{{てた->ていた}}と思います",
            "えーと回り始めるようになってたと思います",
        )
        assert result == "回り始めるようになっていたと思います"

    def test_no_patches_returns_text(self):
        result = _apply_patches(
            "これは修正不要です",
            "これは修正不要です",
        )
        assert result == "これは修正不要です"

    def test_old_not_in_original_returns_none(self):
        result = _apply_patches(
            "{{存在しない->修正}}テスト",
            "テスト",
        )
        assert result is None

    def test_empty_old_is_valid(self):
        # Empty old = insertion (always valid since "" is in any string)
        result = _apply_patches(
            "{{->追加}}テスト",
            "テスト",
        )
        assert result == "追加テスト"


class TestParseResponse:
    def test_basic_parse(self):
        batch = [(0, "あのー今日は"), (1, "えーとはい")]
        response = "1: {{あのー->}}今日は\n2: {{えーと->}}はい"
        result = _parse_response(response, batch)
        assert result[0] == "今日は"
        assert result[1] == "はい"

    def test_missing_line_skipped(self):
        batch = [(0, "line one"), (1, "line two")]
        response = "1: line one"
        result = _parse_response(response, batch)
        assert 1 not in result

    def test_no_numbered_lines_returns_empty(self):
        batch = [(0, "test")]
        response = "This is not a valid response"
        result = _parse_response(response, batch)
        assert result == {}

    def test_unchanged_lines_returned(self):
        batch = [(0, "unchanged")]
        response = "1: unchanged"
        result = _parse_response(response, batch)
        assert result[0] == "unchanged"


class TestFilterTranscript:
    def test_empty_input(self):
        assert filter_transcript([], {}) == []

    @patch("nagare_clip.stage1_5.llm_filter._call_llm")
    def test_successful_filter(self, mock_llm):
        mock_llm.return_value = "1: {{えーと->}}今日は\n2: line two"
        lines = ["えーと今日は", "line two"]
        cfg = {"batch_size": 10, "prompt": "fix"}
        result = filter_transcript(lines, cfg)
        assert result[0] == "今日は"
        assert result[1] == "line two"

    @patch("nagare_clip.stage1_5.llm_filter._call_llm")
    def test_api_failure_keeps_originals(self, mock_llm):
        mock_llm.side_effect = ConnectionError("timeout")
        lines = ["original line"]
        cfg = {"batch_size": 10, "prompt": "fix"}
        result = filter_transcript(lines, cfg)
        assert result == ["original line"]

    @patch("nagare_clip.stage1_5.llm_filter._call_llm")
    def test_garbled_response_keeps_originals(self, mock_llm):
        mock_llm.return_value = "garbled nonsense without line numbers"
        lines = ["original"]
        cfg = {"batch_size": 10, "prompt": "fix"}
        result = filter_transcript(lines, cfg)
        assert result == ["original"]

    @patch("nagare_clip.stage1_5.llm_filter._call_llm")
    def test_added_text_allowed(self, mock_llm):
        """LLM may add helpful text — no edit distance limit."""
        mock_llm.return_value = "1: {{短い->短い文を長い説明に変える}}テスト"
        lines = ["短いテスト"]
        cfg = {"batch_size": 10, "prompt": "fix"}
        result = filter_transcript(lines, cfg)
        assert result[0] == "短い文を長い説明に変えるテスト"
