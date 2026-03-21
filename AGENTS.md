# AGENTS.md

Agent guidance for this repository.

## Objective

Maintain and improve a multi-stage rough-cut pipeline:

1. WhisperX transcription in Docker
1.5. (Optional) LLM text filter — corrects transcription errors
2. Keep-interval computation in Python
3. Blender VSE auto-layout in headless mode

Final deliverable is a `.blend` project for human editing.

## Pipeline Overview

`scripts/run_pipeline.sh` orchestrates all stages end-to-end. Use `--from-stage N` to skip earlier stages and reuse their outputs.

### Stage 1 — WhisperX Transcription

Speech-to-text with word-level alignment. Runs in a single Docker container for all source files to avoid model reload overhead.

- **Inputs:** source video files (mp4/mkv/mov/avi/webm)
- **Outputs:** `{stem}.json` (word timings), `{stem}.txt` (plain text)

### Stage 1.5 — LLM Text Filter (optional)

Corrects transcription errors using an OpenAI-compatible chat API (default: Ollama). Runs per source. Enabled via `stage1_5.enabled: true` in config.

- **Inputs:** `{stem}.txt`, `{stem}.json`
- **Outputs:** `{stem}_filtered.txt`, `{stem}_filtered.json`

### Stage 2 — Keep-Interval Computation

NLP analysis to decide which segments to keep. Uses GiNZA/spaCy for Japanese bunsetsu segmentation. Runs per source via `uv run`.

- **Inputs:** `{stem}.json` (or `{stem}_filtered.json` if Stage 1.5 ran)
- **Outputs:** `{stem}_intervals.json` (keep intervals + captions)

### Stage 3 — Blender VSE Layout

Auto-assembles the rough cut in headless Blender. References original media in-place (no re-encoding). Concatenates all sources onto a single timeline.

- **Inputs:** source video files, `{stem}_intervals.json` for each source
- **Outputs:** `{stem}_edited.blend` — ready for human editing

## Hard Constraints

- Dependency management uses uv + pyproject.toml.
- Runtime NLP dependency is `ginza` + `ja_ginza` (spaCy-based).
- Preserve Stage 2 interval JSON as human-editable contract for Stage 3.
- Stage 3 must reference original media; do not re-encode/copy source media.

## Project Structure

```
config.example.yml            # Documented YAML config template with all defaults
src/nagare_clip/          # Main Python package (src layout)
  config.py                   # Centralised config loading/merging (DEFAULTS dict)
  cli.py                      # Stage 2 CLI entry point
  __main__.py                 # python -m nagare_clip support
  stage1_5/                   # Stage 1.5 modules (LLM text filter)
    llm_filter.py             # LLM API calls and {{old->new}} patch parsing
    sync_json.py              # Sync corrected text back into WhisperX JSON
    cli.py                    # Stage 1.5 CLI entry point
  stage2/                     # Stage 2 modules
    bunsetu.py                # Bunsetsu-level timing (GiNZA)
    speech.py                 # Speech span extraction
    intervals.py              # Interval manipulation
    captions.py               # Caption chunking
    filler.py                 # Filler word config (unused at runtime)
    io.py                     # Source file inference
  stage3/                     # Stage 3 modules
    blender_cli.py            # Stage 3 CLI (runs inside Blender)
    scene.py                  # Blender scene setup
    timeline.py               # Strip and caption placement
scripts/
  run_pipeline.sh             # Main orchestrator
tests/
  test_config.py              # Config module unit tests
  stage1_5/                   # Stage 1.5 unit tests
  stage2/                     # Stage 2 unit tests
  stage3/                     # Stage 3 tests (future)
```

## Configuration System

All tunable parameters are centralised in `src/nagare_clip/config.py`:

- `DEFAULTS` dict holds the canonical defaults for all sections.
- `get_effective_config(config_path, cli_overrides)` merges DEFAULTS ← config file ← CLI overrides (highest priority wins).
- `config.example.yml` documents every key with its default value; copy it to start a project config.

**Priority order (highest first):** CLI flags > YAML config file > built-in defaults.

Both `cli.py` (Stage 2) and `blender_cli.py` (Stage 3) accept a `--config <path>` flag that is passed through by `scripts/run_pipeline.sh` when `--config` is provided.

`scripts/run_pipeline.sh` also reads `pipeline.*` and `stage1.*` config keys directly via Python/yaml for Docker Compose arguments that are not forwarded to a Python CLI.

## Current Runtime Quirks

- Stage 1.5 is an optional LLM text filter that corrects WhisperX transcription errors. It uses `{{old->new}}` inline patch syntax in `.txt` files, then syncs corrections back into the `.json` timing data. Enabled via `stage1_5.enabled: true` in config. Uses OpenAI-compatible chat completions API (default: Ollama at `localhost:11434`). Falls back to original text on any LLM or parse failure.
- Stage 2 keep intervals are expanded by configurable pre/post margins (defaults 1.0s) and merged before Blender export.
- Stage 2 silence-based keep-interval detection uses WhisperX word timings (`word.start`/`word.end`) with a 0.6s per-word span cap so inflated token ends do not hide pauses. The cap is controlled by `stage2.bunsetu.silence_max_word_span` in the config.
- Stage 2 bunsetsu timing (`build_bunsetu_times` in `src/nagare_clip/stage2/bunsetu.py`) uses `ginza.bunsetu_spans(doc)` so particles and auxiliaries attach to the preceding content word, producing natural subtitle line-break units. It detects large intra-bunsetsu character gaps (> 0.6s) caused by WhisperX misalignment and snaps the bunsetsu start forward to the later character cluster so the silence gap is not hidden inside a single bunsetsu. The gap threshold is `stage2.bunsetu.silence_max_word_span`; the end-offset epsilon is `stage2.bunsetu.char_eps`.
- Stage 2 captions are chunked on GiNZA bunsetsu units with bunsetsu-level timing (`end = min(start+char_eps, next start)`), split on silence gaps and keep-boundary crossings; defaults are 12 bunsetsu, 4.0 seconds max, minimum 3 bunsetsu, min duration 1.5s, and silence flush at 1.5s. Bunsetsu units within a chunk are joined with a configurable separator (default `' '`, controlled by `stage2.caption.bunsetu_separator`); a space between units enables word-wrap in Blender TEXT strips.
- Stage 2 captions are preserved as transcript chunks (not pre-filtered by keep intervals), and Stage 2 expands keep intervals to include caption spans so subtitles are retained in Stage 3.
- After caption-based expansion, Stage 2 re-applies `stage2.min_keep` so tiny keep strips are expanded/merged when possible.
- Stage 3 caption style (font size, alignment, position, shadow) is controlled by `stage3.caption_style.*` in the config.
- Stage 3 fallback FPS (used when source metadata is unavailable) is controlled by `stage3.default_fps`.
- Stage 3 supports multiple source files: `blender_cli.py` accepts repeated `--source`/`--intervals` flags; `place_strips()` and `build_timeline_map()` accept `start_cursor` and `idx_offset` to concatenate sources on a single timeline.
- `run_pipeline.sh` discovers all video files (`mp4`, `mkv`, `mov`, `avi`, `webm`) in `INPUT_VIDEOS_DIR` alphabetically when `--source` is not provided. Multiple `--source` flags are also accepted.
- `run_pipeline.sh` accepts `--from-stage N` (1, 1.5, 2, or 3) to skip expensive earlier stages and reuse their outputs. Also configurable via `pipeline.from_stage` in YAML config. When skipping stages, the script validates that required intermediate outputs exist.
- Stage 1 (WhisperX) runs in a **single container** for all source files, passing all relative paths as positional arguments. This avoids model reload overhead between videos. Stage 2 still loops per-source after the single Stage 1 completes.

## Python Execution

Always use `uv run` to invoke Python tools in this repo. Examples:

```bash
uv run pytest
uv run python -m nagare_clip.cli
```

## Preferred Validation

Validation runs automatically via the OpenCode `PostToolUse` hook
defined in `.opencode/plugin/validate.ts`. It triggers after every file
write/edit and runs:

- `docker compose config --services`
- `python -m py_compile` on all Stage 2 and Stage 3 Python modules
- `bash -n scripts/run_pipeline.sh`

If environment allows, also validate with a full run:

```bash
# Single source
./scripts/run_pipeline.sh --source input/<sample>.mp4 ja
# All videos in default directory
./scripts/run_pipeline.sh ja
```

## Documentation Policy

When behavior changes, update all of:

- `README.md` (user-facing usage)
- `plan.md` (implementation/status)
- this `AGENTS.md` (agent guardrails)
