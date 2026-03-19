# AGENTS.md

Agent guidance for this repository.

## Objective

Maintain and improve a 3-stage rough-cut pipeline:

1. WhisperX transcription in Docker
2. Keep-interval computation in Python
3. Blender VSE auto-layout in headless mode

Final deliverable is a `.blend` project for human editing.

## Hard Constraints

- Dependency management uses uv + pyproject.toml.
- Runtime NLP dependency is `ginza` + `ja_ginza` (spaCy-based).
- Preserve Stage 2 interval JSON as human-editable contract for Stage 3.
- Stage 3 must reference original media; do not re-encode/copy source media.

## Project Structure

```
src/video_editor_ai/          # Main Python package (src layout)
  cli.py                      # Stage 2 CLI entry point
  __main__.py                 # python -m video_editor_ai support
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
  stage2/                     # Stage 2 unit tests
  stage3/                     # Stage 3 tests (future)
```

## Current Runtime Quirks

- Stage 2 keep intervals are expanded by configurable pre/post margins (defaults 1.0s) and merged before Blender export.
- Stage 2 silence-based keep-interval detection uses WhisperX word timings (`word.start`/`word.end`) with a 0.6s per-word span cap so inflated token ends do not hide pauses.
- Stage 2 bunsetsu timing (`build_bunsetu_times` in `src/video_editor_ai/stage2/bunsetu.py`) uses `ginza.bunsetu_spans(doc)` so particles and auxiliaries attach to the preceding content word, producing natural subtitle line-break units. It detects large intra-bunsetsu character gaps (> 0.6s) caused by WhisperX misalignment and snaps the bunsetsu start forward to the later character cluster so the silence gap is not hidden inside a single bunsetsu.
- Stage 2 captions are chunked on GiNZA bunsetsu units with bunsetsu-level timing (`end = min(start+0.02s, next start)`), split on silence gaps and keep-boundary crossings; defaults are 12 bunsetsu, 4.0 seconds max, minimum 3 bunsetsu, min duration 1.5s, and silence flush at 1.5s. Bunsetsu units within a chunk are joined with a configurable separator (default `' '`, controlled by `--caption_bunsetu_separator`); a space between units enables word-wrap in Blender TEXT strips.
- Stage 2 captions are preserved as transcript chunks (not pre-filtered by keep intervals), and Stage 2 expands keep intervals to include caption spans so subtitles are retained in Stage 3.
- After caption-based expansion, Stage 2 re-applies `--min_keep` so tiny keep strips are expanded/merged when possible.

## Python Execution

Always use `uv run` to invoke Python tools in this repo. Examples:

```bash
uv run pytest
uv run python -m video_editor_ai.cli
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
./scripts/run_pipeline.sh "input/<sample>.mp4" ja
```

## Documentation Policy

When behavior changes, update all of:

- `README.md` (user-facing usage)
- `plan.md` (implementation/status)
- this `AGENTS.md` (agent guardrails)
