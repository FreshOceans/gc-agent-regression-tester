# AGENTS.md

## Purpose
This file is a working guide for engineers/agents modifying the **Regression Test Harness** repository.
Use it to preserve existing behavior while shipping changes quickly and safely.

## Project Snapshot
- Product: Genesys Cloud regression harness with LLM-as-judge workflows.
- Main run modes:
  - `standard` (intent/goal validation)
  - `journey` (contained + fulfilled path validation)
  - `analytics_journey` (evaluate-now analytics flow)
- Primary UI: `src/web_app.py` + `templates/home.html` + `templates/results.html`.
- Core execution path: `src/orchestrator.py` + `src/conversation_runner.py`.

## General Instructions (Mapped To This Python Repo)
- After finishing edits, run a Python sanity pass:
  - `python3 -m compileall src tests`
- Run targeted tests for the files/area you changed:
  - `.venv/bin/pytest tests/<target_test_file>.py -q`
- Run UI route/regression tests after any template or web behavior change:
  - `.venv/bin/pytest tests/test_web_app.py -q`
- Run full tests before handoff:
  - `.venv/bin/pytest -q`
- Use Playwright-based browser verification for user-facing UI/UX changes (Home, Results, Transcript Suite, Analytics pane).

### JavaScript Workflow Mapping
- `bun run format` / `bun run lint`:
  - This repo has no enforced formatter/linter task in CI; rely on the test gates above and keep diffs PEP8-friendly.
- `bun tsc --noEmit`:
  - Use `python3 -m compileall src tests` as the repo-level static sanity gate.
- `bun run test`:
  - Use targeted `pytest` runs for touched areas.
- `bun run test:e2e`:
  - Use `tests/test_web_app.py` plus Playwright UI verification for end-to-end UX-impacting changes.

## Key Behavioral Contracts

### Conversation and Intent Flow
- Greeting gate is strict: main scenario message is blocked until expected greeting is detected.
- Language pre-step (`language_selection_message`) can run before main utterance.
- Intent-mode behavior remains strict against `expected_intent`.
- Knowledge-mode (`knowledge` / `pets` / `baggage`) switches to goal evaluation.

### Knowledge Timeout Contract
- Base defaults:
  - `response_timeout = 90`
  - `step_skip_timeout_seconds = 90`
  - `knowledge_mode_timeout_seconds = 120`
- For knowledge-mode attempts only:
  - effective response timeout = `max(response_timeout, knowledge_mode_timeout_seconds)`
  - effective step timeout = `max(step_skip_timeout_seconds, knowledge_mode_timeout_seconds)`
- Non-knowledge attempts must keep base timeout behavior.

### Parallelism and Pacing
- Parallel attempts are enabled by default.
- Current worker cap is `1..3` with default `2`.
- Global start pacing default is `5.0s`.
- Adaptive pacing is enabled by default and adjusts based on greeting/pre-greeting pressure signals.

## Home UI Contracts
- Top-level tabs: `Harness`, `Analytics`, `Transcript`, `Defaults`.
- Language selectors are split:
  - Run Language
  - Transcript Language
  - Evaluation & Results Language
- Harness help uses inline `?` popovers.
- Keep `/run`, `/seed`, `/seed/import`, `/seed/url`, `/run/analytics_journey` contracts backward-compatible.

## Data and Privacy Rules
- `local_suites/` is local-only and ignored by git.
- Only `sample_test_suite.yaml` is public-safe by default.
- Transcript URL and analytics artifacts are local-only under `.gc_tester_history/`.
- Do not add private/local suites or raw customer artifacts to tracked files.

## Testing Expectations
Run targeted tests for touched areas, then full suite:

```bash
.venv/bin/pytest tests/test_app_config.py tests/test_conversation_runner.py tests/test_orchestrator.py tests/test_web_app.py -q
.venv/bin/pytest -q
```

CI runs `pytest` on Python `3.9` and `3.11` (`.github/workflows/ci.yml`), so keep local verification compatible with that entrypoint.

## Change Checklist
Before finalizing a change:
1. Preserve existing mode semantics unless explicitly changing them.
2. Keep config/env/UI/CLI wiring aligned for new runtime knobs.
3. Update README when defaults or operator-facing behavior changes.
4. Ensure exports remain backward-compatible unless explicitly expanded.
5. Validate that no local/private artifacts are staged (`local_suites/`, `.gc_tester_history/`, `.hypothesis/`).

## Git Workflow
- Primary push target: `fork` remote.
- Typical flow:

```bash
git status --short
git add <files>
git commit -m "<message>"
git push fork main
```
