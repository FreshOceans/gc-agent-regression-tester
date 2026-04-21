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
- Top-level tabs: `Language`, `Harness Configuration`, `Analytics Journey Regression`, `Transcript Suite`.
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

## Change Checklist
Before finalizing a change:
1. Preserve existing mode semantics unless explicitly changing them.
2. Keep config/env/UI/CLI wiring aligned for new runtime knobs.
3. Update README when defaults or operator-facing behavior changes.
4. Ensure exports remain backward-compatible unless explicitly expanded.
5. Validate that no local/private files are staged.

## Git Workflow
- Primary push target: `fork` remote.
- Typical flow:

```bash
git status --short
git add <files>
git commit -m "<message>"
git push fork main
```

