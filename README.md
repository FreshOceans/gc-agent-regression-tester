# Regression Test Harness

A regression testing tool for Genesys Cloud Agentic Virtual Agents. Uses an LLM-as-judge methodology — an Ollama-hosted LLM plays a simulated user with a persona and goal, drives multi-turn conversations with your deployed agent via the Web Messaging API, and evaluates whether the goal was achieved across multiple attempts.

## Prerequisites

- Python 3.9+
- [Ollama](https://ollama.ai) running locally with a model pulled (e.g., `ollama pull llama3.2`)
- A Genesys Cloud Web Messaging deployment (deployment ID + region)

## Setup

```bash
cd gc-agent-regression-tester
pip install -r requirements.txt
```

## Running the Web UI

```bash
python3 -m src.web_app
```

Open http://localhost:5000 in your browser. Fill in:
- **Deployment ID** — your Genesys Cloud Web Messaging deployment ID
- **Region** — e.g., `mypurecloud.com`
- **Ollama Model** — e.g., `llama3.2`
- **Test Suite File** — upload a YAML or JSON test suite

UI theme behavior:
- Dark mode defaults to your system preference.
- Use the top-right theme toggle on Home, Results, and Transcript Suite Preview to override.

The app now derives the Web Messaging Origin header automatically from Region (for example, `mypurecloud.com` -> `https://apps.mypurecloud.com`).

Phase 4 MVP is now available in the home UI:
- Use **Transcript Suite** to upload transcript files (`.json`, `.yaml`, `.yml`, `.txt`, `.log`, `.csv`, `.tsv`).
- Review generated scenarios and editable YAML in preview.
- Download the seeded YAML, then upload it in the main Run form.
- Use **Import by Conversation IDs** to fetch transcripts directly from Genesys Cloud by IDs file, pasted IDs, or auto-query mode.
- Optional built-in daily import scheduling is configurable from the Transcript Suite panel.

## Running via CLI

```bash
python3 -m src.cli test_suite.yaml \
  --region mypurecloud.com \
  --deployment-id YOUR_DEPLOYMENT_ID \
  --ollama-model llama3.2
```

## Test Suite Format

```yaml
name: My Regression Suite

scenarios:
  - name: Account Balance Inquiry
    persona: >
      Customer named Margaret. Her 8-digit login code is 12345678.
    goal: >
      Check account balance. Provide login code when asked.
      Goal is achieved when the agent provides a balance amount.
    first_message: "What is my account balance?"
    attempts: 3

  - name: Intent Classification (Test Mode)
    persona: >
      Traveler asking to cancel a booking.
    goal: >
      Confirm the classifier returns the expected intent.
    first_message: "I want to cancel my booking"
    expected_intent: "flight_cancel"
    attempts: 1
```

### Fields

| Field | Required | Description |
|-------|----------|-------------|
| `name` | Yes | Scenario name shown in results |
| `persona` | Yes | Who the simulated user is, including any auth details they'd know |
| `goal` | Yes | What the user is trying to accomplish and how to know it's done |
| `first_message` | No | Exact first message to send (if omitted, LLM generates it) |
| `expected_intent` | No | Enables intent-assertion mode. The runner compares detected intent from agent text (e.g. `INTENT=flight_cancel` or `{"intent":"flight_cancel"}`) and falls back to Conversations API participant attributes when configured |
| `attempts` | No | Number of times to run this scenario (default: 5) |

When using `expected_intent`, the tester tries this order:
1. Parse intent from agent text in the chat transcript.
2. If not found, resolve IDs from explicit pulled fields (`conversationId`/`conversation_id`) or explicit transcript labels (for example `conversation_id: <uuid>`, `participant_id: <uuid>`, `"conversation_id":"<uuid>"`, `"participant_id":"<uuid>"`), then query Conversations API participant attributes (default: `detected_intent`).

Special handling for knowledge scenarios:
- If `expected_intent` is `knowledge`, `pets`, or `baggage` (or starts with `knowledge`), the runner switches to goal-evaluation mode for that attempt.
- In that mode, success is determined by whether the agent actually answers the user question (LLM judge), rather than strict intent-string matching.

Special handling for `flight_priority_change`:
- The tester simulates the follow-up answer to the 72-hour question (`yes` or `no`).
- If answer is `yes`, expected intent is updated to `flight_change_priority_within_72_hours`.
- If answer is `no`, expected intent is updated to `flight_change_later_than_72_hours`.

Special handling for vacation inquiry flows:
- Use `expected_intent: vacation_inquiry_flight_only` when the follow-up choice should be `flight only`.
- Use `expected_intent: vacation_flight_and_hotel` when the follow-up choice should be `flight and hotel`.
- Legacy `expected_intent: vacation_inquiry` is still supported, but the runner resolves it dynamically based on the simulated follow-up answer.

If you want text-mode intent validation, configure your bot to return a test-mode message like:

```text
INTENT=flight_cancel
```

The results UI will show a `Detected Intent` badge on each attempt when an intent marker is found.

For knowledge-style flows that emit final intent only after the user closes the interaction, the runner automatically sends:

```text
no, thank you that is all
```

before falling back to Conversations API lookup.

### Intent Fallback Troubleshooting

If AI Studio preview shows intent but Messenger tests do not, verify all of these:

1. The bot writes intent to participant data on the same participant queried by fallback (default attribute name: `detected_intent`).
2. The flow version used by your Messenger deployment is published and includes that write step.
3. If you surface IDs in transcript text, use explicit labels such as:

```text
"conversation_id":"e1f5c9e3-79eb-44d5-80fc-9c6568e51201"
"participant_id":"8bb02d61-9fdc-4000-b83a-70a0b11893e3"
```

## Configuration

You can set defaults via environment variables or a `config.yaml` file:

| Env Variable | Config Key | Description |
|-------------|------------|-------------|
| `GC_REGION` | `gc_region` | Genesys Cloud region |
| `GC_DEPLOYMENT_ID` | `gc_deployment_id` | Web Messaging deployment ID |
| `GC_CLIENT_ID` | `gc_client_id` | OAuth client id for Conversations API intent fallback |
| `GC_CLIENT_SECRET` | `gc_client_secret` | OAuth client secret for Conversations API intent fallback |
| `OLLAMA_BASE_URL` | `ollama_base_url` | Ollama URL (default: http://localhost:11434) |
| `OLLAMA_MODEL` | `ollama_model` | Ollama model name |
| `GC_TESTER_INTENT_ATTRIBUTE_NAME` | `intent_attribute_name` | Participant attribute name used for intent fallback (default: `detected_intent`) |
| `GC_TESTER_DEBUG_CAPTURE_FRAMES` | `debug_capture_frames` | Capture compact Web Messaging frame metadata for debugging missing IDs (default: false) |
| `GC_TESTER_DEBUG_CAPTURE_FRAME_LIMIT` | `debug_capture_frame_limit` | Max number of debug frame summaries stored per attempt (default: 8) |
| `GC_TESTER_HISTORY_DIR` | `history_dir` | Local directory used to persist run history for dashboard trends and comparisons (default: `.gc_tester_history`) |
| `GC_TESTER_HISTORY_MAX_RUNS` | `history_max_runs` | Maximum number of persisted runs kept in local history (default: `50`) |
| `GC_TESTER_HISTORY_FULL_JSON_RUNS` | `history_full_json_runs` | Number of newest runs kept as full `.json` report files before compaction (default: `20`) |
| `GC_TESTER_HISTORY_GZIP_RUNS` | `history_gzip_runs` | Number of subsequent runs kept as compressed `.json.gz` report files before summary-only archival (default: `20`) |
| `GC_TESTER_TRANSCRIPT_IMPORT_ENABLED` | `transcript_import_enabled` | Enable built-in daily transcript import scheduler (default: `false`) |
| `GC_TESTER_TRANSCRIPT_IMPORT_TIME` | `transcript_import_time` | Daily scheduler local time in `HH:MM` (default: `02:00`) |
| `GC_TESTER_TRANSCRIPT_IMPORT_TIMEZONE` | `transcript_import_timezone` | IANA timezone for daily scheduler (default: `UTC`) |
| `GC_TESTER_TRANSCRIPT_IMPORT_MAX_IDS` | `transcript_import_max_ids` | Max conversations per transcript import run (default: `50`) |
| `GC_TESTER_TRANSCRIPT_IMPORT_FILTER_JSON` | `transcript_import_filter_json` | Custom filter JSON for auto-query transcript import mode (default: `{}`) |
| `GC_TESTER_TRANSCRIPT_IMPORT_DIR` | `transcript_import_dir` | Local directory for transcript import manifests/raw payload artifacts (default: `.gc_tester_history/transcript_imports`) |
| `GC_TESTER_JUDGE_WARMUP_ENABLED` | `judge_warmup_enabled` | Run an automatic Judge LLM warm-up call before scenario execution (default: true) |
| `GC_TESTER_DEFAULT_ATTEMPTS` | `default_attempts` | Default attempts per scenario (default: 5) |
| `GC_TESTER_MAX_TURNS` | `max_turns` | Max conversation turns (default: 20) |
| `GC_TESTER_MIN_ATTEMPT_INTERVAL_SECONDS` | `min_attempt_interval_seconds` | Minimum seconds between attempt starts (default: 15) |
| `GC_TESTER_STEP_SKIP_TIMEOUT_SECONDS` | `step_skip_timeout_seconds` | Max allowed duration for a single attempt step before the attempt is skipped (default: 90) |
| `GC_TESTER_RESPONSE_TIMEOUT` | `response_timeout` | Timeout in seconds (default: 90) |
| `GC_TESTER_SUCCESS_THRESHOLD` | `success_threshold` | Regression threshold (default: 0.8) |
| `GC_TESTER_EXPECTED_GREETING` | `expected_greeting` | Greeting text required before first user message |

Precedence: Web UI > Environment variables > config.yaml > defaults

## Roadmap

Feature roadmap (priority order) with current status:

### Phase 1: Live Progress + Run Control
Status: Delivered

- Live progress bar with `% complete`, completed attempts, and ETA.
- Stop-run flow with clear stop-requested/run-complete states.
- Attempt step panel + recent step log for debugging active runs.

### Phase UX-1: Dark Mode + Product Rename
Status: Delivered

- Product branding refreshed to **Regression Test Harness** across UI/docs/export-facing labels.
- Full light/dark theme support with semantic tokens and system-default + local override behavior.
- Top-right theme toggle available on Home, Results, and Transcript Suite Preview pages.

### Phase 2: Tool Execution Tracking
Status: Planned

- Track tool/data-action execution per attempt and turn.
- Capture tool metadata (name, timestamp, status) for UI + exports.

### Phase 3: Tool Execution Validation
Status: Planned

- Add expected tool assertions to suite schema.
- Fail attempts when expected tool behavior is not observed.

### Phase 4: Transcript-to-Suite Seeding
Status: Delivered (Phase 4.2)

- Transcript upload + seeded scenario preview + editable YAML export.
- Conversation-ID transcript import (file/paste/auto-query), partial-failure diagnostics, and failure manifest download.
- Daily built-in transcript import scheduler with custom filter-driven auto-query and local artifact persistence.

### Phase 5: Local Time Everywhere
Status: Delivered

- Local/UTC toggle for timestamps in results and live step logs.
- Timezone-aware labels across key result surfaces.

### Phase 6: Metrics Dashboard + Visual Reporting
Status: Delivered (Phase 6.2 infographic PDF)

- Adaptive duration formatting across UI + PDF.
- Collapsible metrics legend and responsive export action row.
- Paged attempt rendering (`Load more attempts`) for large runs.
- Rich dashboard with outcome mix, scenario health, trend, compare, and infographic-style 2-page PDF export.

### Phase 7: Baseline Selector + Run-to-Run Diff
Status: Delivered

- Manual baseline selection on results page (default still previous same-suite run).
- Baseline-aware compare context propagated to dashboard PDF export.
- History endpoint for baseline picker data and safe same-suite fallback behavior.

### Phase 8: Rerun Subsets
Status: Delivered

- Re-run failed/timeout/skipped bucket in one click.
- Re-run selected scenarios from the latest suite snapshot.
- Guardrails for empty eligibility and invalid subset requests.

### Phase 9: Flakiness and Stability Metrics
Status: Delivered

- Per-scenario stability analytics across recent same-suite runs.
- Flip rate, failure occurrence, volatility, and instability scoring.
- Unstable scenario ranking in the results dashboard and dashboard PDF.

### Phase 10: History Compaction and Archival
Status: Delivered

- Tiered history storage: full JSON -> gzipped JSON -> summary-only archival.
- Local retention with metadata preserving compare capability for summary-only baselines.
- Configurable compaction windows under `GC_TESTER_HISTORY_MAX_RUNS`.

## What's Next

- Expand tool-execution observability and validation (Phases 2 and 3) with strong UX for actionable failures.
- Improve transcript seeding depth (Phase 4 follow-up) to reduce manual suite editing.
- Continue dashboard ergonomics and performance for very large enterprise regression suites.

## Results

The results page shows per-scenario success rates with all attempts expandable to review the full conversation, including per-message timestamps, per-turn timing, and total attempt duration. Export formats available from the results page:
- Live progress bar during active runs (`% complete`, completed attempts, ETA)
- Live attempt-step panel for in-progress debugging (including early-stop context)
- Skipped-attempt metric when a single attempt step exceeds the step timeout threshold
- Adaptive duration formatting (`s`, `m s`, `h m s`) across dashboard cards, attempt cards, compare deltas, and PDF
- Time display toggle on the results page (`Local` / `UTC`) for timestamps in report summary, message timeline, attempt timings, and live step log
- Dark mode with persisted user preference (`light`/`dark`) and system-default fallback
- Collapsible `Metrics Legend & Definitions` panel (instead of always-visible legend text)
- Re-run Last Test Suite button (reuses the latest uploaded suite and settings)
- Re-run subset controls (failed/timeout/skipped bucket and selected scenarios)
- Baseline selector for same-suite compare (with summary-only baseline fallback support)
- Paged attempt rendering with `Load more attempts` for large runs
- CSV summary
- JSON full report
- JUnit XML (CI-friendly)
- ZIP of per-attempt conversation transcripts
- Bundle ZIP containing `report.json`, `report.csv`, `report.junit.xml`, and transcripts
- Dashboard PDF containing a 2-page infographic (executive metrics + scenario deep dive) (`/results/export?format=dashboard_pdf`)

If a run is stopped early, exports still work using partial completed-attempt data collected so far.
Step logs are included in `report.json`, JUnit `system-out`, and transcript ZIP outputs.

When debugging missing `conversationId` values, enable debug frame capture and inspect the `Debug Frames` section on each attempt card. The fallback now only uses explicit pulled conversation-id fields (not generic message `id` values).

The CLI exits with code 1 if any scenario falls below the success threshold, making it CI/CD friendly.
