# Regression Test Harness

A regression testing tool for Genesys Cloud Agentic Virtual Agents. It supports:
- **Standard** intent/goal validation over live Web Messaging conversations
- **Journey** validation (contained + fulfilled path) with configurable category strategy
- **Analytics Journey Regression (evaluate-now)** from Bot Flow analytics conversations

The harness captures deterministic tool and journey evidence, evaluates outcomes with an LLM-as-judge workflow, and publishes results to a single dashboard/export pipeline.

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

Open http://localhost:5000 in your browser.

### Operator Quick Flow

1. In **Language**, set `Run Language`, `Transcript Language`, and `Evaluation & Results Language`.
2. In **Harness Configuration**, upload a suite and run `standard` or `journey` regression.
3. In **Analytics Journey Regression** (optional), run evaluate-now analytics checks for a Bot Flow.
4. In **Transcript Suite** (optional), seed suites from file/IDs/URL or update automation settings.
5. Review `/results`, then export CSV/JSON/JUnit/transcripts/bundle/PDF/PNG as needed.

The Home page uses four top-level panes:
- **Language**
- **Harness Configuration**
- **Analytics Journey Regression**
- **Transcript Suite** (with sub-tabs: **Upload File**, **Conversation IDs**, **Transcript URL**, **Automation**)

In **Language**, set:
- **Run Language** — conversation simulation language for test runs (`en`, `fr`, `fr-CA`, `es`)
- **Transcript Language** — transcript seeding/import and automation language (`en`, `fr`, `fr-CA`, `es`)
- **Evaluation & Results Language** — Judge explanations and Results UI language (`inherit`, `en`, `fr`, `fr-CA`, `es`)

In **Harness Configuration**, fill in:
- **Deployment ID** — your Genesys Cloud Web Messaging deployment ID
- **Region** — e.g., `mypurecloud.com`
- **Ollama Model** — e.g., `llama3.2`
- **Test Suite File** — upload a YAML or JSON test suite
- **Max Conversation Turns** *(optional)* — cap user turns per attempt (default: `10`)
- **Harness Mode** *(optional run override)* — `standard` or `journey`
- **Journey Category Strategy** *(optional run override)* — `rules_first` or `llm_first`
- **Enable Judging Mechanics** *(advanced, optional)* — activate score-threshold gating for judge-driven paths
- **Judging Objective / Strictness / Tolerance / Weights / Explanation Mode** *(advanced, optional)* — tune Phase 11 scoring behavior per run
- **Enable Journey Dashboard** *(advanced, optional)* — activate Phase 12 taxonomy rollups + dynamic journey views in Results
- **Genesys OAuth Client ID / Secret** *(optional)* — needed for Conversations API intent fallback and conversation-ID transcript imports
- **Intent Participant Attribute Name** *(optional)* — participant data field used for intent fallback (default: `detected_intent`)
- **Capture Debug WebSocket Frames + Frame Limit** *(optional)* — helps diagnose missing `conversation_id` / `participant_id`
- Use inline **`?`** help popovers beside field labels for field impact/details.

In **Analytics Journey Regression**, configure and run evaluate-now analytics checks:
- **Bot Flow ID**
- **Divisions** *(optional, comma-separated)*
- **Interval / Date Range** (required)
  - rich range picker with date+time selection
  - quick presets: `Today`, `Yesterday`, `Last 7 Days`, `Last 24 Hours`, `Clear`
  - local-time picks are converted to canonical UTC ISO interval strings (`start_iso/end_iso`) for Genesys APIs
  - manual interval typing is still supported as fallback
- **Page Size** and **Max Conversations**
- **Language Filter** *(optional)*
- **Advanced Raw Filter JSON** *(optional)*
- Submit action posts to `POST /run/analytics_journey`.

UI theme behavior:
- Dark mode defaults to your system preference.
- Use the top-right theme toggle on Home, Results, and Transcript Suite Preview to override.

The app now derives the Web Messaging Origin header automatically from Region (for example, `mypurecloud.com` -> `https://apps.mypurecloud.com`).

Transcript Suite capabilities:
- **Upload File**: seed from transcript files (`.json`, `.yaml`, `.yml`, `.txt`, `.log`, `.csv`, `.tsv`).
- **Conversation IDs**: import transcripts from Genesys Cloud by ID via file, paste, or auto-query mode.
- **Transcript URL**: fetch transcript JSON from an allowed HTTPS URL and seed a draft suite.
  - Supports `seed_strategy=utterance` and `seed_strategy=journey`.
  - Journey URL seeding supports `journey_category_strategy` (`rules_first` / `llm_first`).
- **Automation**: save daily transcript import scheduler settings without launching an import run.
- Partial success is supported for imports. Preview includes fetched/failed/skipped counts and optional failure manifest download.
- Transcript URL and fetched transcript artifacts are local-only and stored under the local history area. URL query tokens are redacted in UI summaries.

## Running via CLI

```bash
python3 -m src.cli test_suite.yaml \
  --region mypurecloud.com \
  --deployment-id YOUR_DEPLOYMENT_ID \
  --ollama-model llama3.2 \
  --language fr-CA
```

## Test Suite Format

```yaml
name: My Regression Suite
language: fr-CA

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
| `language` (suite-level) | No | Default suite language (`en`, `fr`, `fr-CA`, `es`) |
| `harness_mode` (suite-level) | No | Default suite execution mode: `standard` (intent/goal) or `journey` (contained + fulfilled path validation) |
| `primary_categories` (suite-level) | No | Optional journey category definitions (`name`, `keywords`, `rubric`) used by journey category resolution/evaluation |
| `name` | Yes | Scenario name shown in results |
| `persona` | Yes | Who the simulated user is, including any auth details they'd know |
| `goal` | Yes | What the user is trying to accomplish and how to know it's done |
| `first_message` | No | Exact first message to send (if omitted, LLM generates it) |
| `language_selection_message` | No | Optional pre-step user message sent before the main scenario message (for example `english`, `francais`, `espanol`) |
| `expected_intent` | No | Enables intent-assertion mode. The runner compares detected intent from agent text (e.g. `INTENT=flight_cancel` or `{"intent":"flight_cancel"}`) and falls back to Conversations API participant attributes when configured |
| `intent_follow_up_user_message` | No | Optional deterministic second-turn user reply for intent flows that require confirmation/branching |
| `journey_category` | No | Expected primary category for journey mode (for example `flight_cancel`, `flight_change`, `baggage`, `pets`, `vacation`, `speak_to_agent`, `guidelines`) |
| `journey_validation` | No | Journey validation controls (`require_containment`, `require_fulfillment`, optional `path_rubric`, optional `category_rubric_override`) |
| `tool_validation` | No | Scenario-level deterministic tool validation rules (`loose_rule` required, `strict_rule` optional) |
| `attempts` | No | Number of times to run this scenario (default: 5) |

### Harness Modes

- `standard`: existing intent/knowledge/goal evaluation behavior.
- `journey`: full journey evaluation. `expected_intent` is ignored, and pass/fail is based on containment + fulfillment/path checks (and category match when `journey_category` is configured).

Journey pass gate defaults:
- `require_containment=true`
- `require_fulfillment=true`

Containment resolution:
- metadata-first from conversation payload/participants
- LLM fallback when metadata is unavailable
- unresolved containment fails with explicit `containment_unknown`

Practical journey suite example:

```yaml
name: WestJet Journey Suite
language: fr-CA
harness_mode: journey
primary_categories:
  - name: flight_cancel
    keywords: [cancel, refund, cancel my booking]
    rubric: The journey should follow cancellation handling and provide valid next steps.

scenarios:
  - name: Cancellation Journey
    persona: >
      Un voyageur qui veut annuler son vol.
    goal: >
      Validate the full journey for cancellation. The customer must remain contained and
      be routed through the correct cancellation path.
    first_message: Je veux annuler ma reservation
    journey_category: flight_cancel
    journey_validation:
      require_containment: true
      require_fulfillment: true
      path_rubric: The journey should confirm cancellation options and next steps.
    attempts: 1
```

### Tool Validation

Use `tool_validation` when you want the attempt to verify tool-calling behavior in addition to intent/goal or journey outcomes.

Supported structure:
- `loose_rule`: required when `tool_validation` is present. This is the gating rule.
- `strict_rule`: optional diagnostics-only rule, commonly used for order checks.

Rule expression operators:
- `all`: every nested rule must pass.
- `any`: at least one nested rule must pass.
- `not`: nested rule must fail.
- `in_order`: nested rules must match in sequence.

Leaf rule fields:
- `tool`: normalized tool name to match.
- `min_count`: minimum number of matches for that tool leaf (default: `1`).
- `status_in`: optional status filter list (for example `["success", "completed"]`).

Validation semantics:
- If `tool_validation` is configured, final attempt success requires `loose_rule` to pass.
- `strict_rule` is recorded as a diagnostic signal and does not gate success/failure.
- If no valid tool signal is captured for an attempt with `tool_validation`, the attempt fails with a missing-tool-signal outcome.

Tool evidence capture sources:
- Primary: participant attributes (`GC_TESTER_TOOL_ATTRIBUTE_KEYS`, default: `rth_tool_events,tool_events`).
- Fallback: explicit response markers (`GC_TESTER_TOOL_MARKER_PREFIXES`, default: `tool_event:`), for example:

```text
tool_event: {"tool":"flight_lookup","status":"success"}
```

Practical example:

```yaml
name: Tool Validation Suite

scenarios:
  - name: Priority Change Tool Path
    persona: Traveler changing a flight within 72 hours
    goal: Complete a priority flight-change flow
    first_message: I need to change my flight
    expected_intent: flight_change_priority_within_72_hours
    attempts: 1
    tool_validation:
      loose_rule:
        all:
          - tool: flight_lookup
            status_in: [success, completed]
          - any:
              - tool: flight_change_priority
              - tool: flight_change_standard
      strict_rule:
        in_order:
          - tool: flight_lookup
          - any:
              - tool: flight_change_priority
              - tool: flight_change_standard
```

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

Special handling for `speak_to_agent`:
- Default follow-up confirmation is localized by selected language (English default: `Yes, connect me to a live agent`) when no explicit override is provided.
- You can override that second turn with `intent_follow_up_user_message` for scenario-specific branching.
- Final pass/fail remains strict against `expected_intent` after the follow-up turn.

Validation combinations:
- `standard` mode + `expected_intent` + `tool_validation`: intent/goal checks plus loose tool-validation pass must both pass.
- `journey` mode + `tool_validation`: journey pass gate plus loose tool-validation pass must both pass.

Greeting gate behavior:
- Before the main scenario utterance, the runner must detect the configured greeting.
- If greeting is not detected in time, the attempt is marked as timed out and the main utterance is not sent.
- This strict gate still applies when `language_selection_message` pre-steps are configured.

If you want text-mode intent validation, configure your bot to return a test-mode message like:

```text
INTENT=flight_cancel
```

The results UI will show a `Detected Intent` badge on each attempt when an intent marker is found.

For knowledge-style flows that emit final intent only after the user closes the interaction, the runner automatically sends:

```text
no, thank you that is all
```

The closure message is localized by selected language profile unless explicitly overridden in runtime config.

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
| `GC_TESTER_CONFIG_FILE` | n/a | Path to configuration file (default: `config.yaml`) |
| `GC_REGION` | `gc_region` | Genesys Cloud region |
| `GC_DEPLOYMENT_ID` | `gc_deployment_id` | Web Messaging deployment ID |
| `GC_CLIENT_ID` | `gc_client_id` | OAuth client id used for Conversations API intent fallback, transcript import, and Analytics Journey API runs |
| `GC_CLIENT_SECRET` | `gc_client_secret` | OAuth client secret used for Conversations API intent fallback, transcript import, and Analytics Journey API runs |
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
| `GC_TESTER_TRANSCRIPT_URL_ALLOWLIST` | `transcript_url_allowlist` | Comma-separated allowlist for transcript URL mode host matching (default: `pure.cloud,mypurecloud.com`) |
| `GC_TESTER_TRANSCRIPT_URL_TIMEOUT_SECONDS` | `transcript_url_timeout_seconds` | Timeout in seconds for transcript URL fetches (default: `30`) |
| `GC_TESTER_TRANSCRIPT_URL_MAX_BYTES` | `transcript_url_max_bytes` | Max response size for transcript URL fetches in bytes (default: `5000000`) |
| `GC_TESTER_TOOL_ATTRIBUTE_KEYS` | `tool_attribute_keys` | Comma-separated participant attribute keys used for primary tool event capture (default: `rth_tool_events,tool_events`) |
| `GC_TESTER_TOOL_MARKER_PREFIXES` | `tool_marker_prefixes` | Comma-separated response marker prefixes used for fallback tool event capture (default: `tool_event:`) |
| `GC_TESTER_LANGUAGE` | `language` | Default execution language (`en`, `fr`, `fr-CA`, `es`; default: `en`) |
| `GC_TESTER_EVALUATION_RESULTS_LANGUAGE` | `evaluation_results_language` | Default Judge explanation + Results UI language (`inherit`, `en`, `fr`, `fr-CA`, `es`; default: `inherit`) |
| `GC_TESTER_HARNESS_MODE` | `harness_mode` | Default harness mode (`standard` or `journey`; default: `standard`) |
| `GC_TESTER_JOURNEY_CATEGORY_STRATEGY` | `journey_category_strategy` | Default journey category strategy (`rules_first` or `llm_first`; default: `rules_first`) |
| `GC_TESTER_JOURNEY_PRIMARY_CATEGORIES_JSON` | `journey_primary_categories_json` | Optional JSON array override for journey primary categories |
| `GC_TESTER_JOURNEY_PRIMARY_CATEGORIES_FILE` | `journey_primary_categories_file` | Optional path to JSON file containing journey primary category overrides |
| `GC_TESTER_JUDGING_MECHANICS_ENABLED` | `judging_mechanics_enabled` | Enable Phase 11 judging score mechanics (default: `false`) |
| `GC_TESTER_JUDGING_OBJECTIVE_PROFILE` | `judging_objective_profile` | Objective profile (`intent_focused`, `journey_focused`, `blended`; default: `blended`) |
| `GC_TESTER_JUDGING_STRICTNESS` | `judging_strictness` | Threshold band (`strict`, `balanced`, `lenient`; default: `balanced`) |
| `GC_TESTER_JUDGING_TOLERANCE` | `judging_tolerance` | Threshold relaxation value (default: `0.0`, capped in runtime logic) |
| `GC_TESTER_JUDGING_CONTAINMENT_WEIGHT` | `judging_containment_weight` | Journey scoring weight for containment criterion (default: `0.35`) |
| `GC_TESTER_JUDGING_FULFILLMENT_WEIGHT` | `judging_fulfillment_weight` | Journey scoring weight for fulfillment criterion (default: `0.45`) |
| `GC_TESTER_JUDGING_PATH_WEIGHT` | `judging_path_weight` | Journey scoring weight for path correctness criterion (default: `0.20`) |
| `GC_TESTER_JUDGING_EXPLANATION_MODE` | `judging_explanation_mode` | Judging summary verbosity (`concise`, `standard`, `verbose`; default: `standard`) |
| `GC_TESTER_JOURNEY_DASHBOARD_ENABLED` | `journey_dashboard_enabled` | Enable Phase 12 journey taxonomy dashboard in Results (default: `false`) |
| `GC_TESTER_JOURNEY_TAXONOMY_OVERRIDES_JSON` | `journey_taxonomy_overrides_json` | Optional JSON object mapping keywords -> taxonomy label |
| `GC_TESTER_JOURNEY_TAXONOMY_OVERRIDES_FILE` | `journey_taxonomy_overrides_file` | Optional path to JSON keyword->label mapping file for taxonomy overrides |
| `GC_TESTER_ANALYTICS_JOURNEY_ENABLED` | `analytics_journey_enabled` | Enable Analytics Journey Regression evaluate-now mode (default: `false`) |
| `GC_TESTER_ANALYTICS_JOURNEY_DEFAULT_PAGE_SIZE` | `analytics_journey_default_page_size` | Default analytics page size for evaluate-now runs (default: `50`) |
| `GC_TESTER_ANALYTICS_JOURNEY_DEFAULT_MAX_CONVERSATIONS` | `analytics_journey_default_max_conversations` | Default max conversations evaluated per analytics run (default: `150`) |
| `GC_TESTER_ANALYTICS_JOURNEY_POLICY_MAP_JSON` | `analytics_journey_policy_map_json` | Optional JSON policy map for auth/transfer expectations in analytics gating |
| `GC_TESTER_ANALYTICS_JOURNEY_POLICY_MAP_FILE` | `analytics_journey_policy_map_file` | Optional JSON file path for analytics policy-map overrides |
| `GC_TESTER_ANALYTICS_JOURNEY_DEFAULT_LANGUAGE_FILTER` | `analytics_journey_default_language_filter` | Optional default analytics language filter for evaluate-now runs |
| `GC_TESTER_ANALYTICS_JOURNEY_ARTIFACT_DIR` | `analytics_journey_artifact_dir` | Local-only directory for analytics payload/enrichment artifacts (default: `.gc_tester_history/analytics_journey`) |
| `GC_TESTER_ATTEMPT_PARALLEL_ENABLED` | `attempt_parallel_enabled` | Enable global parallel attempt execution worker pool for standard/journey runs (default: `true`) |
| `GC_TESTER_MAX_PARALLEL_ATTEMPT_WORKERS` | `max_parallel_attempt_workers` | Max parallel attempt workers, clamped to `1..3` (default: `2`) |
| `GC_TESTER_JUDGE_WARMUP_ENABLED` | `judge_warmup_enabled` | Run an automatic Judge LLM warm-up call before scenario execution (default: true) |
| `GC_TESTER_DEFAULT_ATTEMPTS` | `default_attempts` | Default attempts per scenario (default: 5) |
| `GC_TESTER_MAX_TURNS` | `max_turns` | Max conversation turns (default: 10) |
| `GC_TESTER_MIN_ATTEMPT_INTERVAL_SECONDS` | `min_attempt_interval_seconds` | Minimum seconds between attempt starts (float supported; default: `7.5`, enforced globally across the worker pool) |
| `GC_TESTER_STEP_SKIP_TIMEOUT_SECONDS` | `step_skip_timeout_seconds` | Max allowed duration for a single attempt step before the attempt is skipped (default: 90) |
| `GC_TESTER_RESPONSE_TIMEOUT` | `response_timeout` | Timeout in seconds (default: 90) |
| `GC_TESTER_SUCCESS_THRESHOLD` | `success_threshold` | Regression threshold (default: 0.8) |
| `GC_TESTER_EXPECTED_GREETING` | `expected_greeting` | Greeting text required before first user message |

Precedence: Web UI > Environment variables > config.yaml > defaults

Language precedence per run: runtime override (UI/CLI/form) > `suite.language` > `GC_TESTER_LANGUAGE` / `config.language` > `en`.

Harness mode precedence per run: runtime override (Web UI run form) > `suite.harness_mode` > `GC_TESTER_HARNESS_MODE` / `config.harness_mode` > `standard`.

Evaluation/Results language precedence per run: runtime override (UI/form) > `GC_TESTER_EVALUATION_RESULTS_LANGUAGE` / `config.evaluation_results_language` > `inherit` (then resolves to run language).

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
Status: Delivered

- Deterministic tool capture from participant attributes (primary) plus explicit response markers (fallback).
- Per-attempt tool execution timeline with source/status/timestamp metadata.
- Tool evidence exported across JSON, CSV, JUnit, transcripts ZIP, bundle ZIP, and dashboard PDF.

### Phase 3: Tool Execution Validation
Status: Delivered

- Advanced `tool_validation` rules with boolean expression blocks (`all`, `any`, `not`, `in_order`).
- Dual outcomes per attempt: loose pass (gating) and strict-order pass (diagnostic).
- Missing-signal hard failure when tool validation is configured but no valid events are captured.

### Phase 4: Transcript-to-Suite Seeding
Status: Delivered

- Transcript upload + seeded scenario preview + editable YAML export.
- Conversation-ID transcript import (file/paste/auto-query), partial-failure diagnostics, and failure manifest download.
- Daily built-in transcript import scheduler with custom filter-driven auto-query and local artifact persistence.
- Transcript URL import mode with allowlisted HTTPS fetch, safe URL redaction, and local-only artifact storage.
- Journey URL seeding strategy (`one scenario per conversation`) with configurable category strategy.

### Phase 4.1: Transcript Seeding Quality + Coverage
Status: Delivered

- Expanded structured extraction coverage (JSON/YAML/CSV/TSV variants and nested payloads).
- Improved deterministic speaker classification, dedupe, and noise filtering.
- Extraction summary banner and warning diagnostics in transcript preview.

### Phase 5: Local Time Everywhere
Status: Delivered

- Local/UTC toggle for timestamps in results and live step logs.
- Timezone-aware labels across key result surfaces.

### Phase 6: Metrics Dashboard + Visual Reporting
Status: Delivered

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

### Phase L-1: Multilingual Execution
Status: Delivered

- End-to-end language support for `en`, `fr`, `fr-CA`, and `es`.
- Language-aware Judge instructions, transcript seeding/import parsing, and runtime follow-up defaults.
- Suite portability via `suite.language` and runtime language overrides.

### Phase Journey-1: Journey Regression (No Intent Exposure)
Status: Delivered

- New journey harness mode for full customer-journey evaluation without intent text exposure.
- Category strategy support (`rules_first`, `llm_first`) and configurable primary category definitions.
- Journey pass gating on containment + fulfillment/path correctness (plus category match when expected).

### Phase UX-2: Home Workflow Refresh
Status: Delivered

- Toolbar-based Home navigation (`Language`, `Harness Configuration`, `Transcript Suite`) with persistence.
- Transcript Suite sub-tabs (`Upload File`, `Conversation IDs`, `Transcript URL`, `Automation`).
- Quick-start cards, progressive disclosure for advanced run settings, and stronger inline validation UX.

### Phase UX-L2.4: Help Popovers + Intent-Grouped Results
Status: Delivered

- Harness help upgraded to inline `?` popovers for cleaner field-level guidance.
- Results grouped as **Expected Intent -> Scenario -> Attempt** with fallback bucket **Behavior / Journey**.
- Grouped structure applies to both completed runs and live SSE rendering.

### Phase UX-L2.5: Collapsible Attempts + Bulk Toggle
Status: Delivered

- Added top-level **All Attempts** collapsible panel to reduce visual noise by default.
- Added **Expand All / Collapse All** controls for Intent -> Scenario -> Attempt levels.
- Bulk toggle behavior applies to both completed and live run trees (diagnostic sub-panels excluded).

### Phase AJR-UX-1: Analytics Date-Range Picker
Status: Delivered

- Analytics Journey Regression now includes a rich local date-range/time picker for `Interval / Date Range`.
- Added presets: `Today`, `Yesterday`, `Last 7 Days`, `Last 24 Hours`, and `Clear`.
- Picker selections are converted from local time to canonical UTC ISO interval strings before submission.

### Phase 11: Judging Mechanics Parameters
Status: Delivered

- Advanced run-level controls for objective profile, strictness/tolerance, journey weights, and explanation mode.
- Deterministic score metadata on attempts (`score`, `threshold`, criteria breakdown) for traceability.
- Score-threshold gating for judge-driven outcomes when enabled, while preserving existing hard gates.
- Disabled by default for safe rollout; can be enabled per run from Harness Configuration (Advanced).

### Phase 12: Journey Evaluation Dashboard (Dynamic Views)
Status: Delivered

- Added an in-Results **Journey Evaluation Dashboard** with dynamic toolbar views:
  - `Overview`
  - `Live Agent Transfer`
  - `Containment`
  - `Hangup/Disconnect`
  - `Flow/Noise Issues`
- Deterministic taxonomy rollups with configurable keyword->label overrides.
- Rollups and view deltas are carried into exports (including dashboard PDF/PNG capture context).
- Disabled by default for safe rollout; can be enabled per run from Harness Configuration (Advanced).

### Phase 13: Analytics Journey Regression (Evaluate-Now)
Status: Delivered

- Added a dedicated **Analytics Journey Regression** Home tab and run flow.
- New evaluate-now submission route: `POST /run/analytics_journey`.
- Runs fetch analytics reporting turns, evaluate one conversation journey at a time, and publish into the standard Results + Export pipeline.
- Conversation-level gate outcomes (classification/path, auth, transfer, journey quality) are captured in attempt diagnostics and exports.
- Raw analytics and enrichment artifacts remain local-only in the analytics artifact directory.

### Phase 14: Transcript Seed via Analytics API
Status: Planned

- Add a Transcript Suite seed source using Genesys Analytics Bot Flow Reporting Turns API.
- Source endpoint: [Genesys API Explorer: Bot Flow Reporting Turns](https://developer.genesys.cloud/devapps/api-explorer#get-api-v2-analytics-botflows--botFlowId--divisions-reportingturns)
- MVP scope:
  - Input controls: bot flow id, divisions, interval/date range, optional language filter.
  - Ingestion: fetch reporting turns, normalize to seedable records, generate draft suite output.
  - Diagnostics: pulled/seeded/skipped counts with reasoned skip manifest.
  - Compatibility: coexist with existing `Upload File`, `Conversation IDs`, and `Transcript URL` seed paths.

## What's Next

- Phase 14: add transcript seeding from Analytics Bot Flow Reporting Turns with operator diagnostics.
- Add domain-specific taxonomy preset packs for Journey dashboard operators.
- Add side-by-side baseline overlays for Journey dashboard drilldowns.

## Results

The results page organizes attempts as **Expected Intent -> Scenario -> Attempt**, with fallback grouping under **Behavior / Journey** when no expected intent is set.

### Live Run Diagnostics

- Live progress bar with `% complete`, completed attempts, and ETA.
- Live attempt-step panel with recent step logs for in-progress debugging.
- Stop-run flow with clear active, stop-requested, stopping, and complete states.
- `All Attempts` is collapsible, with bulk **Expand All / Collapse All** controls for Intent -> Scenario -> Attempt.
- Live SSE updates follow the same grouping model used for completed results.
- Skipped-attempt tracking when a single attempt step exceeds the step timeout threshold.
- Adaptive duration display (`s`, `m s`, `h m s`) across dashboard and attempt surfaces.
- Time display toggle (`Local` / `UTC`) for summary, timeline, attempt timings, and live step logs.
- Paged attempt rendering with `Load more attempts` for large runs.
- Intent buckets with nested scenario/attempt dropdowns for faster scanability.
- Re-run controls for:
  - last suite,
  - failed/timeout/skipped subset,
  - selected scenarios.

### Dashboard Analytics

- KPI cards for attempts, successes, failures, timeouts, skipped, and success rate.
- Duration analytics for average, median, and p95 attempt duration.
- Outcome mix distribution and scenario health ranking.
- Same-suite trend and baseline compare panel with selectable baseline run.
- Flakiness/stability insights, including unstable scenario ranking.
- Phase 11 judging mechanics diagnostics:
  - scored attempts,
  - threshold pass/fail rates,
  - average judging score,
  - attempt-level criteria breakdown payloads.
- Phase 12 journey dashboard with dynamic views:
  - `Overview`,
  - `Live Agent Transfer`,
  - `Containment`,
  - `Hangup/Disconnect`,
  - `Flow/Noise Issues`.
- Journey dashboard view switches are in-page (no scroll-reset reloads when changing views).
- Journey analytics:
  - validated journey attempts,
  - contained/fulfilled/path-correct pass tracking,
  - category-match tracking when journey categories are configured.
- Tool-effectiveness metrics:
  - validated attempts,
  - loose and strict validation rates,
  - missing-signal counts,
  - order-mismatch counts.
- Attempt-level tool evidence with timeline, source, status, loose/strict badges, and mismatch diagnostics.
- Attempt-level journey evidence with `Contained`, `Fulfilled`, `Path Correct`, `Category Match`, and containment source badges plus diagnostic payloads.
- Analytics Journey runs include gate-level diagnostics (category/auth/transfer/quality) and explicit skip reasons when evidence is inconclusive.
- Collapsible `Metrics Legend & Definitions` and dark-mode support.

### Export Formats

- CSV summary.
- JSON full report.
- JUnit XML (CI-friendly).
- ZIP of per-attempt conversation transcripts.
- Bundle ZIP containing `report.json`, `report.csv`, `report.junit.xml`, and transcripts.
- Dashboard PDF with a 2-page infographic (executive metrics + scenario deep dive) via `/results/export?format=dashboard_pdf`.
- Dashboard PNG screenshot export (client-side, captures the rendered dashboard view including current theme/baseline selection).
- Exports include tool, journey, and analytics gate evidence when present (JSON full payload, CSV scenario columns, JUnit `system-out`, transcript ZIP, and bundle ZIP).

If a run is stopped early, exports still work using partial completed-attempt data collected so far.
Step logs are included in `report.json`, JUnit `system-out`, and transcript ZIP outputs.
Timeout and pre-greeting fast-fail diagnostics are exported in `report.json`, JUnit `system-out`, transcript ZIP, and bundle ZIP.

When debugging missing `conversationId` values, enable debug frame capture and inspect the `Debug Frames` section on each attempt card. The fallback only uses explicit pulled conversation-id fields (not generic message `id` values).

The CLI exits with code `1` if any scenario falls below the success threshold, making it CI/CD friendly.
