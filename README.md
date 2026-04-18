# GC Agent Regression Tester

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

The app now derives the Web Messaging Origin header automatically from Region (for example, `mypurecloud.com` -> `https://apps.mypurecloud.com`).

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
| `GC_TESTER_DEFAULT_ATTEMPTS` | `default_attempts` | Default attempts per scenario (default: 5) |
| `GC_TESTER_MAX_TURNS` | `max_turns` | Max conversation turns (default: 20) |
| `GC_TESTER_MIN_ATTEMPT_INTERVAL_SECONDS` | `min_attempt_interval_seconds` | Minimum seconds between attempt starts (default: 30) |
| `GC_TESTER_RESPONSE_TIMEOUT` | `response_timeout` | Timeout in seconds (default: 30) |
| `GC_TESTER_SUCCESS_THRESHOLD` | `success_threshold` | Regression threshold (default: 0.8) |
| `GC_TESTER_EXPECTED_GREETING` | `expected_greeting` | Greeting text required before first user message |

Precedence: Web UI > Environment variables > config.yaml > defaults

## Roadmap

Planned feature enhancements (in priority order):

### Phase 1: Live Progress Bar

Goal: Provide clearer real-time visibility during long runs.

- Add a live progress bar with `% complete`.
- Show `attempts completed / total attempts`.
- Show estimated remaining time (ETA) based on completed attempts.

### Phase 2: Tool Execution Tracking

Goal: Improve observability of what the agent actually executed.

- Track tool/data-action execution per attempt and turn.
- Capture metadata such as tool name, timestamp, and execution status.
- Show tool execution timeline in the results UI and export payloads.

### Phase 3: Tool Execution Validation

Goal: Verify behavior correctness, not just outcome text.

- Extend test suite schema with expected tool assertions.
- Validate whether the correct tool(s) executed for each customer utterance.
- Mark attempts as failed with explicit mismatch reasons when expected tool execution is not observed.

### Phase 4: Transcript-to-Suite Seeding

Goal: Speed up test authoring from real customer conversations.

- Upload Genesys Cloud transcripts.
- Generate draft suite scenarios from transcript content.
- Pre-fill scenario fields such as `name`, `persona`, `first_message`, and candidate intent/tool expectations.
- Allow user review/edit before saving as YAML/JSON test suite.

### Phase 5: Local Time Everywhere (Delivered in Results UI)

Goal: Improve readability by showing times in the user's local timezone.

- Convert UTC timestamps shown in the UI to local time.
- Use local time in attempt details, step logs, and progress timeline where appropriate.
- Add timezone labels so exported and on-screen timestamps are unambiguous.
- Add a timezone display mode capability (Local / UTC) so users can switch to UTC when coordinating with platform logs and support teams.

## Results

The results page shows per-scenario success rates with all attempts expandable to review the full conversation, including per-message timestamps, per-turn timing, and total attempt duration. Export formats available from the results page:
- Live progress bar during active runs (`% complete`, completed attempts, ETA)
- Live attempt-step panel for in-progress debugging (including early-stop context)
- Time display toggle on the results page (`Local` / `UTC`) for timestamps in report summary, message timeline, attempt timings, and live step log
- Re-run Last Test Suite button (reuses the latest uploaded suite and settings)
- CSV summary
- JSON full report
- JUnit XML (CI-friendly)
- ZIP of per-attempt conversation transcripts
- Bundle ZIP containing `report.json`, `report.csv`, `report.junit.xml`, and transcripts

If a run is stopped early, exports still work using partial completed-attempt data collected so far.
Step logs are included in `report.json`, JUnit `system-out`, and transcript ZIP outputs.

When debugging missing `conversationId` values, enable debug frame capture and inspect the `Debug Frames` section on each attempt card. The fallback now only uses explicit pulled conversation-id fields (not generic message `id` values).

The CLI exits with code 1 if any scenario falls below the success threshold, making it CI/CD friendly.
