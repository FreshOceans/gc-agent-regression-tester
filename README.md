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
- **Allowed Origin** — the origin header for WebSocket auth (try `https://apps.mypurecloud.com`)
- **Test Suite File** — upload a YAML or JSON test suite

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
```

### Fields

| Field | Required | Description |
|-------|----------|-------------|
| `name` | Yes | Scenario name shown in results |
| `persona` | Yes | Who the simulated user is, including any auth details they'd know |
| `goal` | Yes | What the user is trying to accomplish and how to know it's done |
| `first_message` | No | Exact first message to send (if omitted, LLM generates it) |
| `attempts` | No | Number of times to run this scenario (default: 5) |

## Configuration

You can set defaults via environment variables or a `config.yaml` file:

| Env Variable | Config Key | Description |
|-------------|------------|-------------|
| `GC_REGION` | `gc_region` | Genesys Cloud region |
| `GC_DEPLOYMENT_ID` | `gc_deployment_id` | Web Messaging deployment ID |
| `OLLAMA_BASE_URL` | `ollama_base_url` | Ollama URL (default: http://localhost:11434) |
| `OLLAMA_MODEL` | `ollama_model` | Ollama model name |
| `GC_TESTER_DEFAULT_ATTEMPTS` | `default_attempts` | Default attempts per scenario (default: 5) |
| `GC_TESTER_MAX_TURNS` | `max_turns` | Max conversation turns (default: 20) |
| `GC_TESTER_MIN_ATTEMPT_INTERVAL_SECONDS` | `min_attempt_interval_seconds` | Minimum seconds between attempt starts (default: 60) |
| `GC_TESTER_RESPONSE_TIMEOUT` | `response_timeout` | Timeout in seconds (default: 30) |
| `GC_TESTER_SUCCESS_THRESHOLD` | `success_threshold` | Regression threshold (default: 0.8) |
| `GC_TESTER_EXPECTED_GREETING` | `expected_greeting` | Greeting text required before first user message |

Precedence: Web UI > Environment variables > config.yaml > defaults

## Results

The results page shows per-scenario success rates with all attempts expandable to review the full conversation, including per-message timestamps, per-turn timing, and total attempt duration. Export formats available from the results page:
- CSV summary
- JSON full report
- JUnit XML (CI-friendly)
- ZIP of per-attempt conversation transcripts
- Bundle ZIP containing `report.json`, `report.csv`, `report.junit.xml`, and transcripts

The CLI exits with code 1 if any scenario falls below the success threshold, making it CI/CD friendly.
