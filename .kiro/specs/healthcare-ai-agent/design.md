# Design Document — Healthcare AI Agent

## Overview

This design extends the existing Parkinson's Disease Detection System with an **agent orchestration layer** that wraps the current FastAPI + XGBoost + SHAP pipeline. The extension adds three new Python modules (`api/agent.py`, `api/fhir_formatter.py`, `api/summariser.py`), four new API endpoints under `/agent/*`, a new frontend tab, and LLM-generated clinical summaries — all without touching any existing code.

The system is intended for research and hackathon demonstration. It is not a validated clinical diagnostic tool.

### Goals

- Single `POST /agent/screen` call produces a complete, traceable screening result: prediction + SHAP explanation + FHIR R4 DiagnosticReport + LLM clinical summary.
- Output is consumable by the Prompt Opinion Platform without custom adapters.
- LLM provider is swappable via env vars; a `mock` mode enables fully offline development.
- Zero changes to existing endpoints, artifacts, or CI/CD pipeline.

### Non-Goals

- Persistent storage (database, file-based session store).
- Multi-patient batch processing.
- Real-time streaming of LLM output.
- FHIR server integration (output is FHIR-shaped JSON, not a live FHIR server transaction).

---

## Architecture

The agent layer sits **between** the existing FastAPI app and the outside world. It reuses all loaded artifacts from `api/main.py` at import time — no reloading.

```
┌─────────────────────────────────────────────────────────────────┐
│  FastAPI app  (api/main.py)                                     │
│                                                                 │
│  Existing endpoints (unchanged)          New /agent/* router    │
│  POST /predict                    ──►    POST /agent/screen     │
│  GET  /feature-defaults                  GET  /agent/report/{id}│
│  GET  /health                            GET  /agent/health     │
│  GET  /top-features                      GET  /agent/schema     │
│  GET  /model-comparison                                         │
│  GET  /drift-status                                             │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Agent_Orchestrator  (api/agent.py)                      │   │
│  │                                                          │   │
│  │  1. Validate input (Pydantic — reuses FeatureInput)      │   │
│  │  2. Run Prediction_Pipeline (reuse main.py artifacts)    │   │
│  │  3. Extract top-5 SHAP contributions                     │   │
│  │  4. Clinical_Summariser  (api/summariser.py)             │   │
│  │  5. FHIR_Formatter       (api/fhir_formatter.py)         │   │
│  │  6. Assemble Screening_Result + platform_payload         │   │
│  │  7. Store in SessionStore (FIFO, 100-cap)                │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Module Dependency Graph

```
api/main.py
  └── imports api/agent.py (router mounted at startup)
        ├── uses: model, scaler, selector, feature_names, column_order,
        │         explainer  (all loaded once in main.py, passed by reference)
        ├── api/summariser.py   (Clinical_Summariser)
        └── api/fhir_formatter.py (FHIR_Formatter)
```

### Integration with Existing Code

`api/main.py` mounts the agent router with two lines added at the bottom of the file (after all existing code):

```python
from api.agent import agent_router, init_agent
init_agent(model, scaler, selector, feature_names, column_order, explainer)
app.include_router(agent_router, prefix="/agent")
```

This is the **only** change to `api/main.py`. All existing endpoints, artifact loading, and startup logic remain untouched.

---

## Components and Interfaces

### 1. Agent_Orchestrator (`api/agent.py`)

Owns the FastAPI `APIRouter`, the `SessionStore`, and the top-level workflow function.

```python
class AgentConfig:
    llm_provider: str          # from LLM_PROVIDER env var
    llm_model: str             # from LLM_MODEL env var
    llm_api_key: str           # from LLM_API_KEY env var
    llm_timeout: int           # from LLM_TIMEOUT_SECONDS env var (default 8)
    agent_version: str         # from AGENT_VERSION env var (default "1.0.0-hackathon")
    max_sessions: int          # from AGENT_MAX_SESSIONS env var (default 100)

class SessionStore:
    """Thread-safe FIFO in-memory store, capped at max_sessions."""
    def put(session_id: str, result: ScreeningResult) -> None
    def get(session_id: str) -> ScreeningResult | None
    def size() -> int

class Agent_Orchestrator:
    def __init__(model, scaler, selector, feature_names,
                 column_order, explainer, config: AgentConfig)
    async def run_workflow(features: list[float]) -> ScreeningResult
    # Steps called internally:
    def _predict(features) -> PredictionResult
    def _extract_shap(arr_scaled, top_n=5) -> list[BiomarkerContribution]
    async def _summarise(pred, shap_contribs) -> str
    def _format_fhir(pred, shap_contribs, session_id, issued_at) -> dict
    def _assemble_result(...) -> ScreeningResult
```

**Workflow error handling**: each step is wrapped in a try/except. On failure, the orchestrator returns a `ScreeningResult` with `workflow_error` populated, identifying the failed step by name. LLM failures are non-fatal — the summary field receives the fallback string.

### 2. FHIR_Formatter (`api/fhir_formatter.py`)

Pure function module — no state, no I/O.

```python
class FHIR_Formatter:
    @staticmethod
    def format(
        session_id: str,
        prediction_label: str,
        probability: float,
        shap_contributions: list[BiomarkerContribution],
        issued_at: datetime,
    ) -> dict   # FHIR R4 DiagnosticReport JSON-serialisable dict
```

SNOMED CT code mapping:
- `"Parkinson's Detected"` → `49049000`
- `"Healthy"` → `17621005`

### 3. Clinical_Summariser (`api/summariser.py`)

Handles LLM provider dispatch and mock mode.

```python
class Clinical_Summariser:
    def __init__(config: AgentConfig)
    async def summarise(
        prediction_label: str,
        probability: float,
        shap_contributions: list[BiomarkerContribution],
    ) -> str

    # Internal dispatch:
    async def _call_openai(prompt: str) -> str
    async def _call_mock(prompt: str) -> str   # deterministic, no network
    def _build_prompt(...) -> str
```

**Provider dispatch logic**:
```
LLM_PROVIDER == "mock"   → _call_mock()
LLM_PROVIDER == "openai" → _call_openai()  (uses openai>=1.0 async client)
otherwise                → log warning, fall back to mock
```

**Timeout**: `asyncio.wait_for(coro, timeout=config.llm_timeout)` wraps every real LLM call. On `asyncio.TimeoutError` or any exception, returns the fallback string.

### 4. API Endpoints (mounted on `agent_router`)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/agent/screen` | Full workflow; returns `ScreeningResult` |
| `GET` | `/agent/report/{session_id}` | Returns stored FHIR DiagnosticReport JSON |
| `GET` | `/agent/health` | Subsystem health check |
| `GET` | `/agent/schema` | Request/response JSON schema |

CORS is applied to the entire `agent_router` via FastAPI's `CORSMiddleware` (already present in the app) or a per-router `add_middleware` call. `Content-Type: application/fhir+json` is set on `/agent/report/{session_id}` responses via `Response(content=..., media_type="application/fhir+json")`.

### 5. Frontend — AI Agent Tab (`templates/index.html` + `static/script.js`)

A new tab button is appended to the existing `<nav class="tab-nav">`:

```html
<button class="tab-btn" onclick="openTab('agent', this)">
  <span>🤖</span> AI Agent
</button>
```

A new `<div id="agent" class="tabcontent">` section is appended inside `<main>`. It contains:
- **Clinical summary card** — styled card with fallback warning icon when LLM is unavailable.
- **Top-5 SHAP bar chart** — rendered via Chart.js, consistent with existing dark theme.
- **FHIR JSON collapsible** — `<details>/<summary>` with `<pre>` syntax-highlighted JSON.
- **Metadata footer** — Session_ID, agent version, total workflow duration.

A new `runAgentScreen()` JS function calls `POST /agent/screen` and populates the tab. It reuses the existing `_allColumns` / `_allMedians` feature map already loaded by `loadPredictionFields()`.

---

## Data Models

All models are Pydantic v2 classes unless noted.

### Request

```python
class AgentScreenRequest(BaseModel):
    features: Annotated[List[float], Field(
        min_length=753, max_length=753,
        description="753 numeric speech feature values in column_order"
    )]
```

This is structurally identical to the existing `FeatureInput` — the same validation applies.

### Internal Transfer Objects

```python
@dataclass
class PredictionResult:
    prediction: int          # 0 or 1
    label: str               # "Healthy" | "Parkinson's Detected"
    probability: float       # raw float from predict_proba

@dataclass
class BiomarkerContribution:
    feature_index: int
    feature_name: str
    impact: float            # SHAP value (positive = toward Parkinson's)
```

### Response

```python
class ScreeningResult(BaseModel):
    session_id: str                          # UUID v4
    prediction: int
    label: str
    probability: float
    top_contributions: List[BiomarkerContribution]
    clinical_summary: str
    fhir_report: dict                        # full DiagnosticReport
    platform_payload: PlatformPayload
    agent_metadata: AgentMetadata
    workflow_error: Optional[WorkflowError]  # None on success

class PlatformPayload(BaseModel):
    session_id: str
    prediction_label: str
    confidence_pct: float                    # probability × 100, 1 d.p.
    top_biomarkers: List[BiomarkerSummary]   # top 5 name + impact
    clinical_summary: str
    fhir_report_url: str                     # "/agent/report/{session_id}"

class AgentMetadata(BaseModel):
    agent_version: str
    model_type: str          # "XGBClassifier"
    explainer_type: str      # "TreeExplainer"
    workflow_steps: List[str]
    total_duration_ms: int   # excludes LLM latency

class WorkflowError(BaseModel):
    failed_step: str
    reason: str
    session_id: str
```

### FHIR DiagnosticReport Structure

```json
{
  "resourceType": "DiagnosticReport",
  "id": "<session_id>",
  "status": "final",
  "code": {
    "coding": [{
      "system": "http://loinc.org",
      "code": "11488-4",
      "display": "Consult note"
    }]
  },
  "issued": "<ISO-8601 timestamp>",
  "performer": [{"display": "Parkinson Speech Screening AI Agent"}],
  "conclusion": "<prediction_label> (confidence: <probability>)",
  "conclusionCode": [{
    "coding": [{
      "system": "http://snomed.info/sct",
      "code": "49049000 | 17621005",
      "display": "Parkinson's disease | Normal"
    }]
  }],
  "result": [
    {
      "resourceType": "Observation",
      "id": "<session_id>-obs-<rank>",
      "status": "final",
      "code": {"text": "<feature_name>"},
      "valueQuantity": {
        "value": <shap_impact>,
        "unit": "SHAP value"
      }
    }
  ]
}
```

### Session Store Entry

```python
# Internal dict structure (not exposed via API)
_store: OrderedDict[str, ScreeningResult]
# Key: session_id (UUID v4 string)
# Eviction: popitem(last=False) when len > max_sessions
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `"mock"` | LLM backend: `"mock"` or `"openai"` |
| `LLM_MODEL` | `"gpt-4o-mini"` | Model name passed to the LLM API |
| `LLM_API_KEY` | `""` | API key for the LLM provider |
| `LLM_TIMEOUT_SECONDS` | `8` | Seconds before LLM call is abandoned |
| `AGENT_VERSION` | `"1.0.0-hackathon"` | Reported in `agent_metadata` |
| `AGENT_MAX_SESSIONS` | `100` | FIFO session store capacity |

---

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system — essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: FHIR output is valid JSON with required fields

*For any* valid set of prediction inputs (prediction label, probability, SHAP contributions, session ID, timestamp), the `FHIR_Formatter.format()` function SHALL produce a dict that, when serialised with `json.dumps`, parses back without error and contains the keys `resourceType`, `id`, `status`, `code`, `conclusion`, `result`, `issued`, `performer`, and `conclusionCode`.

**Validates: Requirements 2.1, 2.2, 2.5, 2.6, 2.9**

---

### Property 2: FHIR resourceType and status are always fixed values

*For any* prediction result passed to `FHIR_Formatter.format()`, the returned dict SHALL always have `resourceType == "DiagnosticReport"` and `status == "final"`, regardless of the prediction label or probability value.

**Validates: Requirements 2.2**

---

### Property 3: FHIR conclusionCode maps correctly to prediction label

*For any* prediction label that is either `"Parkinson's Detected"` or `"Healthy"`, the `FHIR_Formatter.format()` function SHALL set the SNOMED CT code to `49049000` for Parkinson's and `17621005` for Healthy — and never swap them.

**Validates: Requirements 2.7, 2.8**

---

### Property 4: FHIR result array contains exactly top-5 SHAP observations

*For any* list of exactly 5 `BiomarkerContribution` objects, the `result` array in the DiagnosticReport SHALL contain exactly 5 `Observation`-structured entries, each referencing the corresponding feature name and SHAP impact value.

**Validates: Requirements 2.4**

---

### Property 5: Session store FIFO eviction preserves capacity invariant

*For any* sequence of `put` operations on the `SessionStore`, the number of stored sessions SHALL never exceed `max_sessions`, and when the cap is reached, the oldest session (first inserted) SHALL be the one evicted.

**Validates: Requirements 5.2**

---

### Property 6: Session round-trip — stored result is retrievable by session_id

*For any* `ScreeningResult` stored via `SessionStore.put(session_id, result)`, a subsequent `SessionStore.get(session_id)` SHALL return an equivalent result, as long as the store has not evicted that session.

**Validates: Requirements 5.1, 5.4**

---

### Property 7: LLM fallback — summary is always a non-empty string

*For any* call to `Clinical_Summariser.summarise()`, the returned value SHALL always be a non-empty string — either the LLM-generated summary or the defined fallback string — and SHALL never be `None`, empty, or raise an exception.

**Validates: Requirements 3.4, 3.5, 1.6**

---

### Property 8: Mock provider returns deterministic summary without network I/O

*For any* prediction inputs passed to `Clinical_Summariser.summarise()` when `LLM_PROVIDER == "mock"`, the function SHALL return a non-empty string without making any external network call, and the same inputs SHALL always produce the same output.

**Validates: Requirements 3.7**

---

### Property 9: platform_payload fields are consistent with top-level result

*For any* completed `ScreeningResult`, the `platform_payload` fields SHALL be derivable from the top-level fields: `confidence_pct == round(probability * 100, 1)`, `prediction_label == label`, `fhir_report_url == f"/agent/report/{session_id}"`, and `top_biomarkers` SHALL match the first 5 entries of `top_contributions`.

**Validates: Requirements 6.1, 6.2**

---

### Property 10: Probability encoding in FHIR is rounded to 4 decimal places

*For any* probability value `p` in `[0.0, 1.0]`, the probability encoded in the FHIR DiagnosticReport `conclusion` field SHALL equal `round(p, 4)`, ensuring no floating-point noise leaks into the structured output.

**Validates: Requirements 2.3**

---

## Error Handling

### Workflow Step Failures

Each step in `Agent_Orchestrator.run_workflow()` is wrapped individually:

```
Step 1 — Input validation:   Pydantic raises ValidationError → HTTP 422 (before workflow starts)
Step 2 — Prediction:         Exception → WorkflowError(failed_step="prediction", ...)
Step 3 — SHAP extraction:    Exception → WorkflowError(failed_step="shap_extraction", ...)
Step 4 — LLM summary:        Timeout / Exception → fallback string (non-fatal, workflow continues)
Step 5 — FHIR formatting:    Exception → WorkflowError(failed_step="fhir_formatting", ...)
Step 6 — Result assembly:    Exception → WorkflowError(failed_step="result_assembly", ...)
```

Fatal steps (2, 3, 5, 6) return HTTP 500 with a `WorkflowError` body. Internal stack traces are logged at `ERROR` level but never included in the response body.

### Session Not Found

`GET /agent/report/{session_id}` returns HTTP 404 with:
```json
{"error": "Session not found", "session_id": "<id>"}
```

### Missing Environment Variables

At module import time, `AgentConfig` reads env vars with `os.getenv(key, default)`. Missing vars log a `WARNING` and use the documented default. No exception is raised, so the server starts in a degraded-but-functional state (e.g., LLM calls will use mock mode if `LLM_PROVIDER` is unset).

### LLM Timeout

`asyncio.wait_for(llm_coro, timeout=config.llm_timeout)` is used. On `asyncio.TimeoutError`, the summariser returns `"Clinical summary unavailable — LLM service did not respond."` The orchestrator logs the timeout at `WARNING` level and continues assembling the result.

### Backward Compatibility

The agent router is mounted **after** all existing route registrations. FastAPI resolves routes in registration order, so no existing route can be shadowed. The `init_agent()` function receives references to already-loaded artifacts — it does not call `joblib.load()` again.

---

## Testing Strategy

### Unit Tests

Focus on pure logic with no I/O:

- `FHIR_Formatter.format()` — verify all required fields, SNOMED codes, probability rounding, result array length.
- `Clinical_Summariser._build_prompt()` — verify prompt contains feature names, directional impact, and disclaimer.
- `Clinical_Summariser.summarise()` in mock mode — verify deterministic output, no network calls.
- `SessionStore.put/get` — verify FIFO eviction at cap, retrieval by key, empty-store 404 path.
- `AgentMetadata` assembly — verify `confidence_pct` rounding, `fhir_report_url` format, `workflow_steps` ordering.

### Property-Based Tests

Use **Hypothesis** (Python) with a minimum of 100 examples per property.

Each test is tagged with a comment in the format:
`# Feature: healthcare-ai-agent, Property <N>: <property_text>`

Properties to implement as Hypothesis tests:

| Property | Strategy |
|----------|----------|
| P1 — FHIR valid JSON + required fields | `st.builds(BiomarkerContribution, ...)`, `st.floats(0,1)`, `st.text()` |
| P2 — resourceType/status fixed | Same as P1 |
| P3 — SNOMED code mapping | `st.sampled_from(["Parkinson's Detected", "Healthy"])` |
| P4 — result array length == 5 | `st.lists(st.builds(...), min_size=5, max_size=5)` |
| P5 — FIFO eviction invariant | `st.integers(1, 200)` for session counts |
| P6 — session round-trip | `st.builds(ScreeningResult, ...)` |
| P7 — summary always non-empty string | `st.floats(0,1)`, `st.lists(...)` with mocked LLM |
| P8 — mock is deterministic + no network | Fixed inputs, assert idempotent |
| P9 — platform_payload consistency | `st.builds(ScreeningResult, ...)` |
| P10 — probability rounding | `st.floats(0.0, 1.0)` |

### Integration Tests

- `POST /agent/screen` end-to-end with mock LLM — verify HTTP 200, all top-level fields present.
- `GET /agent/report/{session_id}` after a screen call — verify FHIR JSON returned with correct `Content-Type`.
- `GET /agent/report/<nonexistent>` — verify HTTP 404 with correct body.
- `GET /agent/health` — verify all subsystems reported.
- `GET /agent/schema` — verify valid JSON schema returned.
- LLM timeout simulation — mock LLM to sleep > timeout, verify fallback string in response.

### Regression Tests

- All existing endpoints (`/predict`, `/health`, `/feature-defaults`, `/model-comparison`, `/drift-status`, `/top-features`) must return HTTP 200 after agent router is mounted — verified by running the existing test suite unchanged.
