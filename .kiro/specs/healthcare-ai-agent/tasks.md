# Tasks — Healthcare AI Agent

## Task List

- [x] 1. Environment & Configuration Setup
  - [x] 1.1 Update `.env.example` with all new agent environment variable keys and inline comments (`LLM_PROVIDER`, `LLM_MODEL`, `LLM_API_KEY`, `LLM_TIMEOUT_SECONDS`, `AGENT_VERSION`, `AGENT_MAX_SESSIONS`)
  - [x] 1.2 Add `openai>=1.0` and `hypothesis` to `requirements.txt` (dev/test deps); add `openai>=1.0` to `requirements-api.txt` (runtime dep)

- [x] 2. Data Models (`api/agent.py` — shared types)
  - [x] 2.1 Define `BiomarkerContribution` dataclass with fields: `feature_index: int`, `feature_name: str`, `impact: float`
  - [x] 2.2 Define `PredictionResult` dataclass with fields: `prediction: int`, `label: str`, `probability: float`
  - [x] 2.3 Define `BiomarkerSummary` Pydantic model with fields: `feature_name: str`, `impact: float`
  - [x] 2.4 Define `PlatformPayload` Pydantic model with fields: `session_id`, `prediction_label`, `confidence_pct`, `top_biomarkers`, `clinical_summary`, `fhir_report_url`
  - [x] 2.5 Define `AgentMetadata` Pydantic model with fields: `agent_version`, `model_type`, `explainer_type`, `workflow_steps`, `total_duration_ms`
  - [x] 2.6 Define `WorkflowError` Pydantic model with fields: `failed_step: str`, `reason: str`, `session_id: str`
  - [x] 2.7 Define `ScreeningResult` Pydantic model composing all above models plus `prediction`, `label`, `probability`, `top_contributions`, `clinical_summary`, `fhir_report`, `workflow_error`
  - [x] 2.8 Define `AgentScreenRequest` Pydantic model with `features: List[float]` (min/max length 753, matching existing `FeatureInput`)

- [x] 3. `AgentConfig` and `SessionStore` (`api/agent.py`)
  - [x] 3.1 Implement `AgentConfig` that reads all six env vars via `os.getenv` with documented defaults; log a `WARNING` for each missing var rather than raising
  - [x] 3.2 Implement `SessionStore` as a thread-safe class wrapping `collections.OrderedDict` with `put(session_id, result)`, `get(session_id) -> ScreeningResult | None`, and `size() -> int` methods
  - [x] 3.3 Implement FIFO eviction in `SessionStore.put()`: when `len(_store) >= max_sessions`, call `_store.popitem(last=False)` before inserting the new entry
  - [x] 3.4 Protect `SessionStore` mutations with `threading.Lock` to ensure thread safety under concurrent FastAPI requests

- [x] 4. `FHIR_Formatter` (`api/fhir_formatter.py`)
  - [x] 4.1 Implement `FHIR_Formatter.format()` as a static method accepting `session_id`, `prediction_label`, `probability`, `shap_contributions: list[BiomarkerContribution]`, `issued_at: datetime`
  - [x] 4.2 Build the DiagnosticReport dict with all required FHIR R4 fields: `resourceType`, `id`, `status`, `code`, `issued`, `performer`, `conclusion`, `conclusionCode`, `result`
  - [x] 4.3 Set `resourceType = "DiagnosticReport"` and `status = "final"` unconditionally
  - [x] 4.4 Implement SNOMED CT code mapping: `"Parkinson's Detected"` → `49049000`, `"Healthy"` → `17621005`
  - [x] 4.5 Encode `probability` rounded to 4 decimal places in the `conclusion` string
  - [x] 4.6 Build the `result` array from the top-5 `shap_contributions` as `Observation`-structured dicts, each with `resourceType`, `id`, `status`, `code.text`, and `valueQuantity`
  - [x] 4.7 Format `issued_at` as ISO 8601 string using `datetime.isoformat()`

- [x] 5. `Clinical_Summariser` (`api/summariser.py`)
  - [x] 5.1 Implement `Clinical_Summariser.__init__()` accepting `AgentConfig`; store config
  - [x] 5.2 Implement `_build_prompt()` that constructs a structured prompt including: prediction label, probability, top-5 biomarker names with directional impact (positive = toward Parkinson's, negative = toward Healthy), instruction for 3–5 sentence plain clinical language summary, and the mandatory disclaimer sentence
  - [x] 5.3 Implement `_call_mock()` that returns a deterministic mock summary string without any network I/O (suitable for offline dev and testing)
  - [x] 5.4 Implement `_call_openai()` using the `openai` async client (`AsyncOpenAI`), calling `chat.completions.create` with the built prompt; wrap in `asyncio.wait_for` with `config.llm_timeout`
  - [x] 5.5 Implement `summarise()` dispatcher: route to `_call_mock()` when `LLM_PROVIDER == "mock"`, to `_call_openai()` when `LLM_PROVIDER == "openai"`, log a warning and fall back to mock for unknown providers
  - [x] 5.6 Wrap the entire `summarise()` body in try/except; on any exception (including `asyncio.TimeoutError`), return `"Clinical summary unavailable — LLM service did not respond."` and log at `WARNING` level
  - [x] 5.7 After receiving an LLM response, validate it is a non-empty string; if empty or whitespace-only, return the fallback string

- [x] 6. `Agent_Orchestrator` — Core Workflow (`api/agent.py`)
  - [x] 6.1 Implement `init_agent()` module-level function that receives `model, scaler, selector, feature_names, column_order, explainer` from `api/main.py` and stores them as module-level references (no `joblib.load` calls)
  - [x] 6.2 Implement `_predict()` internal method: build DataFrame from features + column_order, apply `selector.transform()` then `scaler.transform()` (matching train.py order), call `model.predict()` and `model.predict_proba()`, return `PredictionResult`
  - [x] 6.3 Implement `_extract_shap()` internal method: call `explainer.shap_values()`, apply existing `extract_shap_for_class1()` logic (import from `api/main.py` or duplicate), return top-5 `BiomarkerContribution` objects sorted by `abs(impact)` descending
  - [x] 6.4 Implement `run_workflow()` async method: generate UUID v4 session_id, record start time, call steps 2–6 in order, record end time, compute `total_duration_ms` (excluding LLM latency), assemble and return `ScreeningResult`
  - [x] 6.5 Wrap each fatal step (predict, shap, fhir, assemble) in individual try/except blocks; on exception, log at `ERROR` level and raise `HTTPException(500)` with `WorkflowError` body (no stack trace in response)
  - [x] 6.6 Treat LLM step as non-fatal: if `summarise()` returns the fallback string, continue workflow normally; never raise from the LLM step

- [x] 7. API Endpoints (`api/agent.py` — `agent_router`)
  - [x] 7.1 Implement `POST /agent/screen`: validate request with `AgentScreenRequest`, call `orchestrator.run_workflow()`, store result in `session_store`, return `ScreeningResult`
  - [x] 7.2 Implement `GET /agent/report/{session_id}`: look up session in `session_store`; if found, return FHIR JSON with `Response(content=json.dumps(result.fhir_report), media_type="application/fhir+json")`; if not found, return HTTP 404 with `{"error": "Session not found", "session_id": "<id>"}`
  - [x] 7.3 Implement `GET /agent/health`: return dict with status of all subsystems — prediction pipeline (model loaded), FHIR formatter (always ok), LLM summariser (provider name + reachability flag)
  - [x] 7.4 Implement `GET /agent/schema`: return JSON schema dict for `AgentScreenRequest` and `ScreeningResult` using Pydantic's `.model_json_schema()`
  - [x] 7.5 Apply CORS to the agent router: add `CORSMiddleware` to the FastAPI app in `api/main.py` (or configure per-router) allowing all origins on `/agent/*` paths

- [x] 8. Integration into `api/main.py`
  - [x] 8.1 Add two lines at the bottom of `api/main.py` (after all existing code): `from api.agent import agent_router, init_agent` and `init_agent(model, scaler, selector, feature_names, column_order, explainer)`
  - [x] 8.2 Mount the agent router: `app.include_router(agent_router, prefix="/agent")`
  - [x] 8.3 Verify no existing route is shadowed by running the app and confirming all existing endpoints still return HTTP 200

- [x] 9. Frontend — AI Agent Tab
  - [x] 9.1 Add `<button class="tab-btn" onclick="openTab('agent', this)"><span>🤖</span> AI Agent</button>` to the tab navigation in `templates/index.html` without modifying any existing tab buttons
  - [x] 9.2 Add `<div id="agent" class="tabcontent">` section to `templates/index.html` containing: clinical summary card, top-5 SHAP bar chart canvas, collapsible FHIR JSON section, and metadata footer
  - [x] 9.3 Add `runAgentScreen()` function to `static/script.js` that: reuses `_allColumns`/`_allMedians` feature map, calls `POST /agent/screen`, and populates the agent tab with the response
  - [x] 9.4 Render the clinical summary in a styled card; when the summary equals the fallback string, display a `⚠️` warning icon alongside the text
  - [x] 9.5 Render the top-5 SHAP contributions as a horizontal bar chart using Chart.js, matching the existing dark theme (red for positive impact, green for negative)
  - [x] 9.6 Render the FHIR DiagnosticReport JSON in a `<details>/<summary>` collapsible with `<pre>` syntax highlighting using `JSON.stringify(data, null, 2)`
  - [x] 9.7 Render the metadata footer showing Session_ID, agent version, and total workflow duration in milliseconds
  - [x] 9.8 Update `openTab()` in `static/script.js` to handle `tab === "agent"` (no auto-load needed; user triggers via the Run button)

- [x] 10. Unit Tests
  - [x] 10.1 Write unit tests for `FHIR_Formatter.format()`: verify all required fields present, SNOMED code mapping for both labels, probability rounded to 4 d.p., result array length == 5, JSON serialisability
  - [x] 10.2 Write unit tests for `Clinical_Summariser._build_prompt()`: verify prompt contains feature names, directional indicators, and disclaimer text
  - [x] 10.3 Write unit tests for `Clinical_Summariser.summarise()` in mock mode: verify deterministic output and no network calls (use `unittest.mock.patch`)
  - [x] 10.4 Write unit tests for `SessionStore`: verify put/get round-trip, FIFO eviction at cap, `get()` returns `None` for unknown session_id, `size()` accuracy
  - [x] 10.5 Write unit tests for `AgentMetadata` assembly: verify `confidence_pct` rounding, `fhir_report_url` format, `workflow_steps` ordering

- [x] 11. Property-Based Tests (Hypothesis)
  - [x] 11.1 Write property test for P1 (FHIR valid JSON + required fields): use `st.builds` for `BiomarkerContribution`, `st.floats(0,1)`, `st.text()` for session_id; assert all required FHIR fields present and `json.dumps` succeeds
  - [x] 11.2 Write property test for P2 (resourceType/status fixed): same strategy as P1; assert `resourceType == "DiagnosticReport"` and `status == "final"` for all generated inputs
  - [x] 11.3 Write property test for P3 (SNOMED code mapping): use `st.sampled_from(["Parkinson's Detected", "Healthy"])`; assert correct SNOMED code for each label, never swapped
  - [x] 11.4 Write property test for P4 (result array length == 5): use `st.lists(st.builds(BiomarkerContribution, ...), min_size=5, max_size=5)`; assert `len(fhir["result"]) == 5`
  - [x] 11.5 Write property test for P5 (FIFO eviction invariant): use `st.integers(1, 200)` for number of sessions to insert; assert `store.size() <= max_sessions` always, and evicted session is the oldest
  - [x] 11.6 Write property test for P6 (session round-trip): generate random `ScreeningResult` objects, put then get, assert retrieved result equals stored result
  - [x] 11.7 Write property test for P7 (summary always non-empty string): use `st.floats(0,1)` and `st.lists` for contributions with mocked LLM (both success and failure paths); assert result is always a non-empty string
  - [x] 11.8 Write property test for P8 (mock is deterministic + no network): call `summarise()` twice with identical inputs in mock mode; assert outputs are equal and `requests.get/post` was never called
  - [x] 11.9 Write property test for P9 (platform_payload consistency): generate random `ScreeningResult` objects; assert `confidence_pct == round(probability * 100, 1)`, `prediction_label == label`, `fhir_report_url == f"/agent/report/{session_id}"`, `top_biomarkers` matches first 5 of `top_contributions`
  - [x] 11.10 Write property test for P10 (probability rounding): use `st.floats(0.0, 1.0)`; assert probability in FHIR conclusion equals `round(p, 4)`

- [x] 12. Integration Tests
  - [x] 12.1 Write integration test for `POST /agent/screen` end-to-end with mock LLM: verify HTTP 200, all top-level fields present in response, session stored
  - [x] 12.2 Write integration test for `GET /agent/report/{session_id}` after a screen call: verify FHIR JSON returned, `Content-Type: application/fhir+json` header set
  - [x] 12.3 Write integration test for `GET /agent/report/<nonexistent>`: verify HTTP 404 with `{"error": "Session not found", "session_id": "<id>"}` body
  - [x] 12.4 Write integration test for `GET /agent/health`: verify all subsystem keys present in response
  - [x] 12.5 Write integration test for LLM timeout simulation: mock LLM to sleep > `LLM_TIMEOUT_SECONDS`; verify response still returns HTTP 200 with fallback summary string
  - [x] 12.6 Write regression tests confirming all existing endpoints (`/predict`, `/health`, `/feature-defaults`, `/model-comparison`, `/drift-status`, `/top-features`) return HTTP 200 after agent router is mounted
