# Requirements Document

## Introduction

This document defines the requirements for adapting and extending the existing Parkinson's Disease Detection System into an **interoperable healthcare AI agent for explainable Parkinson's speech screening**. The goal is to layer agent-like orchestration, FHIR-compatible output, and LLM-generated clinical summaries on top of the existing FastAPI + XGBoost + SHAP pipeline — without rebuilding it. All new capabilities must be implementable within a 3-day hackathon window, reuse existing artifacts and endpoints, and remain deployable via the existing Docker setup.

The system is intended for research and educational demonstration. It is not a validated clinical diagnostic tool.

---

## Glossary

- **Agent**: The orchestration layer that sequences the steps: receive input → validate → predict → explain → summarise → format output. Implemented as a Python module called by the existing FastAPI app.
- **Agent_Orchestrator**: The new module (`api/agent.py`) that coordinates the multi-step screening workflow.
- **FHIR_Formatter**: The module (`api/fhir_formatter.py`) that converts prediction results into a FHIR R4-compatible `DiagnosticReport` JSON resource.
- **Clinical_Summariser**: The module (`api/summariser.py`) that calls an LLM to generate a plain-language clinical summary from SHAP explanations and prediction output.
- **Prediction_Pipeline**: The existing preprocessing and inference chain in `api/main.py` (select → scale → XGBoost → SHAP).
- **SHAP_Explanation**: The per-prediction top-10 SHAP feature contributions already produced by the Prediction_Pipeline.
- **DiagnosticReport**: A FHIR R4 resource representing the structured screening result, including prediction, confidence, and top biomarkers.
- **Prompt_Opinion_Platform**: An external platform that will consume the Agent's structured output via a defined API contract.
- **LLM**: A large language model (e.g., OpenAI GPT-4o-mini or a locally hosted model) used to generate the clinical summary.
- **Session_ID**: A UUID generated per screening run, used to correlate the prediction, FHIR report, and clinical summary.
- **Biomarker**: A speech feature (e.g., `tqwt_TKEO_std_dec_13`) used as model input and referenced in SHAP explanations.
- **Screening_Result**: The combined output of one agent run: prediction label, probability, SHAP contributions, FHIR report, and clinical summary.

---

## Requirements

---

### Requirement 1: Agent Orchestration Workflow

**User Story:** As a researcher or demo viewer, I want the system to execute a structured, multi-step screening workflow automatically, so that a single API call produces a complete, traceable result including prediction, explanation, FHIR output, and clinical summary.

#### Acceptance Criteria

1. WHEN a screening request is received at `POST /agent/screen`, THE Agent_Orchestrator SHALL execute the following steps in order: input validation → prediction → SHAP explanation → clinical summary generation → FHIR formatting → response assembly.
2. THE Agent_Orchestrator SHALL assign a unique Session_ID (UUID v4) to each screening run and include it in the response.
3. WHEN any step in the workflow fails, THE Agent_Orchestrator SHALL return a structured error response that identifies which step failed, the Session_ID, and a human-readable reason, without exposing internal stack traces.
4. THE Agent_Orchestrator SHALL reuse the existing Prediction_Pipeline artifacts (`model.pkl`, `scaler.pkl`, `selector.pkl`, `feature_names.pkl`, `column_order.pkl`) without reloading them on each request.
5. THE Agent_Orchestrator SHALL complete the full workflow and return a response within 10 seconds for a single screening request under normal operating conditions (excluding LLM network latency).
6. WHEN the LLM service is unavailable, THE Agent_Orchestrator SHALL return the Screening_Result with the clinical summary field set to a fallback message indicating the summary is temporarily unavailable, and SHALL NOT fail the entire request.

---

### Requirement 2: FHIR-Compatible Output

**User Story:** As a healthcare platform integrator, I want the screening result formatted as a FHIR R4 DiagnosticReport, so that the output can be consumed by FHIR-compatible systems and the Prompt Opinion Platform without custom parsing.

#### Acceptance Criteria

1. WHEN a screening run completes, THE FHIR_Formatter SHALL produce a JSON object that conforms to the FHIR R4 `DiagnosticReport` resource structure, including `resourceType`, `id`, `status`, `code`, `conclusion`, and `result` fields.
2. THE FHIR_Formatter SHALL set `resourceType` to `"DiagnosticReport"` and `status` to `"final"` for every completed screening.
3. THE FHIR_Formatter SHALL encode the prediction label, probability score (rounded to 4 decimal places), and Session_ID in the DiagnosticReport.
4. THE FHIR_Formatter SHALL include the top 5 SHAP Biomarker contributions as `Observation`-structured entries within the `result` array of the DiagnosticReport.
5. THE FHIR_Formatter SHALL include an `issued` timestamp in ISO 8601 format representing the time the report was generated.
6. THE FHIR_Formatter SHALL include a `performer` field identifying the system as `"Parkinson Speech Screening AI Agent"`.
7. WHEN the prediction label is `"Parkinson's Detected"`, THE FHIR_Formatter SHALL set the `conclusionCode` to SNOMED CT code `49049000` (Parkinson's disease).
8. WHEN the prediction label is `"Healthy"`, THE FHIR_Formatter SHALL set the `conclusionCode` to SNOMED CT code `17621005` (Normal).
9. THE FHIR_Formatter SHALL produce valid JSON that can be parsed without error by a standard JSON parser.

---

### Requirement 3: LLM-Generated Clinical Summary

**User Story:** As a clinician or demo viewer, I want a plain-language summary of the screening result that explains the key biomarker findings, so that I can understand the model's reasoning without reading raw SHAP values.

#### Acceptance Criteria

1. WHEN a screening run completes, THE Clinical_Summariser SHALL generate a clinical summary by sending the prediction result, probability, and top 5 SHAP Biomarker contributions to an LLM.
2. THE Clinical_Summariser SHALL use a structured prompt that instructs the LLM to produce a summary of 3 to 5 sentences, written in plain clinical language, referencing the specific Biomarker names and their directional impact (positive = toward Parkinson's, negative = toward Healthy).
3. THE Clinical_Summariser SHALL include a disclaimer in the prompt instructing the LLM to append the phrase: `"This result is for research purposes only and does not constitute a medical diagnosis."` to every summary.
4. WHEN the LLM returns a response, THE Clinical_Summariser SHALL validate that the response is a non-empty string before including it in the Screening_Result.
5. IF the LLM API call fails or times out after 8 seconds, THEN THE Clinical_Summariser SHALL return the string `"Clinical summary unavailable — LLM service did not respond."` as the summary value.
6. THE Clinical_Summariser SHALL support configuration of the LLM provider and model name via environment variables (`LLM_PROVIDER`, `LLM_MODEL`, `LLM_API_KEY`) so that the provider can be swapped without code changes.
7. WHERE the `LLM_PROVIDER` environment variable is set to `"mock"`, THE Clinical_Summariser SHALL return a deterministic mock summary without making any external API call, enabling offline development and testing.

---

### Requirement 4: Extended API Endpoints

**User Story:** As a developer integrating with the Prompt Opinion Platform, I want well-defined API endpoints that expose the agent workflow, FHIR output, and clinical summary, so that I can build integrations without modifying the existing prediction endpoint.

#### Acceptance Criteria

1. THE Agent_Orchestrator SHALL expose `POST /agent/screen` that accepts the same `features` array (753 floats) as the existing `POST /predict` endpoint and returns the full Screening_Result.
2. THE Agent_Orchestrator SHALL expose `GET /agent/report/{session_id}` that returns the FHIR DiagnosticReport JSON for a completed screening identified by Session_ID.
3. WHEN a `GET /agent/report/{session_id}` request is made for a Session_ID that does not exist, THE Agent_Orchestrator SHALL return HTTP 404 with a JSON body containing `{"error": "Session not found", "session_id": "<id>"}`.
4. THE Agent_Orchestrator SHALL expose `GET /agent/health` that returns the status of all agent subsystems: prediction pipeline, FHIR formatter, and LLM summariser.
5. THE Agent_Orchestrator SHALL expose `GET /agent/schema` that returns the JSON schema of the `POST /agent/screen` request and response bodies.
6. THE Agent_Orchestrator SHALL set `Content-Type: application/fhir+json` on responses from `GET /agent/report/{session_id}`.
7. THE Agent_Orchestrator SHALL include CORS headers on all `/agent/*` endpoints to allow cross-origin requests from the Prompt Opinion Platform.

---

### Requirement 5: In-Memory Session Store

**User Story:** As a developer, I want completed screening results to be retrievable by Session_ID during the server's lifetime, so that the Prompt Opinion Platform can fetch reports asynchronously after the initial screening call.

#### Acceptance Criteria

1. THE Agent_Orchestrator SHALL store each completed Screening_Result in an in-memory dictionary keyed by Session_ID immediately after the workflow completes.
2. THE Agent_Orchestrator SHALL retain at most 100 sessions in memory at any time; WHEN the limit is reached, THE Agent_Orchestrator SHALL evict the oldest session using a first-in-first-out policy.
3. WHEN the server restarts, THE Agent_Orchestrator SHALL start with an empty session store; session persistence across restarts is not required.
4. THE Agent_Orchestrator SHALL return the stored Screening_Result within 50 milliseconds for a `GET /agent/report/{session_id}` lookup on a populated store.

---

### Requirement 6: Prompt Opinion Platform Integration Contract

**User Story:** As a hackathon integrator, I want the agent to produce output in a format that the Prompt Opinion Platform can consume directly, so that the demo can show end-to-end interoperability without custom adapters.

#### Acceptance Criteria

1. THE Agent_Orchestrator SHALL include a top-level `platform_payload` field in the `POST /agent/screen` response containing: `session_id`, `prediction_label`, `confidence_pct` (probability × 100, rounded to 1 decimal), `top_biomarkers` (list of top 5 feature names and SHAP impacts), `clinical_summary`, and `fhir_report_url` (the URL path to retrieve the FHIR report).
2. THE Agent_Orchestrator SHALL format `fhir_report_url` as `/agent/report/{session_id}`.
3. THE Agent_Orchestrator SHALL include an `agent_metadata` field in the response containing: `agent_version` (string), `model_type` (`"XGBClassifier"`), `explainer_type` (`"TreeExplainer"`), `workflow_steps` (ordered list of step names executed), and `total_duration_ms` (integer milliseconds for the full workflow excluding LLM latency).

---

### Requirement 7: Frontend Agent Tab

**User Story:** As a demo viewer, I want a dedicated tab in the existing UI that shows the agent workflow results — including the FHIR report and clinical summary — so that the hackathon demo is self-contained and visually compelling.

#### Acceptance Criteria

1. THE Agent_Orchestrator SHALL serve the updated `index.html` with a new tab labelled `"🤖 AI Agent"` added to the existing tab navigation without removing or modifying any existing tabs.
2. WHEN the `"🤖 AI Agent"` tab is active and the user submits a screening request, THE frontend SHALL call `POST /agent/screen` and display the Screening_Result in the tab.
3. THE frontend SHALL display the clinical summary text in a styled card within the `"🤖 AI Agent"` tab.
4. THE frontend SHALL display the top 5 SHAP Biomarker contributions as a horizontal bar chart within the `"🤖 AI Agent"` tab, consistent with the dark theme of the existing UI.
5. THE frontend SHALL display a collapsible section containing the raw FHIR DiagnosticReport JSON, formatted with syntax highlighting, within the `"🤖 AI Agent"` tab.
6. THE frontend SHALL display the Session_ID, agent version, and total workflow duration in a metadata footer within the `"🤖 AI Agent"` tab.
7. WHEN the LLM summary is unavailable, THE frontend SHALL display the fallback message in the clinical summary card with a visual indicator (e.g., a warning icon) distinguishing it from a successful summary.

---

### Requirement 8: Configuration and Environment

**User Story:** As a developer deploying the system, I want all new agent configuration to be managed via environment variables consistent with the existing `.env` pattern, so that the deployment process does not change.

#### Acceptance Criteria

1. THE Agent_Orchestrator SHALL read all LLM configuration from environment variables: `LLM_PROVIDER`, `LLM_MODEL`, `LLM_API_KEY`, and `LLM_TIMEOUT_SECONDS` (default: `8`).
2. THE Agent_Orchestrator SHALL read `AGENT_VERSION` from environment variables (default: `"1.0.0-hackathon"`).
3. THE Agent_Orchestrator SHALL read `AGENT_MAX_SESSIONS` from environment variables (default: `100`).
4. THE `.env.example` file SHALL be updated to include all new environment variable keys with placeholder values and inline comments describing each variable.
5. WHEN required environment variables are missing at startup, THE Agent_Orchestrator SHALL log a warning and apply the documented default values rather than raising an exception, so that the system starts in a degraded-but-functional state.

---

### Requirement 9: Backward Compatibility

**User Story:** As an existing user of the system, I want all existing endpoints and UI tabs to continue working exactly as before, so that the agent extension does not break the current demo or CI/CD pipeline.

#### Acceptance Criteria

1. THE Agent_Orchestrator SHALL be implemented as additive changes only; THE existing `POST /predict`, `GET /feature-defaults`, `GET /model-comparison`, `GET /drift-status`, `GET /top-features`, and `GET /health` endpoints SHALL remain unchanged.
2. THE existing six UI tabs (Feature Importance, Learning Curve, Prediction, Model Comparison, Feature Insights, Drift Monitor) SHALL remain fully functional after the agent extension is applied.
3. THE Agent_Orchestrator SHALL not modify `src/train.py`, `src/explain.py`, `src/learning_curve.py`, `src/config.py`, or any existing model artifacts.
4. THE existing Docker build and `uvicorn api.main:app` startup command SHALL continue to work without modification after the agent extension is applied.
5. THE existing Jenkins CI/CD pipeline stages SHALL continue to pass after the agent extension is applied.
