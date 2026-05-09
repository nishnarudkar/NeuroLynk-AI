"""
Healthcare AI Agent — Orchestration layer for Parkinson's speech screening.

Exposes an APIRouter (agent_router) that is mounted by api/main.py under /agent.
All ML artifacts are injected via init_agent() at startup — no reloading.
"""

from __future__ import annotations

import asyncio
import collections
import logging
import os
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from typing import Annotated
import json

logger = logging.getLogger("uvicorn.error")


# ── SHAP normalisation helper (mirrors api/main.py — avoids circular import) ─
def _extract_shap_for_class1(raw: object) -> np.ndarray:
    """
    Robustly extract a 1-D SHAP array for the positive class (class 1).
    Handles all known output shapes from different SHAP versions.
    """
    if isinstance(raw, list):
        arr = np.array(raw[1])
    else:
        arr = np.array(raw)

    if arr.ndim == 3:
        arr = arr[:, :, 1]

    if arr.ndim == 2:
        arr = arr[0]

    if arr.ndim != 1:
        raise ValueError(
            f"Unexpected SHAP output shape after normalisation: {arr.shape}."
        )
    return arr


# ── Module-level artifact references (set by init_agent) ─────────────────────
_model = None
_scaler = None
_selector = None
_feature_names: list[str] = []
_column_order: list[str] = []
_explainer = None

EXPECTED_RAW_FEATURES = 753


# ─────────────────────────────────────────────────────────────────────────────
# Task 2: Data Models
# ─────────────────────────────────────────────────────────────────────────────

# Task 2.1
@dataclass
class BiomarkerContribution:
    """Internal transfer object for a single SHAP feature contribution."""
    feature_index: int
    feature_name: str
    impact: float  # SHAP value — positive = toward Parkinson's, negative = toward Healthy


# Task 2.2
@dataclass
class PredictionResult:
    """Internal transfer object for model prediction output."""
    prediction: int    # 0 or 1
    label: str         # "Healthy" | "Parkinson's Detected"
    probability: float  # raw float from predict_proba


# Task 2.3
class BiomarkerSummary(BaseModel):
    """Pydantic model for platform_payload top_biomarkers list."""
    feature_name: str
    impact: float


# Task 2.4
class PlatformPayload(BaseModel):
    """Structured payload for the Prompt Opinion Platform."""
    session_id: str
    prediction_label: str
    confidence_pct: float          # probability × 100, rounded to 1 d.p.
    top_biomarkers: List[BiomarkerSummary]  # top 5 name + impact
    clinical_summary: str
    fhir_report_url: str           # "/agent/report/{session_id}"


# Task 2.5
class AgentMetadata(BaseModel):
    """Metadata about the agent run included in every ScreeningResult."""
    agent_version: str
    model_type: str        # "XGBClassifier"
    explainer_type: str    # "TreeExplainer"
    workflow_steps: List[str]
    total_duration_ms: int  # excludes LLM latency


# Task 2.6
class WorkflowError(BaseModel):
    """Populated when a fatal workflow step fails."""
    failed_step: str
    reason: str
    session_id: str


# Task 2.7
class ScreeningResult(BaseModel):
    """Full response from POST /agent/screen."""
    session_id: str
    prediction: int
    label: str
    probability: float
    top_contributions: List[BiomarkerSummary]  # serialisable form of BiomarkerContribution
    clinical_summary: str
    fhir_report: dict
    platform_payload: PlatformPayload
    agent_metadata: AgentMetadata
    workflow_error: Optional[WorkflowError] = None


# Task 2.8
class AgentScreenRequest(BaseModel):
    """Request body for POST /agent/screen."""
    features: Annotated[
        List[float],
        Field(
            min_length=EXPECTED_RAW_FEATURES,
            max_length=EXPECTED_RAW_FEATURES,
            description=f"Exactly {EXPECTED_RAW_FEATURES} numeric speech feature values in column_order",
        ),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Task 3: AgentConfig and SessionStore
# ─────────────────────────────────────────────────────────────────────────────

# Task 3.1
class AgentConfig:
    """
    Reads all agent configuration from environment variables.
    Logs a WARNING for each missing var and applies documented defaults.
    Never raises — server starts in a degraded-but-functional state.
    """

    def __init__(self) -> None:
        self.llm_provider = self._get("LLM_PROVIDER", "mock")
        self.llm_model = self._get("LLM_MODEL", "gpt-4o-mini")
        self.llm_api_key = self._get("LLM_API_KEY", "")
        self.llm_timeout = int(self._get("LLM_TIMEOUT_SECONDS", "8"))
        self.agent_version = self._get("AGENT_VERSION", "1.0.0-hackathon")
        self.max_sessions = int(self._get("AGENT_MAX_SESSIONS", "100"))

    @staticmethod
    def _get(key: str, default: str) -> str:
        value = os.getenv(key)
        if value is None:
            logger.warning(
                "AgentConfig: environment variable %s is not set — using default %r",
                key,
                default,
            )
            return default
        return value


# Task 3.2 / 3.3 / 3.4
class SessionStore:
    """
    Thread-safe FIFO in-memory store for ScreeningResult objects.
    Capped at max_sessions; oldest entry is evicted when the cap is reached.
    """

    def __init__(self, max_sessions: int = 100) -> None:
        self._store: collections.OrderedDict[str, ScreeningResult] = (
            collections.OrderedDict()
        )
        self._lock = threading.Lock()
        self._max_sessions = max_sessions

    # Task 3.3 + 3.4
    def put(self, session_id: str, result: ScreeningResult) -> None:
        with self._lock:
            if len(self._store) >= self._max_sessions:
                # FIFO eviction — remove the oldest entry
                self._store.popitem(last=False)
            self._store[session_id] = result

    # Task 3.2
    def get(self, session_id: str) -> Optional[ScreeningResult]:
        with self._lock:
            return self._store.get(session_id)

    def size(self) -> int:
        with self._lock:
            return len(self._store)


# ─────────────────────────────────────────────────────────────────────────────
# Task 6: Agent_Orchestrator — Core Workflow
# ─────────────────────────────────────────────────────────────────────────────

class Agent_Orchestrator:
    """
    Coordinates the multi-step Parkinson's screening workflow.
    Artifacts are injected at construction time — no joblib.load calls here.
    """

    WORKFLOW_STEPS = [
        "input_validation",
        "prediction",
        "shap_extraction",
        "clinical_summary",
        "fhir_formatting",
        "result_assembly",
    ]

    def __init__(
        self,
        model,
        scaler,
        selector,
        feature_names: list[str],
        column_order: list[str],
        explainer,
        config: AgentConfig,
    ) -> None:
        self._model = model
        self._scaler = scaler
        self._selector = selector
        self._feature_names = feature_names
        self._column_order = column_order
        self._explainer = explainer
        self._config = config

        # Lazy imports to avoid circular deps at module load time
        from api.summariser import Clinical_Summariser
        from api.fhir_formatter import FHIR_Formatter

        self._summariser = Clinical_Summariser(config)
        self._fhir_formatter = FHIR_Formatter

    # Task 6.2
    def _predict(self, features: list[float]) -> PredictionResult:
        """Run the preprocessing + inference pipeline."""
        arr = pd.DataFrame([features], columns=self._column_order).values
        arr_selected = self._selector.transform(arr)   # 753 → 100
        arr_scaled = self._scaler.transform(arr_selected)

        prediction = int(self._model.predict(arr_scaled)[0])
        probability = float(self._model.predict_proba(arr_scaled)[0][1])
        label = "Parkinson's Detected" if prediction == 1 else "Healthy"

        # Store scaled array for SHAP extraction
        self._last_arr_scaled = arr_scaled
        return PredictionResult(prediction=prediction, label=label, probability=probability)

    # Task 6.3
    def _extract_shap(self, arr_scaled: np.ndarray, top_n: int = 5) -> list[BiomarkerContribution]:
        """Extract top-N SHAP contributions sorted by abs(impact) descending."""
        raw_shap = self._explainer.shap_values(arr_scaled)
        shap_vals = _extract_shap_for_class1(raw_shap)  # guaranteed 1-D

        top_indices = np.argsort(np.abs(shap_vals))[-top_n:][::-1]
        return [
            BiomarkerContribution(
                feature_index=int(i),
                feature_name=self._feature_names[i],
                impact=float(shap_vals[i]),
            )
            for i in top_indices
        ]

    # Task 6.4 / 6.5 / 6.6
    async def run_workflow(self, features: list[float]) -> ScreeningResult:
        """
        Execute the full screening workflow and return a ScreeningResult.
        Fatal steps raise HTTPException(500) with a WorkflowError body.
        The LLM step is non-fatal — fallback string is used on failure.
        """
        session_id = str(uuid.uuid4())
        workflow_start = datetime.now(timezone.utc)

        # ── Step 2: Prediction ────────────────────────────────────────────────
        try:
            pred_result = self._predict(features)
            arr_scaled = self._last_arr_scaled
        except Exception as exc:
            logger.error("Agent workflow failed at prediction step: %s", exc)
            raise HTTPException(
                status_code=500,
                detail=WorkflowError(
                    failed_step="prediction",
                    reason=str(exc),
                    session_id=session_id,
                ).model_dump(),
            )

        # ── Step 3: SHAP extraction ───────────────────────────────────────────
        try:
            shap_contributions = self._extract_shap(arr_scaled, top_n=5)
        except Exception as exc:
            logger.error("Agent workflow failed at shap_extraction step: %s", exc)
            raise HTTPException(
                status_code=500,
                detail=WorkflowError(
                    failed_step="shap_extraction",
                    reason=str(exc),
                    session_id=session_id,
                ).model_dump(),
            )

        # ── Step 4: Clinical summary (non-fatal) ──────────────────────────────
        llm_start = datetime.now(timezone.utc)
        clinical_summary = await self._summariser.summarise(
            prediction_label=pred_result.label,
            probability=pred_result.probability,
            shap_contributions=shap_contributions,
        )
        llm_duration_ms = int(
            (datetime.now(timezone.utc) - llm_start).total_seconds() * 1000
        )

        # ── Step 5: FHIR formatting ───────────────────────────────────────────
        issued_at = datetime.now(timezone.utc)
        try:
            fhir_report = self._fhir_formatter.format(
                session_id=session_id,
                prediction_label=pred_result.label,
                probability=pred_result.probability,
                shap_contributions=shap_contributions,
                issued_at=issued_at,
            )
        except Exception as exc:
            logger.error("Agent workflow failed at fhir_formatting step: %s", exc)
            raise HTTPException(
                status_code=500,
                detail=WorkflowError(
                    failed_step="fhir_formatting",
                    reason=str(exc),
                    session_id=session_id,
                ).model_dump(),
            )

        # ── Step 6: Result assembly ───────────────────────────────────────────
        try:
            workflow_end = datetime.now(timezone.utc)
            total_ms = int(
                (workflow_end - workflow_start).total_seconds() * 1000
            ) - llm_duration_ms

            top_biomarkers = [
                BiomarkerSummary(feature_name=c.feature_name, impact=c.impact)
                for c in shap_contributions
            ]

            platform_payload = PlatformPayload(
                session_id=session_id,
                prediction_label=pred_result.label,
                confidence_pct=round(pred_result.probability * 100, 1),
                top_biomarkers=top_biomarkers,
                clinical_summary=clinical_summary,
                fhir_report_url=f"/agent/report/{session_id}",
            )

            agent_metadata = AgentMetadata(
                agent_version=self._config.agent_version,
                model_type="XGBClassifier",
                explainer_type="TreeExplainer",
                workflow_steps=self.WORKFLOW_STEPS,
                total_duration_ms=max(0, total_ms),
            )

            result = ScreeningResult(
                session_id=session_id,
                prediction=pred_result.prediction,
                label=pred_result.label,
                probability=pred_result.probability,
                top_contributions=top_biomarkers,
                clinical_summary=clinical_summary,
                fhir_report=fhir_report,
                platform_payload=platform_payload,
                agent_metadata=agent_metadata,
                workflow_error=None,
            )
        except Exception as exc:
            logger.error("Agent workflow failed at result_assembly step: %s", exc)
            raise HTTPException(
                status_code=500,
                detail=WorkflowError(
                    failed_step="result_assembly",
                    reason=str(exc),
                    session_id=session_id,
                ).model_dump(),
            )

        return result


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singletons (initialised by init_agent)
# ─────────────────────────────────────────────────────────────────────────────

_config: Optional[AgentConfig] = None
_session_store: Optional[SessionStore] = None
_orchestrator: Optional[Agent_Orchestrator] = None


# Task 6.1
def init_agent(model, scaler, selector, feature_names, column_order, explainer) -> None:
    """
    Called once from api/main.py after all artifacts are loaded.
    Stores references and initialises the orchestrator — no joblib.load calls.
    """
    global _model, _scaler, _selector, _feature_names, _column_order, _explainer
    global _config, _session_store, _orchestrator

    _model = model
    _scaler = scaler
    _selector = selector
    _feature_names = feature_names
    _column_order = column_order
    _explainer = explainer

    _config = AgentConfig()
    _session_store = SessionStore(max_sessions=_config.max_sessions)
    _orchestrator = Agent_Orchestrator(
        model=model,
        scaler=scaler,
        selector=selector,
        feature_names=feature_names,
        column_order=column_order,
        explainer=explainer,
        config=_config,
    )
    logger.info(
        "AgentOrchestrator initialised — provider=%s version=%s max_sessions=%d",
        _config.llm_provider,
        _config.agent_version,
        _config.max_sessions,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Task 7: API Endpoints
# ─────────────────────────────────────────────────────────────────────────────

agent_router = APIRouter(tags=["AI Agent"])


# Task 7.1
@agent_router.post("/screen", response_model=ScreeningResult)
async def screen(request: AgentScreenRequest) -> ScreeningResult:
    """
    Full agent workflow: validate → predict → SHAP → summarise → FHIR → assemble.
    Returns a ScreeningResult with all fields populated.
    """
    if _orchestrator is None:
        raise HTTPException(status_code=503, detail="Agent not initialised. Call init_agent() first.")

    result = await _orchestrator.run_workflow(request.features)
    _session_store.put(result.session_id, result)
    return result


# Task 7.2
@agent_router.get("/report/{session_id}")
async def get_report(session_id: str) -> Response:
    """
    Return the FHIR DiagnosticReport JSON for a completed screening.
    Content-Type is set to application/fhir+json.
    """
    if _session_store is None:
        raise HTTPException(status_code=503, detail="Agent not initialised.")

    result = _session_store.get(session_id)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "Session not found", "session_id": session_id},
        )

    return Response(
        content=json.dumps(result.fhir_report),
        media_type="application/fhir+json",
    )


# Task 7.3
@agent_router.get("/health")
async def agent_health() -> dict:
    """Return the status of all agent subsystems."""
    model_loaded = _model is not None
    provider = _config.llm_provider if _config else "unknown"

    # Check LLM reachability (only for openai provider; mock is always reachable)
    llm_reachable = True
    if provider == "openai":
        llm_reachable = bool(_config and _config.llm_api_key)

    return {
        "status": "ok",
        "subsystems": {
            "prediction_pipeline": {
                "status": "ok" if model_loaded else "unavailable",
                "model_loaded": model_loaded,
            },
            "fhir_formatter": {
                "status": "ok",
            },
            "llm_summariser": {
                "status": "ok" if llm_reachable else "degraded",
                "provider": provider,
                "reachable": llm_reachable,
            },
        },
    }


# Task 7.4
@agent_router.get("/schema")
async def agent_schema() -> dict:
    """Return JSON schema for AgentScreenRequest and ScreeningResult."""
    return {
        "request": AgentScreenRequest.model_json_schema(),
        "response": ScreeningResult.model_json_schema(),
    }
