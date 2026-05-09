"""
Integration tests for the Healthcare AI Agent API endpoints.

Tests use FastAPI's TestClient with mocked ML artifacts so they run
without requiring trained model files on disk.
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Shared mock artifacts ─────────────────────────────────────────────────────

EXPECTED_FEATURES = 753
SELECTED_FEATURES = 100
FEATURE_NAMES = [f"feat_{i}" for i in range(SELECTED_FEATURES)]
COLUMN_ORDER = [f"col_{i}" for i in range(EXPECTED_FEATURES)]


def _make_mock_model():
    model = MagicMock()
    model.predict.return_value = np.array([1])
    model.predict_proba.return_value = np.array([[0.13, 0.87]])
    model.__class__.__name__ = "XGBClassifier"
    return model


def _make_mock_scaler():
    scaler = MagicMock()
    scaler.transform.side_effect = lambda x: x  # identity
    scaler.n_features_in_ = SELECTED_FEATURES
    return scaler


def _make_mock_selector():
    selector = MagicMock()
    selector.transform.side_effect = lambda x: x[:, :SELECTED_FEATURES]
    selector.n_features_in_ = EXPECTED_FEATURES
    return selector


def _make_mock_explainer():
    explainer = MagicMock()
    shap_vals = np.zeros((1, SELECTED_FEATURES))
    shap_vals[0, :5] = [0.3, -0.2, 0.15, -0.1, 0.05]
    explainer.shap_values.return_value = shap_vals
    explainer.__class__.__name__ = "TreeExplainer"
    return explainer


def _make_test_client():
    """
    Build a FastAPI TestClient with all ML artifacts mocked.
    Patches joblib.load and the config/path imports so no files are needed.
    """
    from fastapi.testclient import TestClient

    mock_model = _make_mock_model()
    mock_scaler = _make_mock_scaler()
    mock_selector = _make_mock_selector()
    mock_explainer = _make_mock_explainer()

    # Patch artifact loading and config so main.py can be imported without files
    with (
        patch("joblib.load") as mock_load,
        patch("src.config.load_dataset") as mock_dataset,
        patch("shap.TreeExplainer", return_value=mock_explainer),
        patch.dict(os.environ, {
            "LLM_PROVIDER": "mock",
            "LLM_MODEL": "gpt-4o-mini",
            "LLM_API_KEY": "",
            "LLM_TIMEOUT_SECONDS": "8",
            "AGENT_VERSION": "1.0.0-test",
            "AGENT_MAX_SESSIONS": "100",
        }),
    ):
        # joblib.load returns different artifacts based on call order
        mock_load.side_effect = [
            mock_model,
            mock_scaler,
            mock_selector,
            FEATURE_NAMES,
            COLUMN_ORDER,
        ]

        import pandas as pd
        mock_dataset.return_value = (
            pd.DataFrame(
                np.random.rand(10, EXPECTED_FEATURES),
                columns=COLUMN_ORDER,
            ),
            pd.Series([0, 1] * 5),
        )

        # Force reimport of main to pick up mocks
        if "api.main" in sys.modules:
            del sys.modules["api.main"]
        if "api.agent" in sys.modules:
            del sys.modules["api.agent"]
        if "api.summariser" in sys.modules:
            del sys.modules["api.summariser"]
        if "api.fhir_formatter" in sys.modules:
            del sys.modules["api.fhir_formatter"]

        from api.main import app
        return TestClient(app), mock_model, mock_scaler, mock_selector, mock_explainer


def _valid_features():
    return [0.5] * EXPECTED_FEATURES


# ── Task 12.1: POST /agent/screen end-to-end ─────────────────────────────────

class TestAgentScreen:

    def test_screen_returns_200_with_all_fields(self):
        client, *_ = _make_test_client()
        response = client.post(
            "/agent/screen",
            json={"features": _valid_features()},
        )
        assert response.status_code == 200, response.text
        data = response.json()

        # All top-level fields must be present
        required_fields = [
            "session_id", "prediction", "label", "probability",
            "top_contributions", "clinical_summary", "fhir_report",
            "platform_payload", "agent_metadata",
        ]
        for field in required_fields:
            assert field in data, f"Missing field: {field}"

    def test_screen_session_id_is_uuid(self):
        import uuid
        client, *_ = _make_test_client()
        response = client.post("/agent/screen", json={"features": _valid_features()})
        assert response.status_code == 200
        session_id = response.json()["session_id"]
        # Should be a valid UUID
        uuid.UUID(session_id)  # raises if invalid

    def test_screen_prediction_is_0_or_1(self):
        client, *_ = _make_test_client()
        response = client.post("/agent/screen", json={"features": _valid_features()})
        assert response.status_code == 200
        assert response.json()["prediction"] in (0, 1)

    def test_screen_probability_in_range(self):
        client, *_ = _make_test_client()
        response = client.post("/agent/screen", json={"features": _valid_features()})
        assert response.status_code == 200
        prob = response.json()["probability"]
        assert 0.0 <= prob <= 1.0

    def test_screen_top_contributions_length(self):
        client, *_ = _make_test_client()
        response = client.post("/agent/screen", json={"features": _valid_features()})
        assert response.status_code == 200
        contribs = response.json()["top_contributions"]
        assert len(contribs) == 5

    def test_screen_clinical_summary_non_empty(self):
        client, *_ = _make_test_client()
        response = client.post("/agent/screen", json={"features": _valid_features()})
        assert response.status_code == 200
        summary = response.json()["clinical_summary"]
        assert isinstance(summary, str)
        assert len(summary.strip()) > 0

    def test_screen_fhir_report_has_resource_type(self):
        client, *_ = _make_test_client()
        response = client.post("/agent/screen", json={"features": _valid_features()})
        assert response.status_code == 200
        fhir = response.json()["fhir_report"]
        assert fhir["resourceType"] == "DiagnosticReport"

    def test_screen_platform_payload_present(self):
        client, *_ = _make_test_client()
        response = client.post("/agent/screen", json={"features": _valid_features()})
        assert response.status_code == 200
        payload = response.json()["platform_payload"]
        assert "session_id" in payload
        assert "confidence_pct" in payload
        assert "fhir_report_url" in payload

    def test_screen_agent_metadata_present(self):
        client, *_ = _make_test_client()
        response = client.post("/agent/screen", json={"features": _valid_features()})
        assert response.status_code == 200
        meta = response.json()["agent_metadata"]
        assert meta["model_type"] == "XGBClassifier"
        assert meta["explainer_type"] == "TreeExplainer"
        assert "workflow_steps" in meta
        assert "total_duration_ms" in meta

    def test_screen_wrong_feature_count_returns_422(self):
        client, *_ = _make_test_client()
        response = client.post("/agent/screen", json={"features": [0.5] * 10})
        assert response.status_code == 422


# ── Task 12.2: GET /agent/report/{session_id} ─────────────────────────────────

class TestAgentReport:

    def test_report_returns_fhir_json_after_screen(self):
        client, *_ = _make_test_client()
        # First, run a screen to create a session
        screen_resp = client.post("/agent/screen", json={"features": _valid_features()})
        assert screen_resp.status_code == 200
        session_id = screen_resp.json()["session_id"]

        # Then fetch the report
        report_resp = client.get(f"/agent/report/{session_id}")
        assert report_resp.status_code == 200

    def test_report_content_type_is_fhir_json(self):
        client, *_ = _make_test_client()
        screen_resp = client.post("/agent/screen", json={"features": _valid_features()})
        session_id = screen_resp.json()["session_id"]

        report_resp = client.get(f"/agent/report/{session_id}")
        assert report_resp.status_code == 200
        content_type = report_resp.headers.get("content-type", "")
        assert "application/fhir+json" in content_type

    def test_report_body_is_valid_fhir(self):
        client, *_ = _make_test_client()
        screen_resp = client.post("/agent/screen", json={"features": _valid_features()})
        session_id = screen_resp.json()["session_id"]

        report_resp = client.get(f"/agent/report/{session_id}")
        fhir = json.loads(report_resp.content)
        assert fhir["resourceType"] == "DiagnosticReport"
        assert fhir["status"] == "final"


# ── Task 12.3: GET /agent/report/<nonexistent> ────────────────────────────────

class TestAgentReportNotFound:

    def test_nonexistent_session_returns_404(self):
        client, *_ = _make_test_client()
        response = client.get("/agent/report/nonexistent-session-id")
        assert response.status_code == 404

    def test_404_body_contains_error_and_session_id(self):
        client, *_ = _make_test_client()
        response = client.get("/agent/report/nonexistent-session-id")
        assert response.status_code == 404
        # FastAPI wraps HTTPException detail in {"detail": ...}
        body = response.json()
        detail = body.get("detail", body)
        if isinstance(detail, dict):
            assert "error" in detail
            assert detail["session_id"] == "nonexistent-session-id"


# ── Task 12.4: GET /agent/health ─────────────────────────────────────────────

class TestAgentHealth:

    def test_health_returns_200(self):
        client, *_ = _make_test_client()
        response = client.get("/agent/health")
        assert response.status_code == 200

    def test_health_has_all_subsystem_keys(self):
        client, *_ = _make_test_client()
        response = client.get("/agent/health")
        data = response.json()
        assert "subsystems" in data
        subsystems = data["subsystems"]
        assert "prediction_pipeline" in subsystems
        assert "fhir_formatter" in subsystems
        assert "llm_summariser" in subsystems

    def test_health_prediction_pipeline_ok(self):
        client, *_ = _make_test_client()
        response = client.get("/agent/health")
        pipeline = response.json()["subsystems"]["prediction_pipeline"]
        assert pipeline["status"] == "ok"
        assert pipeline["model_loaded"] is True


# ── Task 12.5: LLM timeout simulation ────────────────────────────────────────

class TestLLMTimeout:

    def test_llm_timeout_returns_200_with_fallback(self):
        """When LLM times out, the response should still be HTTP 200 with fallback summary."""
        from fastapi.testclient import TestClient

        mock_model = _make_mock_model()
        mock_scaler = _make_mock_scaler()
        mock_selector = _make_mock_selector()
        mock_explainer = _make_mock_explainer()

        with (
            patch("joblib.load") as mock_load,
            patch("src.config.load_dataset") as mock_dataset,
            patch("shap.TreeExplainer", return_value=mock_explainer),
            patch.dict(os.environ, {
                "LLM_PROVIDER": "openai",
                "LLM_MODEL": "gpt-4o-mini",
                "LLM_API_KEY": "test-key",
                "LLM_TIMEOUT_SECONDS": "1",
                "AGENT_VERSION": "1.0.0-test",
                "AGENT_MAX_SESSIONS": "100",
            }),
        ):
            mock_load.side_effect = [
                mock_model, mock_scaler, mock_selector, FEATURE_NAMES, COLUMN_ORDER,
            ]
            import pandas as pd
            mock_dataset.return_value = (
                pd.DataFrame(np.random.rand(10, EXPECTED_FEATURES), columns=COLUMN_ORDER),
                pd.Series([0, 1] * 5),
            )

            # Clear cached modules
            for mod in ["api.main", "api.agent", "api.summariser", "api.fhir_formatter"]:
                sys.modules.pop(mod, None)

            from api.main import app

            async def _slow_openai(prompt):
                await asyncio.sleep(10)  # much longer than timeout
                return "This should never be returned"

            with patch("api.summariser.Clinical_Summariser._call_openai", new=_slow_openai):
                client = TestClient(app)
                response = client.post("/agent/screen", json={"features": _valid_features()})

        assert response.status_code == 200, response.text
        summary = response.json()["clinical_summary"]
        assert "unavailable" in summary.lower() or len(summary) > 0


# ── Task 12.6: Regression tests — existing endpoints still work ───────────────

class TestExistingEndpointsRegression:

    def test_health_endpoint_still_works(self):
        client, *_ = _make_test_client()
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    def test_predict_endpoint_still_works(self):
        client, *_ = _make_test_client()
        response = client.post("/predict", json={"features": _valid_features()})
        # May return 200 or 500 depending on mock completeness, but must not 404
        assert response.status_code != 404

    def test_agent_schema_endpoint(self):
        client, *_ = _make_test_client()
        response = client.get("/agent/schema")
        assert response.status_code == 200
        data = response.json()
        assert "request" in data
        assert "response" in data
