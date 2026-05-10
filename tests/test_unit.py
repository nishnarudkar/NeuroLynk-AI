"""
Unit tests for the Healthcare AI Agent components.

Tests cover:
  - FHIR_Formatter.format()
  - Clinical_Summariser._build_prompt() and summarise() in mock mode
  - SessionStore put/get/eviction/size
  - AgentMetadata assembly helpers
"""

import asyncio
import json
import sys
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from unittest.mock import patch, AsyncMock

import pytest

# Ensure project root is on sys.path so api.* imports resolve
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.fhir_formatter import FHIR_Formatter
from api.agent import (
    AgentConfig,
    BiomarkerContribution,
    BiomarkerSummary,
    SessionStore,
    ScreeningResult,
    PlatformPayload,
    AgentMetadata,
    SharpContext,
)
from api.summariser import Clinical_Summariser, LLM_FALLBACK


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_contributions(n: int = 5) -> list[BiomarkerContribution]:
    return [
        BiomarkerContribution(
            feature_index=i,
            feature_name=f"feature_{i}",
            impact=0.1 * (i + 1) * (1 if i % 2 == 0 else -1),
        )
        for i in range(n)
    ]


def _make_screening_result(session_id: str = "test-session-123") -> ScreeningResult:
    contribs = [
        BiomarkerSummary(feature_name=f"feat_{i}", impact=float(i) * 0.1)
        for i in range(5)
    ]
    return ScreeningResult(
        session_id=session_id,
        prediction=1,
        label="Parkinson's Detected",
        probability=0.87,
        top_contributions=contribs,
        clinical_summary="Mock summary.",
        fhir_report={"resourceType": "DiagnosticReport"},
        platform_payload=PlatformPayload(
            session_id=session_id,
            prediction_label="Parkinson's Detected",
            confidence_pct=87.0,
            top_biomarkers=contribs,
            clinical_summary="Mock summary.",
            fhir_report_url=f"/agent/report/{session_id}",
        ),
        agent_metadata=AgentMetadata(
            agent_version="1.0.0-hackathon",
            model_type="XGBClassifier",
            explainer_type="TreeExplainer",
            workflow_steps=["prediction", "shap_extraction"],
            total_duration_ms=42,
        ),
        workflow_error=None,
    )


def _make_mock_config(provider: str = "mock") -> AgentConfig:
    """Create an AgentConfig with all env vars set to avoid WARNING logs."""
    with patch.dict(os.environ, {
        "LLM_PROVIDER": provider,
        "LLM_MODEL": "gpt-4o-mini",
        "LLM_API_KEY": "test-key",
        "LLM_TIMEOUT_SECONDS": "8",
        "AGENT_VERSION": "1.0.0-hackathon",
        "AGENT_MAX_SESSIONS": "100",
    }):
        return AgentConfig()


# ── Task 10.1: FHIR_Formatter tests ──────────────────────────────────────────

class TestFHIRFormatter:

    def _format(self, label: str = "Parkinson's Detected", prob: float = 0.87654321):
        contribs = _make_contributions(5)
        issued = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        return FHIR_Formatter.format(
            session_id="sess-001",
            prediction_label=label,
            probability=prob,
            shap_contributions=contribs,
            issued_at=issued,
        )

    def test_all_required_fields_present(self):
        report = self._format()
        required = ["resourceType", "id", "status", "code", "issued",
                    "performer", "conclusion", "conclusionCode", "result"]
        for field in required:
            assert field in report, f"Missing required FHIR field: {field}"

    def test_resource_type_is_diagnostic_report(self):
        report = self._format()
        assert report["resourceType"] == "DiagnosticReport"

    def test_status_is_final(self):
        report = self._format()
        assert report["status"] == "final"

    def test_snomed_code_parkinsons(self):
        report = self._format(label="Parkinson's Detected")
        code = report["conclusionCode"][0]["coding"][0]["code"]
        assert code == "49049000", f"Expected 49049000, got {code}"

    def test_snomed_code_healthy(self):
        report = self._format(label="Healthy")
        code = report["conclusionCode"][0]["coding"][0]["code"]
        assert code == "17621005", f"Expected 17621005, got {code}"

    def test_snomed_codes_not_swapped(self):
        parkinsons_report = self._format(label="Parkinson's Detected")
        healthy_report = self._format(label="Healthy")
        pk_code = parkinsons_report["conclusionCode"][0]["coding"][0]["code"]
        hl_code = healthy_report["conclusionCode"][0]["coding"][0]["code"]
        assert pk_code != hl_code, "SNOMED codes must differ between labels"
        assert pk_code == "49049000"
        assert hl_code == "17621005"

    def test_probability_rounded_to_4dp(self):
        report = self._format(prob=0.87654321)
        conclusion = report["conclusion"]
        # Should contain 0.8765 (rounded to 4 d.p.)
        assert "0.8765" in conclusion, f"Expected 0.8765 in conclusion, got: {conclusion}"

    def test_result_array_length_5(self):
        report = self._format()
        assert len(report["result"]) == 5

    def test_json_serialisable(self):
        report = self._format()
        serialised = json.dumps(report)
        parsed = json.loads(serialised)
        assert parsed["resourceType"] == "DiagnosticReport"

    def test_session_id_in_report(self):
        contribs = _make_contributions(5)
        issued = datetime(2024, 1, 15, tzinfo=timezone.utc)
        report = FHIR_Formatter.format(
            session_id="my-unique-session",
            prediction_label="Healthy",
            probability=0.1,
            shap_contributions=contribs,
            issued_at=issued,
        )
        assert report["id"] == "my-unique-session"

    def test_issued_is_iso8601(self):
        report = self._format()
        issued = report["issued"]
        # Should be parseable as ISO 8601
        parsed = datetime.fromisoformat(issued)
        assert parsed is not None

    def test_observation_structure(self):
        report = self._format()
        for obs in report["result"]:
            assert obs["resourceType"] == "Observation"
            assert obs["status"] == "final"
            assert "code" in obs
            assert "text" in obs["code"]
            assert "valueQuantity" in obs
            assert "value" in obs["valueQuantity"]
            assert obs["valueQuantity"]["unit"] == "SHAP value"

    def test_sharp_context_adds_subject_and_encounter(self):
        contribs = _make_contributions(5)
        issued = datetime(2024, 1, 15, tzinfo=timezone.utc)
        ctx = SharpContext(patient_id="pt-123", encounter_id="enc-456")
        report = FHIR_Formatter.format(
            session_id="test-session",
            prediction_label="Healthy",
            probability=0.1,
            shap_contributions=contribs,
            issued_at=issued,
            sharp_context=ctx
        )
        assert "subject" in report
        assert report["subject"]["reference"] == "Patient/pt-123"
        assert "encounter" in report
        assert report["encounter"]["reference"] == "Encounter/enc-456"

    def test_sharp_context_with_only_patient(self):
        contribs = _make_contributions(5)
        issued = datetime(2024, 1, 15, tzinfo=timezone.utc)
        ctx = SharpContext(patient_id="pt-123")
        report = FHIR_Formatter.format(
            session_id="test-session",
            prediction_label="Healthy",
            probability=0.1,
            shap_contributions=contribs,
            issued_at=issued,
            sharp_context=ctx
        )
        assert "subject" in report
        assert report["subject"]["reference"] == "Patient/pt-123"
        assert "encounter" not in report


# ── Task 10.2: Clinical_Summariser._build_prompt() tests ─────────────────────

class TestClinicalSummariserBuildPrompt:

    def setup_method(self):
        self.config = _make_mock_config("mock")
        self.summariser = Clinical_Summariser(self.config)

    def test_prompt_contains_prediction_label(self):
        contribs = _make_contributions(5)
        prompt = self.summariser._build_prompt("Parkinson's Detected", 0.87, contribs)
        assert "Parkinson's Detected" in prompt

    def test_prompt_contains_probability(self):
        contribs = _make_contributions(5)
        prompt = self.summariser._build_prompt("Healthy", 0.23, contribs)
        assert "23.0" in prompt  # 0.23 * 100 = 23.0

    def test_prompt_contains_feature_names(self):
        contribs = _make_contributions(5)
        prompt = self.summariser._build_prompt("Healthy", 0.5, contribs)
        for c in contribs:
            assert c.feature_name in prompt, f"Feature {c.feature_name} missing from prompt"

    def test_prompt_contains_directional_indicators(self):
        contribs = _make_contributions(5)
        prompt = self.summariser._build_prompt("Parkinson's Detected", 0.9, contribs)
        assert "toward Parkinson's" in prompt or "toward Healthy" in prompt

    def test_prompt_contains_disclaimer(self):
        contribs = _make_contributions(5)
        prompt = self.summariser._build_prompt("Healthy", 0.1, contribs)
        assert "research purposes only" in prompt
        assert "medical diagnosis" in prompt

    def test_prompt_contains_3_to_5_sentence_instruction(self):
        contribs = _make_contributions(5)
        prompt = self.summariser._build_prompt("Healthy", 0.1, contribs)
        assert "3 to 5 sentences" in prompt or "3–5 sentences" in prompt


# ── Task 10.3: Clinical_Summariser.summarise() in mock mode ──────────────────

class TestClinicalSummariserMockMode:

    def setup_method(self):
        self.config = _make_mock_config("mock")
        self.summariser = Clinical_Summariser(self.config)

    def _run(self, label: str = "Parkinson's Detected", prob: float = 0.87):
        contribs = _make_contributions(5)
        return asyncio.get_event_loop().run_until_complete(
            self.summariser.summarise(label, prob, contribs)
        )

    def test_mock_returns_non_empty_string(self):
        result = self._run()
        assert isinstance(result, str)
        assert len(result.strip()) > 0

    def test_mock_is_deterministic(self):
        result1 = self._run("Parkinson's Detected", 0.87)
        result2 = self._run("Parkinson's Detected", 0.87)
        assert result1 == result2

    def test_mock_no_network_calls(self):
        """Verify mock mode makes no external network calls."""
        with patch("requests.get") as mock_get, patch("requests.post") as mock_post:
            result = self._run()
            mock_get.assert_not_called()
            mock_post.assert_not_called()
        assert isinstance(result, str)

    def test_mock_healthy_label(self):
        result = self._run("Healthy", 0.1)
        assert isinstance(result, str)
        assert len(result.strip()) > 0

    def test_mock_never_returns_none(self):
        result = self._run()
        assert result is not None

    def test_mock_never_raises(self):
        """summarise() must never raise — even with edge-case inputs."""
        contribs = _make_contributions(5)
        result = asyncio.get_event_loop().run_until_complete(
            self.summariser.summarise("Healthy", 0.0, contribs)
        )
        assert isinstance(result, str)


# ── Task 10.4: SessionStore tests ────────────────────────────────────────────

class TestSessionStore:

    def test_put_and_get_round_trip(self):
        store = SessionStore(max_sessions=10)
        result = _make_screening_result("abc-123")
        store.put("abc-123", result)
        retrieved = store.get("abc-123")
        assert retrieved is not None
        assert retrieved.session_id == "abc-123"

    def test_get_returns_none_for_unknown_session(self):
        store = SessionStore(max_sessions=10)
        assert store.get("nonexistent") is None

    def test_size_accuracy(self):
        store = SessionStore(max_sessions=10)
        assert store.size() == 0
        store.put("s1", _make_screening_result("s1"))
        assert store.size() == 1
        store.put("s2", _make_screening_result("s2"))
        assert store.size() == 2

    def test_fifo_eviction_at_cap(self):
        store = SessionStore(max_sessions=3)
        store.put("first", _make_screening_result("first"))
        store.put("second", _make_screening_result("second"))
        store.put("third", _make_screening_result("third"))
        # Adding a 4th should evict "first"
        store.put("fourth", _make_screening_result("fourth"))
        assert store.size() == 3
        assert store.get("first") is None, "Oldest session should have been evicted"
        assert store.get("fourth") is not None

    def test_size_never_exceeds_max(self):
        max_s = 5
        store = SessionStore(max_sessions=max_s)
        for i in range(20):
            store.put(f"session-{i}", _make_screening_result(f"session-{i}"))
        assert store.size() <= max_s

    def test_eviction_preserves_newest_entries(self):
        store = SessionStore(max_sessions=2)
        store.put("old", _make_screening_result("old"))
        store.put("new1", _make_screening_result("new1"))
        store.put("new2", _make_screening_result("new2"))
        # "old" should be evicted; "new1" and "new2" should remain
        assert store.get("old") is None
        assert store.get("new1") is not None
        assert store.get("new2") is not None


# ── Task 10.5: AgentMetadata assembly tests ───────────────────────────────────

class TestAgentMetadataAssembly:

    def test_confidence_pct_rounding(self):
        payload = PlatformPayload(
            session_id="s1",
            prediction_label="Parkinson's Detected",
            confidence_pct=round(0.87654 * 100, 1),
            top_biomarkers=[],
            clinical_summary="test",
            fhir_report_url="/agent/report/s1",
        )
        assert payload.confidence_pct == 87.7

    def test_fhir_report_url_format(self):
        session_id = "my-session-id"
        payload = PlatformPayload(
            session_id=session_id,
            prediction_label="Healthy",
            confidence_pct=10.0,
            top_biomarkers=[],
            clinical_summary="test",
            fhir_report_url=f"/agent/report/{session_id}",
        )
        assert payload.fhir_report_url == f"/agent/report/{session_id}"

    def test_workflow_steps_ordering(self):
        from api.agent import Agent_Orchestrator
        steps = Agent_Orchestrator.WORKFLOW_STEPS
        assert steps[0] == "input_validation"
        assert steps[1] == "prediction"
        assert steps[2] == "shap_extraction"
        assert steps[3] == "clinical_summary"
        assert steps[4] == "fhir_formatting"
        assert steps[5] == "result_assembly"

    def test_agent_metadata_model_type(self):
        meta = AgentMetadata(
            agent_version="1.0.0",
            model_type="XGBClassifier",
            explainer_type="TreeExplainer",
            workflow_steps=["a", "b"],
            total_duration_ms=100,
        )
        assert meta.model_type == "XGBClassifier"
        assert meta.explainer_type == "TreeExplainer"

    def test_total_duration_ms_is_int(self):
        meta = AgentMetadata(
            agent_version="1.0.0",
            model_type="XGBClassifier",
            explainer_type="TreeExplainer",
            workflow_steps=[],
            total_duration_ms=250,
        )
        assert isinstance(meta.total_duration_ms, int)
        assert meta.total_duration_ms == 250
