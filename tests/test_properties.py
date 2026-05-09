"""
Property-Based Tests for the Healthcare AI Agent.
Uses Hypothesis to verify correctness properties across a wide range of inputs.

Each test is tagged with:
  # Feature: healthcare-ai-agent, Property <N>: <property_text>
"""

import asyncio
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

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
)
from api.summariser import Clinical_Summariser, LLM_FALLBACK


# ── Hypothesis strategies ─────────────────────────────────────────────────────

_biomarker_strategy = st.builds(
    BiomarkerContribution,
    feature_index=st.integers(min_value=0, max_value=99),
    feature_name=st.text(min_size=1, max_size=50, alphabet=st.characters(
        whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="_-"
    )),
    impact=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
)

_five_biomarkers = st.lists(_biomarker_strategy, min_size=5, max_size=5)

_session_id_strategy = st.text(
    min_size=1, max_size=64,
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="-_")
)

_probability_strategy = st.floats(
    min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
)

_label_strategy = st.sampled_from(["Parkinson's Detected", "Healthy"])

_issued_at_strategy = st.datetimes(
    min_value=datetime(2020, 1, 1),
    max_value=datetime(2030, 12, 31),
    timezones=st.just(timezone.utc),
)


def _make_mock_config(provider: str = "mock") -> AgentConfig:
    with patch.dict(os.environ, {
        "LLM_PROVIDER": provider,
        "LLM_MODEL": "gpt-4o-mini",
        "LLM_API_KEY": "test-key",
        "LLM_TIMEOUT_SECONDS": "8",
        "AGENT_VERSION": "1.0.0-hackathon",
        "AGENT_MAX_SESSIONS": "100",
    }):
        return AgentConfig()


def _make_screening_result(session_id: str, probability: float = 0.87) -> ScreeningResult:
    contribs = [
        BiomarkerSummary(feature_name=f"feat_{i}", impact=float(i) * 0.1)
        for i in range(5)
    ]
    return ScreeningResult(
        session_id=session_id,
        prediction=1,
        label="Parkinson's Detected",
        probability=probability,
        top_contributions=contribs,
        clinical_summary="Mock summary.",
        fhir_report={"resourceType": "DiagnosticReport"},
        platform_payload=PlatformPayload(
            session_id=session_id,
            prediction_label="Parkinson's Detected",
            confidence_pct=round(probability * 100, 1),
            top_biomarkers=contribs,
            clinical_summary="Mock summary.",
            fhir_report_url=f"/agent/report/{session_id}",
        ),
        agent_metadata=AgentMetadata(
            agent_version="1.0.0-hackathon",
            model_type="XGBClassifier",
            explainer_type="TreeExplainer",
            workflow_steps=["prediction"],
            total_duration_ms=10,
        ),
        workflow_error=None,
    )


# ── P1: FHIR output is valid JSON with required fields ───────────────────────
# Feature: healthcare-ai-agent, Property 1: FHIR output is valid JSON with required fields

@settings(max_examples=100)
@given(
    session_id=_session_id_strategy,
    prediction_label=_label_strategy,
    probability=_probability_strategy,
    shap_contributions=_five_biomarkers,
    issued_at=_issued_at_strategy,
)
def test_p1_fhir_valid_json_required_fields(
    session_id, prediction_label, probability, shap_contributions, issued_at
):
    # Feature: healthcare-ai-agent, Property 1: FHIR output is valid JSON with required fields
    report = FHIR_Formatter.format(
        session_id=session_id,
        prediction_label=prediction_label,
        probability=probability,
        shap_contributions=shap_contributions,
        issued_at=issued_at,
    )

    # Must be JSON-serialisable
    serialised = json.dumps(report)
    parsed = json.loads(serialised)

    # All required fields must be present
    required = ["resourceType", "id", "status", "code", "conclusion",
                "result", "issued", "performer", "conclusionCode"]
    for field in required:
        assert field in parsed, f"Missing required FHIR field: {field}"


# ── P2: FHIR resourceType and status are always fixed values ─────────────────
# Feature: healthcare-ai-agent, Property 2: FHIR resourceType and status are always fixed values

@settings(max_examples=100)
@given(
    session_id=_session_id_strategy,
    prediction_label=_label_strategy,
    probability=_probability_strategy,
    shap_contributions=_five_biomarkers,
    issued_at=_issued_at_strategy,
)
def test_p2_fhir_resource_type_and_status_fixed(
    session_id, prediction_label, probability, shap_contributions, issued_at
):
    # Feature: healthcare-ai-agent, Property 2: FHIR resourceType and status are always fixed values
    report = FHIR_Formatter.format(
        session_id=session_id,
        prediction_label=prediction_label,
        probability=probability,
        shap_contributions=shap_contributions,
        issued_at=issued_at,
    )
    assert report["resourceType"] == "DiagnosticReport"
    assert report["status"] == "final"


# ── P3: FHIR conclusionCode maps correctly to prediction label ───────────────
# Feature: healthcare-ai-agent, Property 3: FHIR conclusionCode maps correctly to prediction label

@settings(max_examples=100)
@given(
    prediction_label=_label_strategy,
    probability=_probability_strategy,
    shap_contributions=_five_biomarkers,
)
def test_p3_snomed_code_mapping_never_swapped(prediction_label, probability, shap_contributions):
    # Feature: healthcare-ai-agent, Property 3: FHIR conclusionCode maps correctly to prediction label
    report = FHIR_Formatter.format(
        session_id="test-session",
        prediction_label=prediction_label,
        probability=probability,
        shap_contributions=shap_contributions,
        issued_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    code = report["conclusionCode"][0]["coding"][0]["code"]

    if prediction_label == "Parkinson's Detected":
        assert code == "49049000", f"Expected 49049000 for Parkinson's, got {code}"
    elif prediction_label == "Healthy":
        assert code == "17621005", f"Expected 17621005 for Healthy, got {code}"


# ── P4: FHIR result array contains exactly top-5 SHAP observations ───────────
# Feature: healthcare-ai-agent, Property 4: FHIR result array contains exactly top-5 SHAP observations

@settings(max_examples=100)
@given(shap_contributions=_five_biomarkers)
def test_p4_result_array_length_exactly_5(shap_contributions):
    # Feature: healthcare-ai-agent, Property 4: FHIR result array contains exactly top-5 SHAP observations
    report = FHIR_Formatter.format(
        session_id="test",
        prediction_label="Healthy",
        probability=0.5,
        shap_contributions=shap_contributions,
        issued_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    assert len(report["result"]) == 5


# ── P5: Session store FIFO eviction preserves capacity invariant ─────────────
# Feature: healthcare-ai-agent, Property 5: Session store FIFO eviction preserves capacity invariant

@settings(max_examples=100)
@given(n_sessions=st.integers(min_value=1, max_value=200))
def test_p5_fifo_eviction_capacity_invariant(n_sessions):
    # Feature: healthcare-ai-agent, Property 5: Session store FIFO eviction preserves capacity invariant
    max_sessions = 10
    store = SessionStore(max_sessions=max_sessions)

    inserted_ids = []
    for i in range(n_sessions):
        sid = f"session-{i}"
        store.put(sid, _make_screening_result(sid))
        inserted_ids.append(sid)

    # Size must never exceed max_sessions
    assert store.size() <= max_sessions

    # The oldest sessions should have been evicted when cap was exceeded
    if n_sessions > max_sessions:
        # The first (n_sessions - max_sessions) sessions should be evicted
        evicted_count = n_sessions - max_sessions
        for i in range(evicted_count):
            assert store.get(f"session-{i}") is None, (
                f"session-{i} should have been evicted (oldest)"
            )
        # The most recent max_sessions sessions should still be present
        for i in range(evicted_count, n_sessions):
            assert store.get(f"session-{i}") is not None, (
                f"session-{i} should still be in store"
            )


# ── P6: Session round-trip — stored result is retrievable by session_id ───────
# Feature: healthcare-ai-agent, Property 6: Session round-trip — stored result is retrievable by session_id

@settings(max_examples=100)
@given(
    session_id=_session_id_strategy,
    probability=_probability_strategy,
)
def test_p6_session_round_trip(session_id, probability):
    # Feature: healthcare-ai-agent, Property 6: Session round-trip — stored result is retrievable by session_id
    store = SessionStore(max_sessions=100)
    result = _make_screening_result(session_id, probability)
    store.put(session_id, result)

    retrieved = store.get(session_id)
    assert retrieved is not None
    assert retrieved.session_id == session_id
    assert retrieved.probability == probability


# ── P7: LLM fallback — summary is always a non-empty string ──────────────────
# Feature: healthcare-ai-agent, Property 7: LLM fallback — summary is always a non-empty string

@settings(max_examples=100)
@given(
    prediction_label=_label_strategy,
    probability=_probability_strategy,
    shap_contributions=_five_biomarkers,
)
def test_p7_summary_always_non_empty_string_mock(
    prediction_label, probability, shap_contributions
):
    # Feature: healthcare-ai-agent, Property 7: LLM fallback — summary is always a non-empty string
    config = _make_mock_config("mock")
    summariser = Clinical_Summariser(config)

    result = asyncio.get_event_loop().run_until_complete(
        summariser.summarise(prediction_label, probability, shap_contributions)
    )

    assert isinstance(result, str), "Summary must be a string"
    assert len(result.strip()) > 0, "Summary must not be empty or whitespace-only"
    assert result is not None


@settings(max_examples=50)
@given(
    prediction_label=_label_strategy,
    probability=_probability_strategy,
    shap_contributions=_five_biomarkers,
)
def test_p7_summary_always_non_empty_string_on_llm_failure(
    prediction_label, probability, shap_contributions
):
    # Feature: healthcare-ai-agent, Property 7: LLM fallback — summary is always a non-empty string (failure path)
    config = _make_mock_config("openai")
    summariser = Clinical_Summariser(config)

    # Simulate LLM failure by making _call_openai raise
    async def _failing_openai(prompt):
        raise RuntimeError("Simulated LLM failure")

    with patch.object(summariser, "_call_openai", side_effect=_failing_openai):
        result = asyncio.get_event_loop().run_until_complete(
            summariser.summarise(prediction_label, probability, shap_contributions)
        )

    assert isinstance(result, str)
    assert len(result.strip()) > 0


# ── P8: Mock provider returns deterministic summary without network I/O ───────
# Feature: healthcare-ai-agent, Property 8: Mock provider returns deterministic summary without network I/O

@settings(max_examples=50)
@given(
    prediction_label=_label_strategy,
    probability=_probability_strategy,
)
def test_p8_mock_deterministic_no_network(prediction_label, probability):
    # Feature: healthcare-ai-agent, Property 8: Mock provider returns deterministic summary without network I/O
    config = _make_mock_config("mock")
    summariser = Clinical_Summariser(config)
    contribs = [
        BiomarkerContribution(feature_index=i, feature_name=f"feat_{i}", impact=0.1 * i)
        for i in range(5)
    ]

    with patch("requests.get") as mock_get, patch("requests.post") as mock_post:
        result1 = asyncio.get_event_loop().run_until_complete(
            summariser.summarise(prediction_label, probability, contribs)
        )
        result2 = asyncio.get_event_loop().run_until_complete(
            summariser.summarise(prediction_label, probability, contribs)
        )
        mock_get.assert_not_called()
        mock_post.assert_not_called()

    # Same inputs → same output (deterministic)
    assert result1 == result2, "Mock summariser must be deterministic"
    assert isinstance(result1, str)
    assert len(result1.strip()) > 0


# ── P9: platform_payload fields are consistent with top-level result ──────────
# Feature: healthcare-ai-agent, Property 9: platform_payload fields are consistent with top-level result

@settings(max_examples=100)
@given(
    session_id=_session_id_strategy,
    probability=_probability_strategy,
)
def test_p9_platform_payload_consistency(session_id, probability):
    # Feature: healthcare-ai-agent, Property 9: platform_payload fields are consistent with top-level result
    result = _make_screening_result(session_id, probability)

    payload = result.platform_payload

    # confidence_pct == round(probability * 100, 1)
    assert payload.confidence_pct == round(probability * 100, 1), (
        f"confidence_pct mismatch: {payload.confidence_pct} != {round(probability * 100, 1)}"
    )

    # prediction_label == label
    assert payload.prediction_label == result.label

    # fhir_report_url == f"/agent/report/{session_id}"
    assert payload.fhir_report_url == f"/agent/report/{session_id}"

    # top_biomarkers matches first 5 of top_contributions
    assert len(payload.top_biomarkers) == len(result.top_contributions[:5])
    for i, (bm, contrib) in enumerate(
        zip(payload.top_biomarkers, result.top_contributions[:5])
    ):
        assert bm.feature_name == contrib.feature_name, (
            f"top_biomarkers[{i}].feature_name mismatch"
        )
        assert bm.impact == contrib.impact, (
            f"top_biomarkers[{i}].impact mismatch"
        )


# ── P10: Probability encoding in FHIR is rounded to 4 decimal places ─────────
# Feature: healthcare-ai-agent, Property 10: Probability encoding in FHIR is rounded to 4 decimal places

@settings(max_examples=100)
@given(probability=_probability_strategy)
def test_p10_probability_rounding_in_fhir(probability):
    # Feature: healthcare-ai-agent, Property 10: Probability encoding in FHIR is rounded to 4 decimal places
    contribs = [
        BiomarkerContribution(feature_index=i, feature_name=f"feat_{i}", impact=0.0)
        for i in range(5)
    ]
    report = FHIR_Formatter.format(
        session_id="test",
        prediction_label="Healthy",
        probability=probability,
        shap_contributions=contribs,
        issued_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )

    expected_rounded = round(probability, 4)
    conclusion = report["conclusion"]

    # The rounded probability must appear in the conclusion string
    assert str(expected_rounded) in conclusion, (
        f"Expected {expected_rounded} in conclusion, got: {conclusion}"
    )
