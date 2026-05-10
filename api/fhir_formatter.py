"""
FHIR_Formatter — converts Parkinson's screening results into FHIR R4 DiagnosticReport JSON.

Pure function module: no state, no I/O, no side effects.
All methods are static — instantiation is not required.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from api.agent import BiomarkerContribution, SharpContext

# Task 4.4 — SNOMED CT code mapping
_SNOMED_CODES: dict[str, str] = {
    "Parkinson's Detected": "49049000",
    "Healthy": "17621005",
}

_SNOMED_DISPLAYS: dict[str, str] = {
    "Parkinson's Detected": "Parkinson's disease",
    "Healthy": "Normal",
}


class FHIR_Formatter:
    """
    Converts prediction results into a FHIR R4-compatible DiagnosticReport dict.
    The returned dict is JSON-serialisable with the standard `json` module.
    """

    # Task 4.1
    @staticmethod
    def format(
        session_id: str,
        prediction_label: str,
        probability: float,
        shap_contributions: list,  # list[BiomarkerContribution]
        issued_at: datetime,
        sharp_context: Optional['SharpContext'] = None,
    ) -> dict:
        """
        Build a FHIR R4 DiagnosticReport dict.

        Args:
            session_id:          UUID v4 string identifying the screening run.
            prediction_label:    "Parkinson's Detected" or "Healthy".
            probability:         Raw float from model.predict_proba (0.0–1.0).
            shap_contributions:  List of BiomarkerContribution objects (top-5 expected).
            issued_at:           Datetime when the report was generated.
            sharp_context:       Optional SHARP context with patient and encounter IDs.

        Returns:
            A JSON-serialisable dict conforming to FHIR R4 DiagnosticReport.
        """
        # Task 4.4 — SNOMED code lookup
        snomed_code = _SNOMED_CODES.get(prediction_label, "unknown")
        snomed_display = _SNOMED_DISPLAYS.get(prediction_label, prediction_label)

        # Task 4.5 — probability rounded to 4 decimal places
        prob_rounded = round(probability, 4)

        # Task 4.7 — ISO 8601 timestamp
        issued_str = issued_at.isoformat()

        # Task 4.6 — top-5 SHAP contributions as Observation entries
        result_array = FHIR_Formatter._build_observations(session_id, shap_contributions)

        # Task 4.2 / 4.3 — DiagnosticReport with all required FHIR R4 fields
        report: dict = {
            # Task 4.3 — fixed values
            "resourceType": "DiagnosticReport",
            "status": "final",
            # Task 4.2 — required fields
            "id": session_id,
            "code": {
                "coding": [
                    {
                        "system": "http://loinc.org",
                        "code": "11488-4",
                        "display": "Consult note",
                    }
                ]
            },
            "issued": issued_str,
            "performer": [{"display": "Parkinson Speech Screening AI Agent"}],
            "conclusion": (
                f"{prediction_label} (confidence: {prob_rounded})"
            ),
            "conclusionCode": [
                {
                    "coding": [
                        {
                            "system": "http://snomed.info/sct",
                            "code": snomed_code,
                            "display": snomed_display,
                        }
                    ]
                }
            ],
            "result": result_array,
        }

        # Task 4.8 — SHAP Extension Context Binding
        if sharp_context:
            if sharp_context.patient_id:
                report["subject"] = {"reference": f"Patient/{sharp_context.patient_id}"}
            if sharp_context.encounter_id:
                report["encounter"] = {"reference": f"Encounter/{sharp_context.encounter_id}"}

        return report

    @staticmethod
    def _build_observations(session_id: str, shap_contributions: list) -> list[dict]:
        """
        Build the FHIR result array from the top-5 SHAP contributions.
        Each entry is an Observation-structured dict.
        """
        observations = []
        # Task 4.6 — use top-5 contributions (caller is responsible for passing exactly 5)
        for rank, contrib in enumerate(shap_contributions[:5], start=1):
            obs: dict = {
                "resourceType": "Observation",
                "id": f"{session_id}-obs-{rank}",
                "status": "final",
                "code": {
                    "text": contrib.feature_name,
                },
                "valueQuantity": {
                    "value": contrib.impact,
                    "unit": "SHAP value",
                },
            }
            observations.append(obs)
        return observations
