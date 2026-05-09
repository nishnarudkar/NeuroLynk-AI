"""
Clinical_Summariser — generates plain-language clinical summaries from SHAP explanations.

Supports multiple LLM providers via LLM_PROVIDER env var:
  - "mock"   → deterministic offline summary (no network I/O)
  - "openai" → OpenAI chat completions API (requires LLM_API_KEY)
  - "gemini" → Google Gemini API via google-generativeai (requires LLM_API_KEY)
  - other    → logs warning, falls back to mock
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from api.agent import AgentConfig, BiomarkerContribution

logger = logging.getLogger("uvicorn.error")

# Fallback string returned when the LLM is unavailable or returns empty content
LLM_FALLBACK = "Clinical summary unavailable — LLM service did not respond."


class Clinical_Summariser:
    """
    Dispatches clinical summary generation to the configured LLM provider.
    Always returns a non-empty string — never raises, never returns None.
    """

    # Task 5.1
    def __init__(self, config: "AgentConfig") -> None:
        self._config = config

    # Task 5.2
    def _build_prompt(
        self,
        prediction_label: str,
        probability: float,
        shap_contributions: list,  # list[BiomarkerContribution]
    ) -> str:
        """
        Build a structured prompt for the LLM.

        Includes:
        - Prediction label and probability
        - Top-5 biomarker names with directional impact
          (positive SHAP = toward Parkinson's, negative = toward Healthy)
        - Instruction for 3–5 sentence plain clinical language summary
        - Mandatory disclaimer sentence
        """
        confidence_pct = round(probability * 100, 1)

        # Build biomarker lines with directional indicators
        biomarker_lines = []
        for contrib in shap_contributions[:5]:
            direction = "toward Parkinson's" if contrib.impact >= 0 else "toward Healthy"
            sign = "+" if contrib.impact >= 0 else ""
            biomarker_lines.append(
                f"  - {contrib.feature_name}: SHAP impact = {sign}{contrib.impact:.4f} ({direction})"
            )
        biomarkers_text = "\n".join(biomarker_lines)

        prompt = (
            f"You are a clinical AI assistant summarising a Parkinson's disease speech screening result.\n\n"
            f"Screening result:\n"
            f"  - Prediction: {prediction_label}\n"
            f"  - Confidence: {confidence_pct}%\n\n"
            f"Top 5 contributing speech biomarkers (SHAP values):\n"
            f"{biomarkers_text}\n\n"
            f"Instructions:\n"
            f"1. Write a summary of 3 to 5 sentences in plain clinical language.\n"
            f"2. Reference the specific biomarker names listed above.\n"
            f"3. Explain the directional impact of each biomarker "
            f"(positive SHAP = pushes toward Parkinson's, negative = pushes toward Healthy).\n"
            f"4. End your summary with exactly this sentence: "
            f"\"This result is for research purposes only and does not constitute a medical diagnosis.\"\n\n"
            f"Summary:"
        )
        return prompt

    # Task 5.3
    async def _call_mock(self, prompt: str) -> str:
        """
        Return a deterministic mock summary without any network I/O.
        Suitable for offline development and testing.
        The output is derived solely from the prompt content to ensure determinism.
        """
        # Extract prediction label from prompt for a slightly more informative mock
        if "Parkinson's Detected" in prompt:
            label = "Parkinson's Detected"
            detail = (
                "The analysis identified elevated SHAP contributions from several speech biomarkers, "
                "suggesting irregular vocal patterns consistent with Parkinsonian speech characteristics. "
                "Key features showed positive SHAP values, indicating their influence toward the Parkinson's classification. "
                "These findings reflect measurable changes in speech dynamics that the model associates with the condition."
            )
        else:
            label = "Healthy"
            detail = (
                "The analysis found that the dominant speech biomarkers showed negative SHAP values, "
                "indicating their influence toward a Healthy classification. "
                "The vocal patterns captured by the top features appear within the range associated with healthy speech. "
                "No strong indicators of Parkinsonian speech irregularities were detected by the model."
            )

        return (
            f"The screening model classified this sample as {label}. "
            f"{detail} "
            f"This result is for research purposes only and does not constitute a medical diagnosis."
        )

    # Task 5.4
    async def _call_openai(self, prompt: str) -> str:
        """
        Call the OpenAI chat completions API using the async client.
        Wrapped in asyncio.wait_for with the configured timeout.
        """
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=self._config.llm_api_key)

        async def _request() -> str:
            response = await client.chat.completions.create(
                model=self._config.llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.3,
            )
            content = response.choices[0].message.content
            return content if content else ""

        return await asyncio.wait_for(_request(), timeout=self._config.llm_timeout)

    async def _call_gemini(self, prompt: str) -> str:
        """
        Call the Google Gemini API using google-generativeai.
        Runs the synchronous SDK call in a thread to avoid blocking the event loop.
        Wrapped in asyncio.wait_for with the configured timeout.
        """
        import google.generativeai as genai

        genai.configure(api_key=self._config.llm_api_key)
        model = genai.GenerativeModel(self._config.llm_model)

        generation_config = {
            "max_output_tokens": 300,
            "temperature": 0.3,
        }

        async def _request() -> str:
            # google-generativeai is synchronous — run in thread pool
            response = await asyncio.to_thread(
                model.generate_content,
                prompt,
                generation_config=generation_config,
            )
            return response.text or ""

        return await asyncio.wait_for(_request(), timeout=self._config.llm_timeout)

    # Task 5.5 / 5.6 / 5.7
    async def summarise(
        self,
        prediction_label: str,
        probability: float,
        shap_contributions: list,  # list[BiomarkerContribution]
    ) -> str:
        """
        Generate a clinical summary for the given screening result.

        Routes to the appropriate provider based on LLM_PROVIDER config.
        Always returns a non-empty string — never raises, never returns None.
        """
        try:
            prompt = self._build_prompt(prediction_label, probability, shap_contributions)

            # Task 5.5 — provider dispatch
            provider = self._config.llm_provider.lower()
            if provider == "mock":
                result = await self._call_mock(prompt)
            elif provider == "openai":
                result = await self._call_openai(prompt)
            elif provider == "gemini":
                result = await self._call_gemini(prompt)
            else:
                logger.warning(
                    "Clinical_Summariser: unknown LLM_PROVIDER %r — falling back to mock",
                    self._config.llm_provider,
                )
                result = await self._call_mock(prompt)

            # Task 5.7 — validate non-empty response
            if not result or not result.strip():
                logger.warning(
                    "Clinical_Summariser: LLM returned empty response — using fallback"
                )
                return LLM_FALLBACK

            return result

        except asyncio.TimeoutError:
            # Task 5.6 — timeout handling
            logger.warning(
                "Clinical_Summariser: LLM call timed out after %ds — using fallback",
                self._config.llm_timeout,
            )
            return LLM_FALLBACK
        except Exception as exc:
            # Task 5.6 — any other exception
            logger.warning(
                "Clinical_Summariser: LLM call failed (%s) — using fallback", exc
            )
            return LLM_FALLBACK
