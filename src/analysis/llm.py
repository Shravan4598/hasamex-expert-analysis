"""
Grounded LLM service for the Hasamex Expert Analysis application.

This module provides a thin, production-oriented wrapper around Google's
Gemini API.

Design principles:
    1. The LLM is an analysis component, not the source of truth.
    2. Retrieved transcript evidence must be supplied explicitly.
    3. The model is instructed not to invent facts, quotes, experts, or
       timestamps.
    4. Exact quotes must be copied from supplied evidence rather than
       generated from memory.
    5. Low-evidence situations should result in an explicit insufficient-
       evidence response.
    6. API credentials are loaded from application settings and are never
       hard-coded.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from google import genai
from google.genai import types

from exception import SensorException
from logger import logging
from src.config import Settings, get_settings
from src.models import Evidence, GroundedAnswer

logger = logging.getLogger(__name__)


SYSTEM_INSTRUCTION = """
You are the grounded analysis engine for the Hasamex Expert Analysis
application.

Your task is to analyze expert interview transcripts using ONLY the
evidence supplied in the current request.

STRICT SOURCE-GROUNDING RULES:

1. Do not use outside knowledge.
2. Do not invent facts, names, dates, timestamps, markets, roles, or
   opinions.
3. Do not infer a statement that is not reasonably supported by the
   supplied evidence.
4. If the evidence does not support an answer, explicitly state that
   there is insufficient evidence.
5. Never invent a timestamp.
6. Never invent an expert attribution.
7. Never create an exact quote from memory.
8. Exact quotes must be copied verbatim from the supplied evidence.
9. Preserve the meaning of expert statements.
10. Distinguish between:
       - what an expert explicitly said,
       - a synthesis across experts,
       - and evidence that is insufficient.
11. Do not present a model inference as an expert quotation.
12. When experts differ, describe the difference without declaring one
    expert correct unless the supplied evidence itself establishes this.
13. Prefer precise, concise answers over unsupported detail.
14. Every material claim should be traceable to supplied evidence.

The transcript evidence is authoritative for this task.
""".strip()


@dataclass(frozen=True)
class LLMGenerationConfig:
    """
    Runtime generation configuration.
    """

    temperature: float
    max_output_tokens: int


class GeminiLLMService:
    """
    Production wrapper around the Gemini API.

    The service is intentionally small. Retrieval, evidence selection,
    quote verification, and citation construction remain separate
    responsibilities.
    """

    def __init__(
        self,
        settings: Settings | None = None,
    ) -> None:
        """
        Initialize the Gemini client lazily.

        Args:
            settings: Optional application settings instance.
        """
        self.settings = settings or get_settings()

        self.generation_config = LLMGenerationConfig(
            temperature=self.settings.llm_temperature,
            max_output_tokens=self.settings.llm_max_output_tokens,
        )

        self._client: genai.Client | None = None

    @property
    def model_name(self) -> str:
        """Return the configured Gemini model name."""
        return self.settings.llm_model

    @property
    def client(self) -> genai.Client:
        """
        Lazily create and return the Gemini API client.
        """
        if self._client is None:
            api_key = self.settings.google_api_key.strip()

            if not api_key:
                raise SensorException(
                    "GOOGLE_API_KEY is not configured. "
                    "Set it in the .env file before using LLM analysis.",
                    _sys_module(),
                )

            try:
                self._client = genai.Client(
                    api_key=api_key,
                )

                logger.info(
                    "Initialized Gemini client with model=%s",
                    self.model_name,
                )

            except Exception as error:
                logger.exception(
                    "Failed to initialize Gemini client."
                )
                raise SensorException(
                    str(error),
                    _sys_module(),
                ) from error

        return self._client

    def generate(
        self,
        prompt: str,
        *,
        system_instruction: str | None = None,
    ) -> str:
        """
        Generate a grounded text response.

        Args:
            prompt: User/task prompt containing retrieved evidence.
            system_instruction: Optional system instruction. When omitted,
                the application's strict grounding instruction is used.

        Returns:
            Generated text.

        Raises:
            SensorException: If configuration, API invocation, or response
                handling fails.
        """
        try:
            if not prompt.strip():
                raise ValueError(
                    "LLM prompt cannot be empty."
                )

            instruction = (
                system_instruction.strip()
                if system_instruction
                else SYSTEM_INSTRUCTION
            )

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=instruction,
                    temperature=self.generation_config.temperature,
                    max_output_tokens=(
                        self.generation_config.max_output_tokens
                    ),
                ),
            )

            text = self._extract_response_text(response)

            if not text:
                raise RuntimeError(
                    "Gemini returned an empty response."
                )

            logger.info(
                "Gemini generation completed successfully."
            )

            return text.strip()

        except SensorException:
            raise

        except Exception as error:
            logger.exception(
                "Gemini generation failed."
            )
            raise SensorException(
                str(error),
                _sys_module(),
            ) from error

    def generate_json(
        self,
        prompt: str,
        *,
        schema: dict[str, Any] | None = None,
        system_instruction: str | None = None,
    ) -> dict[str, Any]:
        """
        Generate a JSON object from Gemini.

        Structured output is requested from the API when a schema is
        supplied. A defensive JSON extraction fallback is included for
        model/API responses that wrap valid JSON in Markdown fences.

        Args:
            prompt: Grounded analysis prompt.
            schema: Optional JSON schema describing the expected object.
            system_instruction: Optional custom system instruction.

        Returns:
            Parsed JSON dictionary.
        """
        try:
            if not prompt.strip():
                raise ValueError(
                    "LLM prompt cannot be empty."
                )

            instruction = (
                system_instruction.strip()
                if system_instruction
                else SYSTEM_INSTRUCTION
            )

            config_kwargs: dict[str, Any] = {
                "system_instruction": instruction,
                "temperature": self.generation_config.temperature,
                "max_output_tokens": (
                    self.generation_config.max_output_tokens
                ),
                "response_mime_type": "application/json",
            }

            if schema:
                config_kwargs["response_schema"] = schema

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    **config_kwargs,
                ),
            )

            response_text = self._extract_response_text(response)

            if not response_text:
                raise RuntimeError(
                    "Gemini returned an empty JSON response."
                )

            parsed = self._parse_json_response(
                response_text
            )

            if not isinstance(parsed, dict):
                raise TypeError(
                    "Gemini JSON response must be an object."
                )

            logger.info(
                "Gemini structured generation completed successfully."
            )

            return parsed

        except SensorException:
            raise

        except Exception as error:
            logger.exception(
                "Gemini structured generation failed."
            )
            raise SensorException(
                str(error),
                _sys_module(),
            ) from error

    def generate_grounded_answer(
        self,
        question: str,
        evidence: list[Evidence],
        *,
        expert_name: str | None = None,
        market: str | None = None,
    ) -> GroundedAnswer:
        """
        Generate a structured grounded answer from evidence.

        This method is intentionally conservative. If no evidence is
        supplied, the method does not ask the model to answer from its
        own knowledge.

        Args:
            question: User/interview-guide question.
            evidence: Evidence already retrieved and selected.
            expert_name: Optional expert-specific scope.
            market: Optional market-specific scope.

        Returns:
            GroundedAnswer.
        """
        try:
            if not question.strip():
                raise ValueError(
                    "Question cannot be empty."
                )

            if not evidence:
                return GroundedAnswer(
                    question=question,
                    expert_name=expert_name,
                    market=market,
                    answer=(
                        "Insufficient evidence in the provided "
                        "transcript sources to answer this question."
                    ),
                    evidence=[],
                    citations=[],
                    confidence=0.0,
                    evidence_coverage=0.0,
                    evidence_sufficient=False,
                    refusal_reason=(
                        "No supporting transcript evidence was retrieved."
                    ),
                )

            prompt = self._build_grounded_answer_prompt(
                question=question,
                evidence=evidence,
                expert_name=expert_name,
                market=market,
            )

            response = self.generate_json(
                prompt=prompt,
                schema=self._grounded_answer_schema(),
            )

            return self._parse_grounded_answer_response(
                question=question,
                response=response,
                evidence=evidence,
                expert_name=expert_name,
                market=market,
            )

        except SensorException:
            raise

        except Exception as error:
            logger.exception(
                "Grounded answer generation failed."
            )
            raise SensorException(
                str(error),
                _sys_module(),
            ) from error

    def _build_grounded_answer_prompt(
        self,
        question: str,
        evidence: list[Evidence],
        expert_name: str | None,
        market: str | None,
    ) -> str:
        """
        Construct a grounded prompt containing only selected evidence.
        """
        scope_lines: list[str] = []

        if expert_name:
            scope_lines.append(
                f"Expert scope: {expert_name}"
            )

        if market:
            scope_lines.append(
                f"Market scope: {market}"
            )

        scope = (
            "\n".join(scope_lines)
            if scope_lines
            else "Scope: all supplied transcript evidence"
        )

        evidence_block = self._serialize_evidence(
            evidence
        )

        return f"""
Answer the following question using ONLY the transcript evidence
provided below.

QUESTION:
{question}

{scope}

REQUIRED BEHAVIOR:

- Give a concise synthesis of the evidence.
- Do not add facts from outside the evidence.
- Do not claim certainty beyond what the evidence supports.
- If the evidence is insufficient, say so explicitly.
- Do not invent quotes.
- Do not invent timestamps.
- If quoting an expert, copy the quote exactly from the supplied
  evidence.
- The response must distinguish direct expert statements from synthesis.
- Return only the requested JSON structure.

TRANSCRIPT EVIDENCE:
{evidence_block}
""".strip()

    @staticmethod
    def _serialize_evidence(
        evidence: list[Evidence],
    ) -> str:
        """
        Serialize evidence into a compact, model-readable format.
        """
        blocks: list[str] = []

        for index, item in enumerate(evidence, start=1):
            blocks.append(
                "\n".join(
                    [
                        f"[EVIDENCE {index}]",
                        f"Document ID: {item.document_id}",
                        f"Source file: {item.source_file}",
                        f"Expert: {item.expert_name}",
                        f"Market: {item.market}",
                        f"Speaker: {item.speaker}",
                        f"Timestamp start: {item.start_timestamp}",
                        (
                            f"Timestamp end: {item.end_timestamp}"
                            if item.end_timestamp
                            else "Timestamp end: unavailable"
                        ),
                        f"Verified status: {item.status.value}",
                        f"Quote: {item.quote}",
                        f"Source text: {item.source_text}",
                    ]
                )
            )

        return "\n\n".join(blocks)

    @staticmethod
    def _grounded_answer_schema() -> dict[str, Any]:
        """
        Return the JSON schema requested from Gemini.

        The schema deliberately excludes source identifiers from generated
        output. Source identifiers are controlled by the application from
        the original Evidence objects, preventing the LLM from fabricating
        citation metadata.
        """
        return {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                },
                "evidence_sufficient": {
                    "type": "boolean",
                },
                "confidence": {
                    "type": "number",
                },
                "evidence_coverage": {
                    "type": "number",
                },
                "refusal_reason": {
                    "type": "string",
                },
                "evidence_indices": {
                    "type": "array",
                    "items": {
                        "type": "integer",
                    },
                },
                "quotes": {
                    "type": "array",
                    "items": {
                        "type": "string",
                    },
                },
            },
            "required": [
                "answer",
                "evidence_sufficient",
                "confidence",
                "evidence_coverage",
                "refusal_reason",
                "evidence_indices",
                "quotes",
            ],
        }

    def _parse_grounded_answer_response(
        self,
        question: str,
        response: dict[str, Any],
        evidence: list[Evidence],
        expert_name: str | None,
        market: str | None,
    ) -> GroundedAnswer:
        """
        Convert the LLM response into the application's GroundedAnswer.

        Importantly, evidence and citations are taken from application
        objects rather than trusting source metadata generated by the LLM.
        """
        answer = str(
            response.get(
                "answer",
                "",
            )
        ).strip()

        if not answer:
            raise ValueError(
                "Grounded LLM response did not contain an answer."
            )

        raw_sufficient = response.get(
            "evidence_sufficient",
            False,
        )

        raw_confidence = response.get(
            "confidence",
            0.0,
        )

        raw_coverage = response.get(
            "evidence_coverage",
            0.0,
        )

        sufficient = bool(
            raw_sufficient
        )

        confidence = self._clamp_score(
            raw_confidence
        )

        coverage = self._clamp_score(
            raw_coverage
        )

        refusal_reason = str(
            response.get(
                "refusal_reason",
                "",
            )
        ).strip() or None

        evidence_indices = self._safe_indices(
            response.get(
                "evidence_indices",
                [],
            ),
            len(evidence),
        )

        selected_evidence = [
            evidence[index - 1]
            for index in evidence_indices
        ]

        # If the model claims there is sufficient evidence but does not
        # identify any supporting evidence, fail closed.
        if sufficient and not selected_evidence:
            logger.warning(
                "LLM marked answer as sufficient without evidence indices."
            )

            sufficient = False
            confidence = 0.0
            coverage = 0.0
            refusal_reason = (
                "The model did not identify supporting evidence."
            )

        # Likewise, an empty answer is never considered grounded.
        if not answer:
            sufficient = False

        return GroundedAnswer(
            question=question,
            expert_name=expert_name,
            market=market,
            answer=answer,
            evidence=selected_evidence,
            citations=[],
            confidence=confidence,
            evidence_coverage=coverage,
            evidence_sufficient=sufficient,
            refusal_reason=refusal_reason,
        )

    @staticmethod
    def _safe_indices(
        values: Any,
        evidence_count: int,
    ) -> list[int]:
        """
        Validate evidence indices returned by the model.

        Invalid indices are discarded rather than trusted.
        """
        if not isinstance(values, list):
            return []

        valid: list[int] = []
        seen: set[int] = set()

        for value in values:
            try:
                index = int(value)
            except (TypeError, ValueError):
                continue

            if index < 1 or index > evidence_count:
                continue

            if index in seen:
                continue

            seen.add(index)
            valid.append(index)

        return valid

    @staticmethod
    def _clamp_score(
        value: Any,
    ) -> float:
        """
        Convert a model-produced score into a safe [0, 1] value.
        """
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return 0.0

        return max(
            0.0,
            min(
                1.0,
                numeric,
            ),
        )

    @staticmethod
    def _extract_response_text(
        response: Any,
    ) -> str:
        """
        Extract text from a Gemini response defensively.
        """
        text = getattr(
            response,
            "text",
            None,
        )

        if isinstance(text, str):
            return text.strip()

        # Defensive fallback for SDK response variants.
        candidates = getattr(
            response,
            "candidates",
            None,
        )

        if not candidates:
            return ""

        collected: list[str] = []

        for candidate in candidates:
            content = getattr(
                candidate,
                "content",
                None,
            )

            parts = getattr(
                content,
                "parts",
                None,
            )

            if not parts:
                continue

            for part in parts:
                part_text = getattr(
                    part,
                    "text",
                    None,
                )

                if isinstance(part_text, str):
                    collected.append(
                        part_text
                    )

        return "\n".join(
            collected
        ).strip()

    @staticmethod
    def _parse_json_response(
        response_text: str,
    ) -> dict[str, Any]:
        """
        Parse a JSON response, including common Markdown-fenced output.
        """
        cleaned = response_text.strip()

        if cleaned.startswith("```"):
            lines = cleaned.splitlines()

            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]

            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]

            cleaned = "\n".join(lines).strip()

        try:
            parsed = json.loads(
                cleaned
            )
        except json.JSONDecodeError as error:
            raise ValueError(
                "Gemini returned invalid JSON."
            ) from error

        if not isinstance(parsed, dict):
            raise TypeError(
                "Gemini JSON response must be an object."
            )

        return parsed


def get_llm_service() -> GeminiLLMService:
    """
    Return a new lightweight Gemini service wrapper.

    The underlying API client is initialized lazily.
    """
    return GeminiLLMService(
        settings=get_settings()
    )


def _sys_module():
    """Return the active sys module for SensorException."""
    import sys

    return sys


__all__ = [
    "SYSTEM_INSTRUCTION",
    "GeminiLLMService",
    "LLMGenerationConfig",
    "get_llm_service",
]