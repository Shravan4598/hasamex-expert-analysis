"""
Grounded LLM service for the Hasamex Expert Analysis application.

This module provides a thin, production-oriented wrapper around Google's
Gemini API with built-in retry logic for transient errors.
"""

from __future__ import annotations

import copy
import json
import random
import re
import time
from dataclasses import dataclass
from typing import Any

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

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


class GroundedAnswerSchema(BaseModel):
    """Structured response definition for Gemini API output enforcement."""

    answer: str = Field(description="Synthesized grounded answer.")
    evidence_sufficient: bool = Field(
        description="True if supplied evidence is sufficient to answer."
    )
    confidence: float = Field(description="Confidence score between 0.0 and 1.0.")
    evidence_coverage: float = Field(
        description="Coverage score between 0.0 and 1.0."
    )
    refusal_reason: str | None = Field(
        default=None, description="Reason if evidence is insufficient."
    )
    evidence_indices: list[int] = Field(
        default_factory=list,
        description="1-based indices of referenced evidence blocks.",
    )
    quotes: list[str] = Field(
        default_factory=list,
        description="Exact quote strings copied verbatim from evidence.",
    )


@dataclass(frozen=True)
class LLMGenerationConfig:
    """Runtime generation configuration."""

    temperature: float
    max_output_tokens: int


class GeminiLLMService:
    """Production wrapper around the Gemini API."""

    def __init__(
        self,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()

        self.generation_config = LLMGenerationConfig(
            temperature=self.settings.llm_temperature,
            max_output_tokens=self.settings.llm_max_output_tokens,
        )

        self._client: genai.Client | None = None

    @property
    def model_name(self) -> str:
        """Return the configured Gemini model name, safely correcting to available models."""
        raw_model = getattr(self.settings, "llm_model", "gemini-3.6-flash")
        if not raw_model or "gemini-2." in raw_model or "gemini-1." in raw_model:
            return "gemini-3.6-flash"
        return raw_model

    @property
    def client(self) -> genai.Client:
        """Lazily create and return the Gemini API client."""
        if self._client is None:
            api_key = self.settings.google_api_key.strip()

            if not api_key:
                raise SensorException(
                    "GOOGLE_API_KEY is not configured. "
                    "Set it in the .env file before using LLM analysis.",
                    _sys_module(),
                )

            try:
                self._client = genai.Client(api_key=api_key)
                logger.info(
                    "Initialized Gemini client with model=%s",
                    self.model_name,
                )
            except Exception as error:
                logger.exception("Failed to initialize Gemini client.")
                raise SensorException(
                    str(error),
                    _sys_module(),
                ) from error

        return self._client

    # ------------------------------------------------------------------
    # Network Retry Logic
    # ------------------------------------------------------------------

    def _generate_content_with_retry(
        self, model: str, contents: str, config: Any, max_retries: int = 5
    ) -> Any:
        """Wrapper to call Gemini API with exponential backoff for 503 and 429 errors."""
        for attempt in range(max_retries):
            try:
                return self.client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                )
            except Exception as error:
                error_msg = str(error)
                # Check for rate limiting, high demand, or server exhaustion
                is_transient = any(
                    indicator in error_msg 
                    for indicator in ["429", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "Internal Server Error"]
                )

                if is_transient and attempt < max_retries - 1:
                    # Exponential backoff with jitter (e.g., 2s, 4.x s, 8.x s)
                    sleep_time = (2 ** attempt) + random.uniform(0.5, 2.0)
                    logger.warning(
                        "Gemini API transient error detected (%s...). Retrying in %.1f seconds (Attempt %d/%d).",
                        error_msg.split(".")[0], sleep_time, attempt + 1, max_retries
                    )
                    time.sleep(sleep_time)
                else:
                    # If it's not a transient error, or we ran out of retries, raise the error
                    raise

    # ------------------------------------------------------------------
    # Schema Cleaning Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_schema_dict(schema_dict: dict[str, Any]) -> dict[str, Any]:
        """Recursively strip non-compliant OpenAPI keys for Gemini API."""
        disallowed_keys = {
            "additionalProperties",
            "additional_properties",
            "$schema",
            "title",
        }

        cleaned: dict[str, Any] = {}
        for key, value in schema_dict.items():
            if key in disallowed_keys:
                continue
            if isinstance(value, dict):
                cleaned[key] = GeminiLLMService._clean_schema_dict(value)
            elif isinstance(value, list):
                cleaned[key] = [
                    GeminiLLMService._clean_schema_dict(item)
                    if isinstance(item, dict)
                    else item
                    for item in value
                ]
            else:
                cleaned[key] = value

        return cleaned

    @classmethod
    def _prepare_response_schema(
        cls, schema: type[BaseModel] | dict[str, Any]
    ) -> Any:
        """Prepare clean schema representation compatible with Google GenAI SDK."""
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            return schema
        if isinstance(schema, dict):
            return cls._clean_schema_dict(copy.deepcopy(schema))
        return schema

    # ------------------------------------------------------------------
    # Basic text generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        *,
        system_instruction: str | None = None,
    ) -> str:
        """Generate a grounded text response."""
        try:
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("LLM prompt cannot be empty.")

            instruction = (
                system_instruction.strip()
                if system_instruction
                else SYSTEM_INSTRUCTION
            )

            config = types.GenerateContentConfig(
                system_instruction=instruction,
                temperature=self.generation_config.temperature,
                max_output_tokens=self.generation_config.max_output_tokens,
            )

            response = self._generate_content_with_retry(
                model=self.model_name,
                contents=prompt,
                config=config,
            )

            response_text = self._extract_response_text(response)

            if not response_text:
                raise RuntimeError("Gemini returned an empty response.")

            logger.info("Gemini generation completed successfully.")
            return response_text.strip()

        except SensorException:
            raise
        except Exception as error:
            logger.exception("Gemini generation failed.")
            raise SensorException(
                str(error),
                _sys_module(),
            ) from error

    # ------------------------------------------------------------------
    # Structured JSON generation
    # ------------------------------------------------------------------

    def generate_json(
        self,
        prompt: str,
        *,
        schema: type[BaseModel] | dict[str, Any] | None = None,
        system_instruction: str | None = None,
    ) -> Any:
        """Generate structured JSON response enforceably from Gemini."""
        try:
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("LLM prompt cannot be empty.")

            instruction = (
                system_instruction.strip()
                if system_instruction
                else SYSTEM_INSTRUCTION
            )

            config_kwargs: dict[str, Any] = {
                "system_instruction": instruction,
                "temperature": self.generation_config.temperature,
                "max_output_tokens": self.generation_config.max_output_tokens,
                "response_mime_type": "application/json",
            }

            if schema is not None:
                config_kwargs["response_schema"] = self._prepare_response_schema(schema)

            config = types.GenerateContentConfig(**config_kwargs)

            response = self._generate_content_with_retry(
                model=self.model_name,
                contents=prompt,
                config=config,
            )

            # Check candidate finish reasons for output truncation
            if hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                finish_reason = str(getattr(candidate, "finish_reason", "")).upper()

                if finish_reason in ("MAX_TOKENS", "FINISH_REASON_MAX_TOKENS", "2"):
                    logger.warning(
                        "Gemini response reached MAX_TOKENS limit (%d tokens). "
                        "Response was likely truncated causing empty list or missing fields. "
                        "Increase LLM_MAX_OUTPUT_TOKENS in .env.",
                        self.generation_config.max_output_tokens,
                    )
                elif finish_reason and finish_reason not in ("STOP", "FINISH_REASON_STOP", "1", "0"):
                    logger.warning("Gemini generation finished with reason: %s", finish_reason)

            # Native SDK parsing short-circuit
            if getattr(response, "parsed", None) is not None:
                parsed_obj = response.parsed
                if isinstance(parsed_obj, BaseModel):
                    result = parsed_obj.model_dump()
                elif isinstance(parsed_obj, (dict, list)):
                    result = parsed_obj
                else:
                    result = None

                if result is not None:
                    if isinstance(result, list) and not result:
                        logger.warning("Gemini native parsing returned an empty list.")
                    else:
                        logger.info("Gemini structured generation completed via SDK native parsing.")
                        return result

            response_text = self._extract_response_text(response)

            if not response_text:
                raise RuntimeError("Gemini returned an empty JSON response.")

            parsed = self._parse_json_response(response_text)

            if isinstance(parsed, list) and not parsed:
                logger.warning(
                    "Gemini JSON response parsed to an empty list []. "
                    "Raw response text length: %d chars. Raw preview: %r",
                    len(response_text),
                    response_text[:300],
                )

            logger.info("Gemini structured generation completed successfully.")
            return parsed

        except SensorException:
            raise
        except Exception as error:
            logger.exception("Gemini structured generation failed.")
            raise SensorException(
                str(error),
                _sys_module(),
            ) from error

    # ------------------------------------------------------------------
    # Grounded answer
    # ------------------------------------------------------------------

    def generate_grounded_answer(
        self,
        question: str,
        evidence: list[Evidence],
        *,
        expert_name: str | None = None,
        market: str | None = None,
    ) -> GroundedAnswer:
        """Generate a structured grounded answer from evidence."""
        try:
            if not isinstance(question, str) or not question.strip():
                raise ValueError("Question cannot be empty.")

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
                schema=GroundedAnswerSchema,
            )

            if not isinstance(response, dict):
                raise TypeError("Grounded LLM response must be a JSON object.")

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
            logger.exception("Grounded answer generation failed.")
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
        """Construct a grounded prompt containing only selected evidence."""
        scope_lines: list[str] = []

        if expert_name:
            scope_lines.append(f"Expert scope: {expert_name}")

        if market:
            scope_lines.append(f"Market scope: {market}")

        scope = (
            "\n".join(scope_lines)
            if scope_lines
            else "Scope: all supplied transcript evidence"
        )

        evidence_block = self._serialize_evidence(evidence)

        return f"""
Answer the following question using ONLY the transcript evidence provided below.

QUESTION:
{question}

{scope}

REQUIRED BEHAVIOR:
- Give a concise synthesis of the evidence.
- Do not add facts from outside the evidence.
- Do not claim certainty beyond what the evidence supports.
- If the evidence is insufficient, say so explicitly.
- Do not invent quotes or timestamps.
- If quoting an expert, copy the quote exactly from the supplied evidence.
- The response must distinguish direct expert statements from synthesis.
- evidence_indices must contain only valid evidence numbers.
- quotes must contain exact text copied from the supplied evidence.

TRANSCRIPT EVIDENCE:
{evidence_block}
""".strip()

    @staticmethod
    def _serialize_evidence(evidence: list[Evidence]) -> str:
        """Serialize evidence into a compact, model-readable format safely."""
        blocks: list[str] = []

        for index, item in enumerate(evidence, start=1):
            speaker_val = getattr(item, "speaker", None) or getattr(item, "expert_name", "Unknown")
            source_text_val = getattr(item, "source_text", None) or item.quote
            status_val = getattr(item, "status", None)
            status_str = status_val.value if hasattr(status_val, "value") else str(status_val or "unverified")

            blocks.append(
                "\n".join(
                    [
                        f"[EVIDENCE {index}]",
                        f"Document ID: {getattr(item, 'document_id', 'unknown')}",
                        f"Source file: {getattr(item, 'source_file', 'unknown')}",
                        f"Expert: {getattr(item, 'expert_name', 'unknown')}",
                        f"Market: {getattr(item, 'market', 'unknown')}",
                        f"Speaker: {speaker_val}",
                        f"Timestamp start: {getattr(item, 'start_timestamp', 'unavailable')}",
                        (
                            f"Timestamp end: {item.end_timestamp}"
                            if getattr(item, 'end_timestamp', None)
                            else "Timestamp end: unavailable"
                        ),
                        f"Verified status: {status_str}",
                        f"Quote: {item.quote}",
                        f"Source text: {source_text_val}",
                    ]
                )
            )

        return "\n\n".join(blocks)

    def _parse_grounded_answer_response(
        self,
        question: str,
        response: dict[str, Any],
        evidence: list[Evidence],
        expert_name: str | None,
        market: str | None,
    ) -> GroundedAnswer:
        """Convert the structured LLM response into GroundedAnswer."""
        answer = str(response.get("answer", "")).strip()
        sufficient = bool(response.get("evidence_sufficient", False))
        confidence = self._clamp_score(response.get("confidence", 0.0))
        coverage = self._clamp_score(response.get("evidence_coverage", 0.0))
        refusal_reason = (
            str(response.get("refusal_reason", "")).strip() or None
        )

        evidence_indices = self._safe_indices(
            response.get("evidence_indices", []),
            len(evidence),
        )

        selected_evidence = [
            evidence[index - 1] for index in evidence_indices
        ]

        if sufficient and not selected_evidence:
            logger.warning(
                "LLM marked answer as sufficient without evidence indices."
            )
            sufficient = False
            confidence = 0.0
            coverage = 0.0
            refusal_reason = "The model did not identify supporting evidence."

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

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_indices(values: Any, evidence_count: int) -> list[int]:
        """Validate evidence indices returned by the model."""
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
    def _clamp_score(value: Any) -> float:
        """Convert a model score to a safe [0, 1] range."""
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return 0.0

        return max(0.0, min(1.0, numeric))

    # ------------------------------------------------------------------
    # Extraction and Parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_response_text(response: Any) -> str:
        """Extract text from a Gemini response safely."""
        response_text = getattr(response, "text", None)
        if isinstance(response_text, str):
            return response_text.strip()

        candidates = getattr(response, "candidates", None)
        if not candidates:
            return ""

        collected: list[str] = []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None)
            if not parts:
                continue

            for part in parts:
                part_text = getattr(part, "text", None)
                if isinstance(part_text, str):
                    collected.append(part_text)

        return "\n".join(collected).strip()

    @staticmethod
    def _sanitize_json_string(text: str) -> str:
        """Fix invalid backslash escapes without breaking valid JSON escape sequences."""
        return re.sub(r'\\(?![_"/bfnrtu0-9a-fA-F])', r'\\\\', text)

    @staticmethod
    def _try_load_candidate(candidate: str) -> Any:
        """Try standard loading and backslash sanitization."""
        try:
            return json.loads(candidate, strict=False)
        except json.JSONDecodeError:
            pass

        sanitized = GeminiLLMService._sanitize_json_string(candidate)
        try:
            return json.loads(sanitized, strict=False)
        except json.JSONDecodeError:
            pass

        return None

    @staticmethod
    def _parse_json_response(response_text: str) -> Any:
        """Parse Gemini JSON response defensively across fallback patterns."""
        if not isinstance(response_text, str):
            raise TypeError("Gemini returned a non-text JSON response.")

        cleaned = response_text.strip().lstrip("\ufeff").strip()

        if not cleaned:
            raise ValueError("Gemini returned an empty JSON response.")

        # Stage 1: Standard load or sanitized load
        res = GeminiLLMService._try_load_candidate(cleaned)
        if res is not None:
            return res

        # Stage 2: Markdown code blocks
        if "```" in cleaned:
            for block in GeminiLLMService._extract_fenced_blocks(cleaned):
                res = GeminiLLMService._try_load_candidate(block)
                if res is not None:
                    return res

        # Stage 3: Third-party json_repair (if installed)
        try:
            from json_repair import repair_json

            repaired_str = repair_json(cleaned)
            parsed = json.loads(repaired_str)
            if parsed or isinstance(parsed, (dict, list)):
                return parsed
        except ImportError:
            # json_repair is optional
            pass
        except Exception as error:  # noqa: BLE001
            logger.debug("json_repair fallback failed: %s", error)

        # Stage 4: Balanced Object Extraction
        obj_text = GeminiLLMService._extract_balanced_json(cleaned, "{", "}")
        if obj_text:
            res = GeminiLLMService._try_load_candidate(obj_text)
            if res is not None:
                return res

        # Stage 5: Balanced Array Extraction
        arr_text = GeminiLLMService._extract_balanced_json(cleaned, "[", "]")
        if arr_text:
            res = GeminiLLMService._try_load_candidate(arr_text)
            if res is not None:
                return res

        logger.error("Gemini RAW RESPONSE (Failed to parse): %r", response_text)
        raise ValueError("Gemini returned invalid JSON.")

    @staticmethod
    def _extract_fenced_blocks(text: str) -> list[str]:
        """Extract JSON block candidate strings from markdown code fences."""
        blocks: list[str] = []
        inside = False
        current: list[str] = []

        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("```"):
                if not inside:
                    inside = True
                    current = []
                    continue
                inside = False
                content = "\n".join(current).strip()
                if content.lower().startswith("json"):
                    content = content[4:].strip()
                if content:
                    blocks.append(content)
                current = []
                continue

            if inside:
                current.append(line)

        return blocks

    @staticmethod
    def _extract_balanced_json(
        text: str, opening: str, closing: str
    ) -> str | None:
        """Extract first balanced structure matching opening and closing delimiters."""
        start = text.find(opening)
        if start == -1:
            return None

        depth = 0
        in_string = False
        escaped = False

        for index in range(start, len(text)):
            char = text[index]

            if in_string:
                if escaped:
                    escaped = False
                    continue
                if char == "\\":
                    escaped = True
                    continue
                if char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
                continue

            if char == opening:
                depth += 1
            elif char == closing:
                depth -= 1
                if depth == 0:
                    return text[start : index + 1]

        return text[start:]


def get_llm_service() -> GeminiLLMService:
    """Return a new lightweight Gemini service wrapper."""
    return GeminiLLMService(settings=get_settings())


def _sys_module():
    """Return the active sys module for SensorException."""
    import sys

    return sys


__all__ = [
    "SYSTEM_INSTRUCTION",
    "GeminiLLMService",
    "GroundedAnswerSchema",
    "LLMGenerationConfig",
    "get_llm_service",
]