from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .constants import DECISIONS, STAGES
from .schema import StageStateBundle, _extract_stage_states, _norm_decision

JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class ParsedPrediction:
    sample_id: str
    valid_json: bool
    final_decision: str
    policy_id: str
    stage_states: StageStateBundle
    legacy_stage_labels: bool
    raw_json: dict[str, Any] | None
    parse_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.sample_id,
            "valid_json": self.valid_json,
            "final_decision": self.final_decision,
            "policy_id": self.policy_id,
            "stage_states": self.stage_states.to_dict(),
            "legacy_stage_labels": self.legacy_stage_labels,
            "parse_error": self.parse_error,
        }


def parse_prediction(row_or_text: dict[str, Any] | str | None) -> ParsedPrediction:
    if isinstance(row_or_text, dict):
        raw_output = row_or_text.get("model_output", row_or_text.get("output"))
        if raw_output is not None and not _looks_like_prediction_obj(row_or_text):
            parsed = parse_prediction(raw_output)
            return ParsedPrediction(
                sample_id=str(row_or_text.get("id") or row_or_text.get("sample_id") or parsed.sample_id),
                valid_json=parsed.valid_json,
                final_decision=parsed.final_decision,
                policy_id=parsed.policy_id,
                stage_states=parsed.stage_states,
                legacy_stage_labels=parsed.legacy_stage_labels,
                raw_json=parsed.raw_json,
                parse_error=parsed.parse_error,
            )
        return _from_obj(row_or_text, valid_json=True)
    if not row_or_text:
        return _invalid("empty output")
    candidate = extract_json_object(str(row_or_text))
    if candidate is None:
        return _invalid("no json object found")
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError as exc:
        return _invalid(str(exc))
    if not isinstance(obj, dict):
        return _invalid("json root is not object")
    return _from_obj(obj, valid_json=True)


def parse_predictions(rows: list[dict[str, Any]]) -> list[ParsedPrediction]:
    return [parse_prediction(row) for row in rows]


def extract_json_object(text: str) -> str | None:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    fence_match = JSON_FENCE_RE.search(stripped)
    if fence_match:
        candidate = fence_match.group(1).strip()
        if candidate.startswith("{") and candidate.endswith("}"):
            return candidate
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        return stripped[start : end + 1]
    return None


def _from_obj(obj: dict[str, Any], valid_json: bool) -> ParsedPrediction:
    decision = _norm_decision(
        obj.get("final_decision")
        or obj.get("decision")
        or obj.get("predicted_decision")
        or obj.get("gold_decision")
    )
    parse_error = None if decision in DECISIONS else "invalid final_decision"
    stage_states = _extract_stage_states(obj)
    legacy = isinstance(obj.get("stage_labels"), dict) and not any(
        isinstance(obj.get(key), dict) for key in ("stage_states", "stage_diagnosis", "stage_state")
    )
    return ParsedPrediction(
        sample_id=str(obj.get("id") or obj.get("sample_id") or ""),
        valid_json=valid_json,
        final_decision=decision,
        policy_id=str(obj.get("policy_id") or "unknown"),
        stage_states=stage_states,
        legacy_stage_labels=legacy,
        raw_json=obj,
        parse_error=parse_error,
    )


def _invalid(error: str) -> ParsedPrediction:
    return ParsedPrediction(
        sample_id="",
        valid_json=False,
        final_decision="review",
        policy_id="unknown",
        stage_states=StageStateBundle().normalized(),
        legacy_stage_labels=False,
        raw_json=None,
        parse_error=error,
    )


def _looks_like_prediction_obj(obj: dict[str, Any]) -> bool:
    keys = {
        "final_decision",
        "decision",
        "predicted_decision",
        "stage_states",
        "stage_diagnosis",
        "stage_labels",
    }
    return bool(keys & set(obj))


def stage_errors(prediction: ParsedPrediction) -> list[str]:
    errors: list[str] = []
    states = prediction.stage_states.normalized().states
    for stage in STAGES:
        state = states[stage]
        if state.R not in (0, 1, None):
            errors.append(f"{stage}.R")
        if state.E not in ("absent", "present", "uncertain", None):
            errors.append(f"{stage}.E")
        if state.F not in (0, 1, None):
            errors.append(f"{stage}.F")
    return errors

