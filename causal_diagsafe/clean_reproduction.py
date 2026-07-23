from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

from .algorithm_data import AlgorithmExample


CLEAN_REPRO_VERSION = "e20_clean_reproduction_v3_model_visible_interventions"

POLICY_DECK = """Policy deck:
- P_BENIGN: benign general, educational, historical, creative, or safety-oriented content.
- P_GENERAL_SAFETY: sensitive but non-operational content requiring cautious, safety-oriented help.
- P_HATE_TOXICITY: hate, harassment, dehumanization, targeted abuse, or toxic persuasion.
- P_VIOLENCE: actionable violent harm, weapons, physical abuse, or injury facilitation.
- P_CYBER: malware, credential theft, exploits, evasion, unauthorized access, or cyber abuse.
- P_PRIVACY: privacy invasion, covert tracking, doxxing, credential exposure, or surveillance.
- P_FINANCIAL_CRIME: fraud, scams, money laundering, theft, or financial crime.
- P_DRUGS: illegal drug procurement, synthesis, trafficking, or misuse instructions.
- P_SEXUAL: sexual exploitation, coercive sexual content, or unsafe sexual requests.
- P_MISINFO: harmful deception, impersonation, or high-impact misinformation.
- P_MANIPULATION: coercion, blackmail, manipulation, or abusive persuasion.
- P_PUBLIC_SAFETY: unsafe high-impact medical, legal, public-safety, or emergency content.
- P_OTHER_POLICY: reviewed policy boundary that does not fit a narrower category.
- P_OTHER: ambiguous or uncategorized policy risk."""

POLICY_DECK_IDS = frozenset(
    {
        "P_BENIGN",
        "P_GENERAL_SAFETY",
        "P_HATE_TOXICITY",
        "P_VIOLENCE",
        "P_CYBER",
        "P_PRIVACY",
        "P_FINANCIAL_CRIME",
        "P_DRUGS",
        "P_SEXUAL",
        "P_MISINFO",
        "P_MANIPULATION",
        "P_PUBLIC_SAFETY",
        "P_OTHER_POLICY",
        "P_OTHER",
        "no_match",
        "insufficient_policy",
        "unknown",
    }
)

SYSTEM_PROMPT_PREFIX = """You are a multimodal safety decision assistant. Evaluate the actual user
query, every provided image, and any candidate response together. Apply only the policy text supplied
in this prompt. Do not infer safety from the dataset source, file name, or output format. Return
exactly one compact JSON object with the keys final_decision, policy_id, safe_response_behavior, and
safe_response_policy. final_decision must be one of allow, warn, review, or refuse. Keep
safe_response_policy brief and policy-safe; never provide operational harmful details."""

SYSTEM_PROMPT = f"{SYSTEM_PROMPT_PREFIX}\n\n{POLICY_DECK}"

DECISION_TO_BEHAVIOR = {
    "allow": "answer_normally",
    "warn": "answer_with_caution_and_correction",
    "review": "ask_clarification_or_defer",
    "refuse": "refuse_with_safe_alternative",
}


@dataclass(frozen=True)
class CleanReproductionBundle:
    joint_sft: tuple[dict[str, Any], ...]
    safety_teacher: tuple[dict[str, Any], ...]
    helpful_teacher: tuple[dict[str, Any], ...]
    text_only_exact: tuple[dict[str, Any], ...]
    calibration_rows: tuple[dict[str, Any], ...]
    calibration_groups: tuple[dict[str, Any], ...]
    report: dict[str, Any]


def build_clean_reproduction_bundle(
    examples: Iterable[AlgorithmExample],
    *,
    max_calibration_groups: int = 512,
    seed: int = 83,
) -> CleanReproductionBundle:
    source_examples = list(examples)
    joint_rows = [build_clean_sft_row(example) for example in source_examples]
    safety_rows = [
        build_clean_sft_row(example, dataset_role="safety_teacher")
        for example in source_examples
        if example.role == "unsafe" and example.final_decision in {"refuse", "review"}
    ]
    helpful_rows = [
        build_clean_sft_row(example, dataset_role="helpful_teacher")
        for example in source_examples
        if example.role in {"safe", "helpful"}
        and example.final_decision in {"allow", "warn"}
    ]
    text_only_rows = [
        build_clean_sft_row(example, dataset_role="e20_clean_exact_textonly")
        for example in source_examples
        if not example.images
    ]
    calibration_rows, calibration_groups = build_legacy_calibration_triplets(
        source_examples,
        max_groups=max_calibration_groups,
        seed=seed,
    )

    report = {
        "format": "e20_clean_reproduction_report_v1",
        "version": CLEAN_REPRO_VERSION,
        "seed": seed,
        "max_calibration_groups": max_calibration_groups,
        "source_count": len(source_examples),
        "joint_sft": _rows_summary(joint_rows),
        "safety_teacher": _rows_summary(safety_rows),
        "helpful_teacher": _rows_summary(helpful_rows),
        "text_only_exact": _rows_summary(text_only_rows),
        "calibration": {
            **_rows_summary(calibration_rows),
            "group_count": len(calibration_groups),
            "domain_key": dict(Counter(group["domain_key"] for group in calibration_groups)),
        },
        "prompt_contract": {
            "visible_fields": [
                "model_input.query",
                "model_input.images",
                "model_input.response_candidate when supplied",
                "model_input.policy_text when supplied, otherwise fixed_policy_deck",
            ],
            "forbidden_fields": [
                "role",
                "target_stage",
                "changed_factor",
                "source_dataset",
                "manual evidence",
                "label provenance",
            ],
            "target_fields": [
                "final_decision",
                "policy_id",
                "safe_response_behavior",
                "safe_response_policy",
            ],
        },
        "claim_boundary": (
            "The calibration triplets reproduce E20's role-balanced training control. Members are "
            "independently sampled and are not causal counterfactuals. They must never be scored as "
            "stage-intervention groups."
        ),
    }
    return CleanReproductionBundle(
        joint_sft=tuple(joint_rows),
        safety_teacher=tuple(safety_rows),
        helpful_teacher=tuple(helpful_rows),
        text_only_exact=tuple(text_only_rows),
        calibration_rows=tuple(calibration_rows),
        calibration_groups=tuple(calibration_groups),
        report=report,
    )


def build_clean_sft_row(
    example: AlgorithmExample,
    *,
    dataset_role: str = "joint_sft",
) -> dict[str, Any]:
    model_input: dict[str, Any] = {
        "query": example.query,
        "images": list(example.images),
    }
    if example.response_candidate:
        model_input["response_candidate"] = example.response_candidate
    if example.policy_text_supplied:
        model_input["policy_text"] = example.policy_text
    user_text = _visible_user_text(model_input)
    system_prompt = policy_system_prompt(
        example.policy_text,
        supplied=example.policy_text_supplied,
    )
    target = {
        "final_decision": example.final_decision,
        "policy_id": example.policy_id,
        "safe_response_behavior": DECISION_TO_BEHAVIOR[example.final_decision],
        "safe_response_policy": _safe_response_policy(example),
    }
    return {
        "id": example.sample_id,
        "split": example.split,
        "model_input": model_input,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
            {
                "role": "assistant",
                "content": json.dumps(target, ensure_ascii=False, separators=(",", ":")),
            },
        ],
        "images": list(example.images),
        "training_target": target,
        "training_control": {
            "dataset_role": dataset_role,
            "role": example.role,
            "source_role": example.source_role,
            "final_decision": example.final_decision,
            "target_stage": example.target_stage,
            "changed_factor": example.changed_factor,
            "modality_scope": example.modality_scope,
            "source_dataset": example.source_dataset,
            "group_id": example.group_id,
            "label_source": list(example.stage.source),
        },
        "provenance": {
            "clean_reproduction_version": CLEAN_REPRO_VERSION,
            "source_id": example.sample_id,
            "split": example.split,
        },
    }


def policy_system_prompt(policy_text: str, *, supplied: bool) -> str:
    if not supplied:
        return SYSTEM_PROMPT
    visible_policy = policy_text.strip() or "Policy deck unavailable. Return review when policy is required."
    return f"{SYSTEM_PROMPT_PREFIX}\n\n{visible_policy}"


def _visible_user_text(model_input: dict[str, Any]) -> str:
    parts = [f"User query:\n{str(model_input.get('query') or '').strip()}"]
    if "response_candidate" in model_input:
        parts.append(
            "Candidate response to audit:\n"
            + str(model_input.get("response_candidate") or "").strip()
        )
    return "\n\n".join(parts)


def build_legacy_calibration_triplets(
    examples: Iterable[AlgorithmExample],
    *,
    max_groups: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    source = list(examples)
    harmful = [
        example
        for example in source
        if example.role == "unsafe" and example.final_decision == "refuse"
    ]
    safe = [
        example
        for example in source
        if example.role == "safe" and example.final_decision in {"allow", "warn"}
    ]
    helpful = [
        example
        for example in source
        if example.role == "helpful" and example.final_decision in {"allow", "warn"}
    ]
    rng.shuffle(harmful)
    rng.shuffle(safe)
    rng.shuffle(helpful)

    safe_buckets = _bucket_examples(safe)
    helpful_buckets = _bucket_examples(helpful)
    fallback_safe = list(safe)
    fallback_helpful = list(helpful)
    used_safe: set[str] = set()
    used_helpful: set[str] = set()
    rows: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []

    for harmful_example in harmful:
        if len(groups) >= max(0, max_groups):
            break
        key = _domain_key(harmful_example)
        safe_example = _pop_bucket(safe_buckets, key, used_safe)
        if safe_example is None:
            safe_example = _pop_unused(fallback_safe, used_safe)
        helpful_example = _pop_bucket(helpful_buckets, key, used_helpful)
        if helpful_example is None:
            helpful_example = _pop_unused(fallback_helpful, used_helpful)
        if safe_example is None or helpful_example is None:
            break

        group_id = f"clean_calibration_{len(groups) + 1:06d}"
        members = (
            ("unsafe", harmful_example),
            ("safe", safe_example),
            ("helpful", helpful_example),
        )
        member_ids: list[str] = []
        for role, example in members:
            row = build_clean_sft_row(example, dataset_role="e20_clean_calibration")
            row["training_control"].update(
                {
                    "calibration_triplet_id": group_id,
                    "calibration_role": role,
                    "calibration_domain_key": key,
                    "causal_group": False,
                    **_legacy_role_multipliers(role),
                }
            )
            rows.append(row)
            member_ids.append(example.sample_id)
        groups.append(
            {
                "group_id": group_id,
                "method": "clean_fsa_calibration_triplets_v1",
                "domain_key": key,
                "member_ids": member_ids,
                "member_roles": ["unsafe", "safe", "helpful"],
                "causal_group": False,
                "claim_boundary": (
                    "Independent role-balanced calibration samples; no target-stage or "
                    "counterfactual relation is asserted."
                ),
            }
        )
    return rows, groups


def validate_clean_rows(rows: Iterable[dict[str, Any]]) -> list[str]:
    issues: list[str] = []
    seen_ids: Counter[str] = Counter()
    for index, row in enumerate(rows):
        row_id = str(row.get("id") or "")
        seen_ids[row_id] += 1
        model_input = row.get("model_input")
        messages = row.get("messages")
        target = row.get("training_target")
        if not row_id:
            issues.append(f"row[{index}]:missing_id")
        if not isinstance(model_input, dict) or not str(model_input.get("query") or "").strip():
            issues.append(f"{row_id or index}:missing_query")
        if not isinstance(messages, list) or len(messages) != 3:
            issues.append(f"{row_id or index}:invalid_messages")
            continue
        if any(not isinstance(message, dict) for message in messages):
            issues.append(f"{row_id or index}:invalid_message_object")
            continue
        if not isinstance(target, dict):
            issues.append(f"{row_id or index}:missing_target")
            continue
        if target.get("final_decision") not in DECISION_TO_BEHAVIOR:
            issues.append(f"{row_id or index}:invalid_decision")
        policy_id = str(target.get("policy_id") or "")
        if not policy_id:
            issues.append(f"{row_id or index}:missing_policy_id")
        elif policy_id not in POLICY_DECK_IDS:
            issues.append(f"{row_id or index}:policy_id_not_in_deck:{policy_id}")
        control = row.get("training_control")
        if not isinstance(control, dict):
            issues.append(f"{row_id or index}:missing_training_control")
        prompt = " ".join(
            str(message.get("content") or "") for message in messages[:2]
        )
        for forbidden_value in (
            (control or {}).get("source_dataset") if isinstance(control, dict) else None,
            (control or {}).get("group_id") if isinstance(control, dict) else None,
        ):
            if forbidden_value and str(forbidden_value) in prompt:
                issues.append(f"{row_id or index}:control_value_leaked_into_prompt")
    duplicate_ids = sorted(key for key, count in seen_ids.items() if key and count > 1)
    if duplicate_ids:
        issues.append(f"duplicate_ids:{duplicate_ids[:20]}")
    return issues


def _safe_response_policy(example: AlgorithmExample) -> str:
    value = " ".join(example.gold_response_policy.split())
    if value:
        return value
    fallback = {
        "allow": "Answer the benign request directly and accurately.",
        "warn": "Correct unsafe framing and provide high-level, non-operational guidance.",
        "review": "Ask for clarifying context or defer when evidence or policy is insufficient.",
        "refuse": "Briefly refuse harmful assistance and offer a safe alternative.",
    }
    return fallback[example.final_decision]


def _legacy_role_multipliers(role: str) -> dict[str, Any]:
    base = {
        "format_repair_priority": "high",
        "repair_sft_multiplier": 1.35,
        "stage_lock_multiplier": 0.0,
    }
    if role == "unsafe":
        return {
            **base,
            "safety_anchor_sft_multiplier": 1.12,
            "safety_anchor_margin_multiplier": 1.28,
            "helpful_anchor_sft_multiplier": 1.0,
            "helpful_anchor_margin_multiplier": 1.0,
        }
    if role == "safe":
        return {
            **base,
            "safety_anchor_sft_multiplier": 1.0,
            "safety_anchor_margin_multiplier": 1.0,
            "helpful_anchor_sft_multiplier": 1.55,
            "helpful_anchor_margin_multiplier": 1.65,
        }
    return {
        **base,
        "safety_anchor_sft_multiplier": 1.0,
        "safety_anchor_margin_multiplier": 1.0,
        "helpful_anchor_sft_multiplier": 1.4,
        "helpful_anchor_margin_multiplier": 1.45,
    }


def _domain_key(example: AlgorithmExample) -> str:
    return f"{example.source_dataset}|{example.modality_scope}"


def _bucket_examples(examples: Iterable[AlgorithmExample]) -> dict[str, list[AlgorithmExample]]:
    buckets: dict[str, list[AlgorithmExample]] = defaultdict(list)
    for example in examples:
        buckets[_domain_key(example)].append(example)
    return buckets


def _pop_bucket(
    buckets: dict[str, list[AlgorithmExample]],
    key: str,
    used_ids: set[str],
) -> AlgorithmExample | None:
    return _pop_unused(buckets.get(key, []), used_ids)


def _pop_unused(
    examples: list[AlgorithmExample],
    used_ids: set[str],
) -> AlgorithmExample | None:
    while examples:
        example = examples.pop(0)
        if example.sample_id not in used_ids:
            used_ids.add(example.sample_id)
            return example
    return None


def _rows_summary(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    materialized = list(rows)
    return {
        "count": len(materialized),
        "decision": dict(
            Counter(row["training_target"]["final_decision"] for row in materialized)
        ),
        "role": dict(Counter(row["training_control"].get("role", "") for row in materialized)),
        "modality_scope": dict(
            Counter(row["training_control"].get("modality_scope", "") for row in materialized)
        ),
        "image_rows": sum(bool(row.get("images")) for row in materialized),
    }
