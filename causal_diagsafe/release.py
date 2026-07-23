from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from .io_utils import read_jsonl
from .preflight import preflight_rows
from .schema import normalize_record


MODEL_INPUT_FIELDS = frozenset(
    {
        "query",
        "images",
        "response_candidate",
        "policy_text",
    }
)
TARGET_CONSISTENCY_FIELDS = (
    "final_decision",
    "target_stage",
    "role",
    "changed_factor",
    "policy_id",
)
MANUAL_EVIDENCE_FIELDS = (
    "manual_visual_evidence",
    "manual_binding_evidence",
    "manual_text_evidence",
)


def inspect_release(config: dict[str, Any], *, check_images: bool = True) -> dict[str, Any]:
    """Inspect configured release files without materializing images or changing data."""

    releases = config.get("releases")
    if not isinstance(releases, dict) or not releases:
        raise ValueError("config.releases must be a non-empty object")

    result: dict[str, Any] = {
        "format": "causal_diagsafe_release_inspection_v1",
        "schema_version": config.get("schema_version", "unknown"),
        "releases": {},
        "blocking_issues": [],
        "warnings": [],
    }
    for name, spec in releases.items():
        if not isinstance(spec, dict):
            result["blocking_issues"].append(f"{name}: release spec must be an object")
            continue
        release_summary = _inspect_one_release(name, spec, check_images=check_images)
        result["releases"][name] = release_summary
        result["blocking_issues"].extend(release_summary.pop("blocking_issues"))
        result["warnings"].extend(release_summary.pop("warnings"))

    result["blocking_issue_count"] = len(result["blocking_issues"])
    result["warning_count"] = len(result["warnings"])
    result["ready_for_training_code_integration"] = result["blocking_issue_count"] == 0
    return result


def _inspect_one_release(
    name: str,
    spec: dict[str, Any],
    *,
    check_images: bool,
) -> dict[str, Any]:
    blocking: list[str] = []
    warnings: list[str] = []
    input_format = str(spec.get("input_format") or "nested_model_input")
    if input_format not in {"nested_model_input", "top_level_query_images"}:
        blocking.append(f"{name}: unsupported input_format={input_format}")
    group_mapping = _load_group_mapping(spec.get("group_registry"), name, blocking)
    split_specs = spec.get("splits")
    if not isinstance(split_specs, dict):
        return {
            "splits": {},
            "blocking_issues": [f"{name}: splits must be an object"],
            "warnings": warnings,
        }

    split_summaries: dict[str, Any] = {}
    split_id_sets: dict[str, set[str]] = {}
    for split_name in ("train", "dev", "test"):
        split_spec = split_specs.get(split_name)
        if not isinstance(split_spec, dict):
            blocking.append(f"{name}/{split_name}: missing split configuration")
            continue
        summary, split_ids, split_blocking, split_warnings = _inspect_split(
            name,
            split_name,
            split_spec,
            group_mapping=group_mapping,
            check_images=check_images,
            input_format=input_format,
        )
        split_summaries[split_name] = summary
        split_id_sets[split_name] = split_ids
        blocking.extend(split_blocking)
        warnings.extend(split_warnings)

    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        overlap = split_id_sets.get(left, set()) & split_id_sets.get(right, set())
        if overlap:
            blocking.append(f"{name}: {left}/{right} id overlap={len(overlap)}")

    if split_summaries.get("dev", {}).get("row_count") == 0:
        warnings.append(f"{name}: dev split is empty")
    if split_summaries.get("test", {}).get("row_count") == 0:
        warnings.append(f"{name}: test split is empty")

    return {
        "splits": split_summaries,
        "group_registry_count": len(set(group_mapping.values())),
        "blocking_issues": blocking,
        "warnings": warnings,
    }


def _inspect_split(
    release_name: str,
    split_name: str,
    spec: dict[str, Any],
    *,
    group_mapping: dict[str, str],
    check_images: bool,
    input_format: str,
) -> tuple[dict[str, Any], set[str], list[str], list[str]]:
    label = f"{release_name}/{split_name}"
    blocking: list[str] = []
    warnings: list[str] = []
    input_path = Path(str(spec.get("input", "")))
    if not input_path.is_file():
        return (
            {"row_count": 0, "input": str(input_path)},
            set(),
            [f"{label}: missing input file {input_path}"],
            warnings,
        )

    rows = read_jsonl(input_path)
    ids, duplicate_ids = _collect_ids(rows)
    if duplicate_ids:
        blocking.append(f"{label}: duplicate ids={len(duplicate_ids)}")

    contract_issues = _validate_model_input_contract(rows, input_format=input_format)
    if contract_issues:
        blocking.extend(f"{label}: {issue}" for issue in contract_issues[:50])

    audit_rows: list[dict[str, Any]] = []
    audit_path_value = spec.get("audit")
    if audit_path_value:
        audit_path = Path(str(audit_path_value))
        if not audit_path.is_file():
            blocking.append(f"{label}: missing audit file {audit_path}")
        else:
            audit_rows = read_jsonl(audit_path)
            blocking.extend(_audit_alignment_issues(label, rows, audit_rows))

    group_mapped = sum(1 for sample_id in ids if sample_id in group_mapping)
    if group_mapping and group_mapped != len(ids):
        blocking.append(f"{label}: group registry missing {len(ids) - group_mapped} ids")

    image_paths = sorted(
        {
            image
            for row in rows
            for image in normalize_record(row).images
            if image
        }
    )
    missing_images: list[str] = []
    if check_images:
        missing_images = [path for path in image_paths if not Path(path).is_file()]
        if missing_images:
            blocking.append(f"{label}: missing image files={len(missing_images)}")

    evidence_coverage = _evidence_coverage(audit_rows)
    preflight = preflight_rows(rows)
    if rows and all(
        stage_counts["R"] == 0 and stage_counts["F"] == 0
        for stage_counts in preflight["coverage"]["stage_label_coverage"].values()
    ):
        warnings.append(
            f"{label}: no explicit R/F labels; only requirement/evidence may use the "
            "documented partial bridge and F must remain masked"
        )

    summary = {
        "input": str(input_path),
        "audit": str(audit_path_value or ""),
        "row_count": len(rows),
        "unique_id_count": len(ids),
        "duplicate_id_count": len(duplicate_ids),
        "input_contract_error_count": len(contract_issues),
        "audit_row_count": len(audit_rows),
        "group_registry_mapped": group_mapped,
        "unique_image_count": len(image_paths),
        "missing_image_count": len(missing_images),
        "missing_image_examples": missing_images[:20],
        "manual_evidence_coverage": evidence_coverage,
        "distributions": preflight["distributions"],
        "stage_label_coverage": preflight["coverage"]["stage_label_coverage"],
    }
    return summary, ids, blocking, warnings


def _load_group_mapping(
    path_value: Any,
    release_name: str,
    blocking: list[str],
) -> dict[str, str]:
    if not path_value:
        return {}
    path = Path(str(path_value))
    if not path.is_file():
        blocking.append(f"{release_name}: missing group registry {path}")
        return {}
    mapping: dict[str, str] = {}
    for row in read_jsonl(path):
        group_id = str(row.get("group_id") or "")
        candidate_ids = row.get("candidate_ids")
        if not group_id or not isinstance(candidate_ids, list):
            blocking.append(f"{release_name}: malformed group registry row")
            continue
        for candidate_id in candidate_ids:
            sample_id = str(candidate_id)
            old = mapping.get(sample_id)
            if old is not None and old != group_id:
                blocking.append(f"{release_name}: id {sample_id} assigned to two groups")
            mapping[sample_id] = group_id
    return mapping


def _collect_ids(rows: list[dict[str, Any]]) -> tuple[set[str], set[str]]:
    counts = Counter(_row_id(row) for row in rows)
    counts.pop("", None)
    return set(counts), {sample_id for sample_id, count in counts.items() if count > 1}


def _row_id(row: dict[str, Any]) -> str:
    return str(row.get("id") or row.get("sample_id") or row.get("candidate_id") or "")


def _validate_model_input_contract(
    rows: list[dict[str, Any]],
    *,
    input_format: str,
) -> list[str]:
    issues: list[str] = []
    for index, row in enumerate(rows, 1):
        sample_id = _row_id(row) or f"line:{index}"
        model_input = row.get("model_input")
        if input_format == "top_level_query_images":
            query = row.get("query")
            images = row.get("images")
            if not isinstance(query, str) or not query.strip():
                issues.append(f"{sample_id}: query missing")
            if not isinstance(images, list) or any(not isinstance(item, str) for item in images):
                issues.append(f"{sample_id}: images must be list[str]")
            if isinstance(model_input, dict):
                if model_input.get("query") != query or model_input.get("images") != images:
                    issues.append(f"{sample_id}: nested/top-level model input mismatch")
            continue
        if not isinstance(model_input, dict):
            issues.append(f"{sample_id}: model_input missing or invalid")
            continue
        unexpected = set(model_input) - MODEL_INPUT_FIELDS
        if unexpected:
            issues.append(f"{sample_id}: unexpected model_input fields={sorted(unexpected)}")
        if not isinstance(model_input.get("query"), str) or not model_input.get("query", "").strip():
            issues.append(f"{sample_id}: model_input.query missing")
        images = model_input.get("images")
        if not isinstance(images, list) or any(not isinstance(item, str) for item in images):
            issues.append(f"{sample_id}: model_input.images must be list[str]")
        for optional_text in ("response_candidate", "policy_text"):
            if optional_text in model_input and not isinstance(model_input[optional_text], str):
                issues.append(f"{sample_id}: model_input.{optional_text} must be str")
        targets: dict[str, Any] = (
            row["targets"] if isinstance(row.get("targets"), dict) else {}
        )
        changed_factor = str(targets.get("changed_factor") or row.get("changed_factor") or "")
        if changed_factor == "policy_rule" and "policy_text" not in model_input:
            issues.append(f"{sample_id}: policy_rule intervention requires model_input.policy_text")
        if changed_factor == "response_execution" and not str(
            model_input.get("response_candidate") or ""
        ).strip():
            issues.append(
                f"{sample_id}: response_execution intervention requires "
                "model_input.response_candidate"
            )
    return issues


def _audit_alignment_issues(
    label: str,
    input_rows: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
) -> list[str]:
    issues: list[str] = []
    audit_by_id = {_row_id(row): row for row in audit_rows if _row_id(row)}
    input_ids, _ = _collect_ids(input_rows)
    audit_ids, audit_duplicates = _collect_ids(audit_rows)
    if audit_duplicates:
        issues.append(f"{label}: duplicate audit ids={len(audit_duplicates)}")
    if input_ids != audit_ids:
        issues.append(
            f"{label}: input/audit id mismatch input_only={len(input_ids-audit_ids)} "
            f"audit_only={len(audit_ids-input_ids)}"
        )

    mismatch_counts: Counter[str] = Counter()
    for row in input_rows:
        sample_id = _row_id(row)
        audit = audit_by_id.get(sample_id)
        if audit is None:
            continue
        normalized = normalize_record(row)
        normalized_audit = normalize_record(audit)
        for field in TARGET_CONSISTENCY_FIELDS:
            if getattr(normalized, field) != getattr(normalized_audit, field):
                mismatch_counts[field] += 1
        if normalized.query != normalized_audit.query:
            mismatch_counts["query"] += 1
        if normalized.images != normalized_audit.images:
            mismatch_counts["images"] += 1
        if normalized.response_candidate != normalized_audit.response_candidate:
            mismatch_counts["response_candidate"] += 1
        if normalized.policy_text != normalized_audit.policy_text:
            mismatch_counts["policy_text"] += 1
        if normalized.policy_text_supplied != normalized_audit.policy_text_supplied:
            mismatch_counts["policy_text_supplied"] += 1
    for field, count in mismatch_counts.items():
        issues.append(f"{label}: input/audit {field} mismatch={count}")
    return issues


def _evidence_coverage(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        field: sum(1 for row in rows if _has_meaningful_text(row.get(field)))
        for field in MANUAL_EVIDENCE_FIELDS
    }


def _has_meaningful_text(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return bool(text and text not in {"none", "n/a", "na", "null", "unknown"})


def render_release_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# CausalDiagSafe-VL 数据发布核验",
        "",
        f"- schema_version: {summary.get('schema_version', 'unknown')}",
        f"- blocking_issue_count: {summary.get('blocking_issue_count', 0)}",
        f"- warning_count: {summary.get('warning_count', 0)}",
        f"- ready_for_training_code_integration: {summary.get('ready_for_training_code_integration', False)}",
        "",
    ]
    for release_name, release in summary.get("releases", {}).items():
        lines.extend([f"## {release_name}", ""])
        for split_name, split in release.get("splits", {}).items():
            lines.extend(
                [
                    f"### {split_name}",
                    f"- rows: {split.get('row_count', 0)}",
                    f"- audit_rows: {split.get('audit_row_count', 0)}",
                    f"- unique_images: {split.get('unique_image_count', 0)}",
                    f"- missing_images: {split.get('missing_image_count', 0)}",
                    f"- group_registry_mapped: {split.get('group_registry_mapped', 0)}",
                    f"- input_contract_errors: {split.get('input_contract_error_count', 0)}",
                    "",
                ]
            )
    lines.extend(["## 阻断问题", ""])
    lines.extend(f"- {item}" for item in summary.get("blocking_issues", []))
    if not summary.get("blocking_issues"):
        lines.append("- 无")
    lines.extend(["", "## 警告", ""])
    lines.extend(f"- {item}" for item in summary.get("warnings", []))
    if not summary.get("warnings"):
        lines.append("- 无")
    return "\n".join(lines) + "\n"
