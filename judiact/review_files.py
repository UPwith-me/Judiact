"""Resolve supervision inputs without depending on a server or snapshot date."""


LEGACY_PATTERNS = {
    "supervision.jsonl": "supervision_metadata_*.jsonl",
    "projection.jsonl": "train_projection_metadata_*.jsonl",
    "conditions.jsonl": "policy_*_adjudications_*.jsonl",
    "historical_conditions.jsonl": "history/*/action_semantics_*.jsonl",
}


def review_file(release, name):
    directory = release / "reviews"
    current = directory / name
    if current.is_file():
        return current
    matches = sorted(directory.glob(LEGACY_PATTERNS[name]))
    if len(matches) > 1:
        raise ValueError(f"Ambiguous review input; use the canonical path {current}")
    return matches[0] if matches else current
