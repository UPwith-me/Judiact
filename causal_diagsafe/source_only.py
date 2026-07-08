from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from .constants import STAGES
from .metrics import decision_metrics, stage_collapse_metrics
from .parser import ParsedPrediction
from .schema import DiagRecord, StageState, StageStateBundle, normalize_record


DEFAULT_METADATA_FIELDS: tuple[str, ...] = (
    "source_dataset",
    "modality_scope",
    "harm_category",
    "attack_mechanism",
    "target_stage",
    "policy_id",
)


class MetadataMajorityBaseline:
    """Leakage/control baseline that predicts from metadata only."""

    def __init__(self, fields: tuple[str, ...] = DEFAULT_METADATA_FIELDS) -> None:
        self.fields = fields
        self.decision_by_key: dict[tuple[str, ...], str] = {}
        self.f_by_key: dict[tuple[str, ...], tuple[int | None, ...]] = {}
        self.fallback_decision = "review"
        self.fallback_f: tuple[int | None, ...] = (None, None, None, None, None)

    def fit(self, records: list[DiagRecord]) -> "MetadataMajorityBaseline":
        decision_counter = Counter(record.final_decision for record in records)
        self.fallback_decision = decision_counter.most_common(1)[0][0] if decision_counter else "review"
        fallback_f = []
        for idx, stage in enumerate(STAGES):
            counts = Counter(record.stage_states.f_vector()[idx] for record in records)
            counts.pop(None, None)
            fallback_f.append(counts.most_common(1)[0][0] if counts else None)
        self.fallback_f = tuple(fallback_f)

        grouped_decisions: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
        grouped_f: dict[tuple[str, ...], list[Counter[int | None]]] = defaultdict(
            lambda: [Counter() for _ in STAGES]
        )
        for record in records:
            key = self._key(record)
            grouped_decisions[key][record.final_decision] += 1
            for idx, value in enumerate(record.stage_states.f_vector()):
                grouped_f[key][idx][value] += 1
        self.decision_by_key = {
            key: counts.most_common(1)[0][0] for key, counts in grouped_decisions.items()
        }
        self.f_by_key = {}
        for key, counters in grouped_f.items():
            values = []
            for idx, counter in enumerate(counters):
                counter.pop(None, None)
                values.append(counter.most_common(1)[0][0] if counter else self.fallback_f[idx])
            self.f_by_key[key] = tuple(values)
        return self

    def predict(self, records: list[DiagRecord]) -> list[ParsedPrediction]:
        predictions: list[ParsedPrediction] = []
        for record in records:
            key = self._key(record)
            decision = self.decision_by_key.get(key, self.fallback_decision)
            f_vector = self.f_by_key.get(key, self.fallback_f)
            states = StageStateBundle(
                {
                    stage: StageState(R=None, E="uncertain", F=f_vector[idx])
                    for idx, stage in enumerate(STAGES)
                }
            ).normalized()
            predictions.append(
                ParsedPrediction(
                    sample_id=record.sample_id,
                    valid_json=True,
                    final_decision=decision,
                    policy_id="metadata_only",
                    stage_states=states,
                    legacy_stage_labels=False,
                    raw_json=None,
                    parse_error=None,
                )
            )
        return predictions

    def _key(self, record: DiagRecord) -> tuple[str, ...]:
        return tuple(str(getattr(record, field, "unknown") or "unknown") for field in self.fields)


def run_metadata_baseline(
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    *,
    fields: tuple[str, ...] = DEFAULT_METADATA_FIELDS,
) -> dict[str, Any]:
    train_records = [normalize_record(row) for row in train_rows]
    eval_records = [normalize_record(row) for row in eval_rows]
    model = MetadataMajorityBaseline(fields=fields).fit(train_records)
    predictions = model.predict(eval_records)
    return {
        "format": "causal_diagsafe_metadata_baseline_v0.1",
        "fields": list(fields),
        "train_count": len(train_records),
        "eval_count": len(eval_records),
        "decision_metrics": decision_metrics(eval_records, predictions),
        "stage_collapse_metrics": stage_collapse_metrics(predictions),
        "interpretation": (
            "If this metadata-only baseline approaches CSSB/EGCR performance, the method is likely "
            "using source/modality/policy shortcuts rather than grounded causal stage states."
        ),
    }

