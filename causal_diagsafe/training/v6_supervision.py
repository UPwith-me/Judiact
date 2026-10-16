"""Reference-measure weights for the actual risk/action supervision, not thresholds."""

from collections import defaultdict

from .v4_losses import MemberLosses


BALANCED_FIELDS = frozenset(("action", "F1", "F2", "F3", "F4", "F5"))


def risk_action_balance(data, determinate_only=False):
    """Equal observed-class mass within each task/field, retaining its total budget.

    Count the coefficients actually used by absolute_loss, including missing
    evidence fields and same-root member weights. Dynamic policy IDs, native
    labels and evidence tokens are not class-balanced.
    """
    masses = defaultdict(lambda: defaultdict(float))
    for sid in data.records:
        case = data.cases[sid]
        targets = case["targets"]
        diagnostics = len(targets) - 1
        for field in BALANCED_FIELDS & targets.keys():
            weight = data.absolute_weights[sid]
            if field != "action":
                weight /= diagnostics
            masses[case["task_type"] + "/" + field][str(targets[field])] += weight
    report = {}
    for key, counts in masses.items():
        total = sum(counts.values())
        active = {label: mass for label, mass in counts.items()
                  if not (determinate_only and key.rsplit("/", 1)[-1] != "action" and label == "uncertain")}
        active_mass = sum(active.values())
        report[key] = {"observed_class_mass": dict(counts), "total_mass": total,
                       "prior": {label: mass / total for label, mass in counts.items()},
                       "balanced_labels": sorted(active),
                       "loss_multiplier": {label: active_mass / (len(active) * mass) if label in active else 1.
                                           for label, mass in counts.items()}}
    return report


def balanced_members(members, cases, balance):
    result = {}
    for sid, member in members.items():
        if member.native is not None:
            result[sid] = member
            continue
        case = cases[sid]

        def scale(field, value):
            if field not in BALANCED_FIELDS:
                return value
            key = case["task_type"] + "/" + field
            return value * balance[key]["loss_multiplier"][str(case["targets"][field])]

        result[sid] = MemberLosses(scale("action", member.action),
            {field: scale(field, value) for field, value in member.diagnostic.items()}, None)
    return result
