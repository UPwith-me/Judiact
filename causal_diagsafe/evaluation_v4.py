"""Existing action/localization metrics, with v4's actual continuous-audit semantics."""

from .evaluation_v3 import summarize as summarize_v3


def summarize(samples, predictions, natural_ids, roots, edges, quartets):
    report, groups = summarize_v3(samples, predictions, natural_ids, roots, edges, quartets)
    report["method"] = "contextual_v4; one generated audit then its conditional action"
    for part in report["subsets"].values():
        if "evidence" in part:
            part["evidence"]["decoder"] = "continuous greedy audit; exact quote/occurrence decoding; original-image 0-1000 grounding"
    report["probability_semantics"] = "conditional on the model-generated audit; not a gold prefix or a marginal over all possible audits"
    return report, groups
