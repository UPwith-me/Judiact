"""Masked supervision and group-normalized input-intervention losses."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def categorical_js(a, b):
    p, q = a.float().softmax(-1), b.float().softmax(-1)
    m = (p + q) * 0.5
    tiny = torch.finfo(p.dtype).tiny
    log_m = m.clamp_min(tiny).log()
    return 0.5 * (p * (p.clamp_min(tiny).log() - log_m)
                  + q * (q.clamp_min(tiny).log() - log_m)).sum(-1)


def masked_ce(logits, labels, zero):
    valid = labels != -100
    return F.cross_entropy(logits[valid], labels[valid]) if valid.any() else zero


def evidence_loss(weights, targets, row_indices, zero):
    rows = []
    for row in row_indices:
        items = []
        for target in targets:
            if target["row"] == row:
                mass = weights[row, target["stage"], target["field"], target["positions"]].sum()
                items.append(-mass.clamp_min(torch.finfo(torch.float32).tiny).log())
        if items:
            rows.append(torch.stack(items).mean())
    return torch.stack(rows).mean() if rows else zero


def supervised_loss(output, labels, evidence, indices, kind):
    zero = output["stage"]["F"][indices].sum() * 0
    if kind == "auxiliary":
        parts = []
        for task, logits in output["native"].items():
            selected = [i for i in indices if labels["native_tasks"][i] == task]
            if selected:
                parts.append(F.cross_entropy(logits[selected], labels["native_target"][selected],
                                             reduction="none"))
        return torch.cat(parts).mean()
    if kind == "response":
        pub = masked_ce(output["response_publishability"][indices],
                        labels["response_publishability"][indices], zero)
        state = torch.stack([masked_ce(output["stage"][v][indices, 4],
                                        labels[v][indices, 4], zero) for v in ("E", "F")]).mean()
        return pub + state + 0.5 * evidence_loss(output["evidence_weights"], evidence, indices, zero)
    stage_losses = []
    for k in range(4):
        variables = [masked_ce(output["stage"][v][indices, k], labels[v][indices, k], zero)
                     for v in ("R", "E", "F")]
        stage_losses.append(torch.stack(variables).mean())
    return (masked_ce(output["request_action"][indices], labels["request_action"][indices], zero)
            + torch.stack(stage_losses).mean()
            + masked_ce(output["policy_joint"][indices], labels["policy_joint"][indices], zero)
            + masked_ce(output["intent"][indices], labels["intent"][indices], zero)
            + 0.5 * evidence_loss(output["evidence_weights"], evidence, indices, zero)
            + 0.25 * masked_ce(output["safe_response_behavior"][indices],
                               labels["safe_response_behavior"][indices], zero))


def group_loss(output, labels, groups):
    zero = output["stage"]["F"].sum() * 0
    risk_score = output["stage"]["F"][..., 1] - output["stage"]["F"][..., 0]
    group_losses = []
    for group in groups:
        items = []
        for edge in group["edges"]:
            a, b, k = edge["left"], edge["right"], edge["stage"]
            terms = []
            if edge["target_flip_direction"]:
                terms.append(F.softplus(-edge["target_flip_direction"] *
                                        (risk_score[b, k] - risk_score[a, k])))
            if edge["stable_target_F"]:
                terms.append(categorical_js(output["stage"]["F"][a, k], output["stage"]["F"][b, k]))
            for stage in edge["stable_non_target_stages"]:
                index = int(stage[1:]) - 1
                for variable in ("R", "E", "F"):
                    left, right = labels[variable][a, index], labels[variable][b, index]
                    if left != -100 and left == right:
                        terms.append(categorical_js(output["stage"][variable][a, index],
                                                    output["stage"][variable][b, index]))
            key = "response_publishability" if k == 4 else "request_action"
            if edge["stable_decision"]:
                terms.append(categorical_js(output[key][a], output[key][b]))
            if edge["decision_contrast"]:
                y_a, y_b = edge["decision_targets"]
                log_a, log_b = output[key][a].log_softmax(-1), output[key][b].log_softmax(-1)
                contrast = log_a[y_a] - log_a[y_b] + log_b[y_b] - log_b[y_a]
                terms.append(F.softplus(-contrast))
            if edge["policy_alignment"]:
                pairs = edge["policy_alignment"]
                terms.append(categorical_js(output["policy_joint"][a, [p[0] for p in pairs]],
                                            output["policy_joint"][b, [p[1] for p in pairs]]))
            if terms:
                items.append(torch.stack(terms).mean())
        quartet = group.get("quartet")
        if quartet and quartet["use_interaction_loss"]:
            a, b, c, d = quartet["rows"]
            delta = risk_score[d, 1] - risk_score[c, 1] - risk_score[b, 1] + risk_score[a, 1]
            items.append(F.softplus(-quartet["interaction_direction"] * delta))
        if items:
            group_losses.append(torch.stack(items).mean())
    return torch.stack(group_losses).mean() if group_losses else zero


def training_loss(output, batch, use_relations=True):
    units = []
    for unit in batch["units"]:
        loss = supervised_loss(output, batch["labels"], batch["evidence"],
                               unit["rows"], unit["kind"])
        if use_relations and unit["groups"]:
            loss = loss + 0.5 * group_loss(output, batch["labels"], unit["groups"])
        units.append(loss)
    return torch.stack(units).mean()


def final_only_loss(output, batch):
    zero = output["response_publishability"].sum() * 0
    units = []
    for unit in batch["units"]:
        indices, kind = unit["rows"], unit["kind"]
        labels = batch["labels"]
        if kind == "auxiliary":
            terms = []
            for task, logits in output["native"].items():
                selected = [i for i in indices if labels["native_tasks"][i] == task]
                if selected:
                    terms.append(F.cross_entropy(logits[selected], labels["native_target"][selected],
                                                 reduction="none"))
            units.append(torch.cat(terms).mean())
        else:
            key = "response_publishability" if kind == "response" else "request_action"
            units.append(masked_ce(output[key][indices], labels[key][indices], zero))
    return torch.stack(units).mean()


def natural_nll(output, labels):
    zero = output["response_publishability"].sum() * 0
    values = []
    for key in ("request_action", "response_publishability"):
        valid = labels[key] != -100
        if valid.any():
            values.extend(F.cross_entropy(output[key][valid], labels[key][valid], reduction="none").unbind())
    return torch.stack(values).mean() if values else zero
