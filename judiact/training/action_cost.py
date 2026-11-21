"""Frozen full-population four-action costs with a hard-label anchor."""

import math

import torch

from ..data_contract import DECISIONS


TASKS = ("request_decision", "response_audit")
ALGORITHM = "gold_anchored_full_information_cost"
GAIN_WEIGHT = .5


def action_population(data):
    totals = {task: math.fsum(data.absolute_weights[sid] for sid in data.records
              if data.cases[sid]["task_type"] == task) for task in TASKS}
    return {sid: {"task": data.cases[sid]["task_type"],
                  "action": data.cases[sid].get("sft_targets", data.cases[sid]["targets"])["action"],
                  "mass": data.absolute_weights[sid] / totals[data.cases[sid]["task_type"]]}
            for sid in sorted(data.records)}


def cost_table(confusion):
    c = torch.as_tensor(confusion, dtype=torch.float64)
    assert c.shape == (4, 4) and torch.isfinite(c).all() and (c >= 0).all()
    m, s = c.sum(1), c.sum(0)
    assert (m > 0).all() and math.isclose(float(m.sum()), 1., abs_tol=1e-10)
    den = m + s
    gain = .5 * (torch.diag(1 / den) - (c.diag() / den.square())[None, :])
    value = gain - gain.min(1, keepdim=True).values
    row_sum = value.sum(1)
    normalizer = m @ row_sum
    hard = (1 - GAIN_WEIGHT) * torch.diag(1 / (4 * m))
    metric = GAIN_WEIGHT * value / normalizer
    return {"confusion": c.tolist(), "mass": m.tolist(), "predicted_mass": s.tolist(),
            "macro_F1": float((2 * c.diag() / den).mean()),
            "gain": gain.tolist(), "value": value.tolist(), "row_sum": row_sum.tolist(),
            "normalizer": float(normalizer), "hard_coefficients": hard.tolist(),
            "gain_coefficients": metric.tolist(),
            "source_class_budget": (m * (hard + metric).sum(1)).tolist(),
            "gold_class_budget": (m * (hard + metric).diag()).tolist(),
            "prediction_column_budget": (m[:, None] * (hard + metric)).sum(0).tolist()}


def make_cost_snapshot(population, predictions, source_adapter, release):
    assert population.keys() == predictions.keys(), "Costs require all and only current train members"
    tables = {}
    for task in TASKS:
        ids = [sid for sid, row in population.items() if row["task"] == task]
        y = torch.tensor([DECISIONS.index(population[sid]["action"]) for sid in ids])
        mass = torch.tensor([population[sid]["mass"] for sid in ids], dtype=torch.float64)
        logp = torch.tensor([predictions[sid] for sid in ids], dtype=torch.float64)
        assert logp.shape == (len(ids), 4) and torch.isfinite(logp).all()
        assert torch.allclose(logp.exp().sum(1), torch.ones(len(ids), dtype=torch.float64), atol=1e-6, rtol=0)
        pred = logp.argmax(1)
        c = torch.zeros(16, dtype=torch.float64).index_add_(0, 4 * y + pred, mass).reshape(4, 4)
        table = cost_table(c)
        table["members"] = len(ids)
        table["member_confusion"] = torch.bincount(4 * y + pred, minlength=16).reshape(4, 4).tolist()
        table["class_readout"] = {
            label: {"members": int((y == a).sum()), "correct_members": int(((y == a) & (pred == a)).sum()),
                    "root_weighted_nll": float(-(mass[y == a] * logp[y == a, a]).sum() / mass[y == a].sum())}
            for a, label in enumerate(DECISIONS)}
        tables[task] = table
    return {"algorithm": ALGORITHM, "labels": list(DECISIONS), "gain_weight": GAIN_WEIGHT,
            "source_adapter": str(source_adapter), "release": str(release), "tasks": tables,
            "members": population, "scope": "fixed for one complete epoch; train-only hard direct predictions",
            "hidden_test_read": False}


def check_cost_population(snapshot, data):
    assert snapshot["algorithm"] == ALGORITHM and snapshot["gain_weight"] == GAIN_WEIGHT
    assert snapshot["labels"] == list(DECISIONS), "Action column order changed"
    current = action_population(data)
    assert current.keys() == snapshot["members"].keys(), "Frozen costs use different training members"
    for sid, row in current.items():
        old = snapshot["members"][sid]
        assert (row["task"], row["action"]) == (old["task"], old["action"]), sid
        assert math.isclose(row["mass"], old["mass"], rel_tol=1e-10, abs_tol=1e-14), sid
    for task in TASKS:
        derived = cost_table(snapshot["tasks"][task]["confusion"])
        for key in ("mass", "hard_coefficients", "gain_coefficients"):
            assert torch.allclose(torch.tensor(derived[key], dtype=torch.float64),
                                  torch.tensor(snapshot["tasks"][task][key], dtype=torch.float64),
                                  rtol=1e-12, atol=1e-12), (task, key)


def cost_tensors(snapshot, device):
    return {task: torch.tensor([table["hard_coefficients"], table["gain_coefficients"]],
                               dtype=torch.float32, device=device)
            for task, table in snapshot["tasks"].items()}


def action_cost_terms(log_probabilities, target_index, coefficients):
    return -(coefficients[:, target_index] * log_probabilities[None, :]).sum(-1)
