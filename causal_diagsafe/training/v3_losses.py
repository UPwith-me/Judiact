"""Proper semantic NLL and verified probability-residual finite differences."""

from functools import lru_cache

import torch


@lru_cache(maxsize=512)
def contrast_basis(rows):
    matrix = torch.tensor(rows, dtype=torch.float64)
    _, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
    rank = int((singular > singular.max() * max(matrix.shape) * torch.finfo(matrix.dtype).eps).sum())
    return vh[:rank].float()


def projected_score(log_probabilities, targets, operators):
    probability = torch.stack([item.exp() for item in log_probabilities])
    truth = torch.nn.functional.one_hot(torch.tensor(targets, device=probability.device),
                                        num_classes=probability.shape[-1]).float()
    basis = contrast_basis(tuple(tuple(row) for row in operators)).to(probability.device)
    return (basis @ (probability - truth)).square().mean()


def job_loss(model, job, encoder, relation_coefficient, view_batch_size=4):
    if job["kind"] == "classify_batch":
        logs = encoder.classification(model, [(item["record"], item["name"]) for item in job["items"]])
        return torch.stack([-values[item["target"]] * item["weight"]
                            for values, item in zip(logs, job["items"])]).sum()
    if job["kind"] == "evidence":
        return encoder.evidence_loss(model, job["record"], job["stage"], job["field"], job["answer"]) * job["weight"]
    logs = []
    for start in range(0, len(job["specs"]), view_batch_size):
        specs = job["specs"][start:start + view_batch_size]
        values = encoder.classification(model, [(item["record"], item["name"]) for item in specs])
        logs.extend(value[item["gather"]] if "gather" in item else value for item, value in zip(specs, values))
    # lambda=0 still performs the same forwards/backwards; it changes no absolute exposure.
    return projected_score(logs, [item["target"] for item in job["specs"]], job["operators"]) \
        * job["weight"] * relation_coefficient


def stream_jobs(packet, view_batch_size):
    yield from packet["other"]
    items = packet["classify"]
    for start in range(0, len(items), view_batch_size):
        yield {"kind": "classify_batch", "items": items[start:start + view_batch_size]}
