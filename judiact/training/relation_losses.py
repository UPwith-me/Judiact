"""Root-normalized semantic supervision on the model's own stopped-gradient audit."""

from __future__ import annotations

import json
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..tensor_inputs import move_tensors


@dataclass
class EvidenceEncoding:
    text: str
    token_ids: list
    semantic_weights: torch.Tensor
    observations: dict


def encode_evidence(text, tokenizer):
    """Use structured JSON values and fast-tokenizer offsets, not substring searches."""
    value = json.loads(text)
    pieces, ranges = [], []
    length = 0

    def append(piece):
        nonlocal length
        pieces.append(piece)
        length += len(piece)

    def emit(item, semantic_ranges=None):
        if isinstance(item, dict):
            append("{")
            for index, (key, child) in enumerate(item.items()):
                if index:
                    append(",")
                append(json.dumps(key, ensure_ascii=False) + ":")
                emit(child, semantic_ranges)
            append("}")
        elif isinstance(item, list):
            append("[")
            for index, child in enumerate(item):
                if index:
                    append(",")
                emit(child, semantic_ranges)
            append("]")
        else:
            literal = json.dumps(item, ensure_ascii=False)
            start = length + (1 if isinstance(item, str) else 0)
            end = length + len(literal) - (1 if isinstance(item, str) else 0)
            if semantic_ranges is not None and item is not None and end > start:
                semantic_ranges.append((start, end))
            append(literal)

    append("{")
    for array_index, name in enumerate(("quotes", "regions")):
        if array_index:
            append(",")
        append(json.dumps(name) + ":[")
        for index, record in enumerate(value[name]):
            if index:
                append(",")
            record_ranges = []
            emit(record, record_ranges)
            ranges.append(record_ranges)
        append("]")
    append("}")
    canonical = "".join(pieces)
    assert json.loads(canonical) == value
    encoded = tokenizer(canonical, add_special_tokens=False, truncation=False, return_offsets_mapping=True)
    weights = torch.zeros(len(ranges), len(encoded["input_ids"]), dtype=torch.float32)
    for row, intervals in enumerate(ranges):
        for column, (start, end) in enumerate(encoded["offset_mapping"]):
            if end > start:
                covered = sum(max(0, min(end, b) - max(start, a)) for a, b in intervals)
                weights[row, column] = covered / (end - start)
    assert len(ranges) and bool((weights.sum(-1) > 0).all()), "only positively localized evidence has a target"
    token_mass = weights.sum(0)
    observations = {"target_tokens": len(encoded["input_ids"]), "evidence_records": len(ranges),
                    "semantic_token_mass": float(token_mass.sum()),
                    "format_only_tokens": int((token_mass == 0).sum()),
                    "mixed_semantic_format_tokens": int(((token_mass > 0) & (token_mass < 1)).sum())}
    return EvidenceEncoding(canonical, encoded["input_ids"], weights, observations)


def evidence_nll(encoder, model, trajectory, name, target):
    encoding = encode_evidence(target, encoder.processor.tokenizer)
    selected = encoding.semantic_weights.sum(0).nonzero().flatten()
    entry = next(item for item in trajectory.entries if item["name"] == name)
    device = next(model.parameters()).device
    # Each target token is predicted from its preceding position; later fields remain untouched.
    inputs = move_tensors(trajectory.inputs(through=name, target_ids=encoding.token_ids), device)
    positions = (selected + entry["answer_start"] - 1).to(device)
    with encoder.visual_context(model, [trajectory.visible]):
        logits = model(**inputs, use_cache=False, logits_to_keep=positions).logits[0].float()
    target_ids = torch.tensor(encoding.token_ids, dtype=torch.long, device=device)[selected.to(device)]
    token_losses = F.cross_entropy(logits, target_ids, reduction="none")
    weights = encoding.semantic_weights[:, selected].to(device)
    loss = ((weights @ token_losses) / weights.sum(-1)).mean()
    return loss, {**encoding.observations, "target_exceeds_rollout_budget": len(encoding.token_ids) > encoder.max_evidence_tokens}


def evidence_nll_batch(encoder, model, examples):
    """Batch independent local gold blocks without exposing them to any later audit slot."""
    encodings = [encode_evidence(target, encoder.processor.tokenizer) for _, _, target in examples]
    prefixes = [trajectory.inputs(through=name, target_ids=encoding.token_ids)["input_ids"].flatten()
                for (trajectory, name, _), encoding in zip(examples, encodings)]
    length = max(prefix.numel() for prefix in prefixes)
    selected = [encoding.semantic_weights.sum(0).nonzero().flatten() for encoding in encodings]
    positions = []
    for (trajectory, name, _), prefix, indices in zip(examples, prefixes, selected):
        entry = next(item for item in trajectory.entries if item["name"] == name)
        positions.append(indices + entry["answer_start"] - 1 + length - prefix.numel())
    unique = sorted({position for row in positions for position in row.tolist()})
    columns = {position: column for column, position in enumerate(unique)}
    device = next(model.parameters()).device
    inputs = move_tensors(encoder.batch_inputs([item[0].base_inputs for item in examples], prefixes), device)
    with encoder.visual_context(model, [item[0].visible for item in examples]):
        logits = model(**inputs, use_cache=False, logits_to_keep=torch.tensor(unique, device=device)).logits.float()
    result = []
    for row, (encoding, indices) in enumerate(zip(encodings, selected)):
        local = logits[row, torch.tensor([columns[value] for value in positions[row].tolist()], device=device)]
        targets = torch.tensor(encoding.token_ids, device=device)[indices.to(device)]
        token_losses = F.cross_entropy(local, targets, reduction="none")
        weights = encoding.semantic_weights[:, indices].to(device)
        result.append(((weights @ token_losses) / weights.sum(-1)).mean())
    return result


@dataclass
class MemberLosses:
    action: torch.Tensor | None
    diagnostic: dict
    native: torch.Tensor | None


def member_losses(case, trajectory, readouts, evidence_losses=None):
    labels = {entry["name"]: entry["labels"] for entry in trajectory.entries if entry["kind"] == "classify"}
    losses = {name: -readouts[name][labels[name].index(target)] for name, target in case["targets"].items()
              if name in labels}
    native = case["task_type"].startswith("native_")
    if native:
        assert set(losses) == {case["task_type"]}
        return MemberLosses(None, {}, losses[case["task_type"]])
    assert "action" in losses
    action = losses.pop("action")
    expected = {name for name in case["targets"] if name.startswith("evidence_")}
    assert set(evidence_losses or {}) == expected, (case["sample_id"], "missing or fabricated evidence supervision")
    losses.update(evidence_losses or {})
    return MemberLosses(action, losses, None)


def absolute_weights(populations):
    """Equal request/response budgets, then roots/contexts, then members within a root."""
    weights = {}
    for kind, units in populations.items():
        budget = 1. if kind == "auxiliary" else .5
        for members in units.values():
            for sample_id in members:
                assert sample_id not in weights
                weights[sample_id] = budget / (len(units) * len(members))
    return weights


def absolute_loss(members, weights):
    """Partial batches sum their full-population contributions without changing relative budgets."""
    exemplar = next((row.action if row.action is not None else row.native) for row in members.values())
    zero = exemplar * 0
    action, diagnostic, native = zero, zero, zero
    for sample_id, row in members.items():
        weight = weights[sample_id]
        if row.action is not None:
            action = action + weight * row.action
            if row.diagnostic:
                diagnostic = diagnostic + weight * torch.stack(list(row.diagnostic.values())).mean()
        if row.native is not None:
            native = native + weight * row.native
    return {"L_A": action, "L_D": diagnostic, "L_N": native}


def joint_loss(absolute, relation, relation_coefficient=1., native_coefficient=.1):
    """The matched no-relation model still reads all the same member prefixes."""
    return absolute["L_A"] + absolute["L_D"] + relation_coefficient * relation + native_coefficient * absolute["L_N"]
