"""Read the frozen Core contract without legacy label inference or defaults."""

from __future__ import annotations

from typing import Any, Iterable

from .io_utils import read_jsonl


STAGES = ("S1", "S2", "S3", "S4", "S5")
DECISIONS = ("allow", "warn", "review", "refuse")
INTENTS = (
    "malicious", "jailbreak_or_evasion", "benign_educational", "benign_protective",
    "benign_authorized_routine", "benign_news_or_discussion", "benign_fiction_or_art", "ambiguous",
)
BEHAVIORS = (
    "answer_normally", "brief_warning", "ask_clarification",
    "refuse_with_safe_alternative", "human_review",
)
IGNORE_LABEL = -100
NATIVE_HARM_LABELS = {"safe": 0, "unharmful": 0, "unsafe": 1, "harmful": 1}


def audit_target_withheld(name, task, fields):
    """Translate internal gold-field masks without exposing them to the model."""
    if len(name) > 1 and name[0] in "REF" and name[1:].isdigit():
        paths = [f"stage_states.S{name[1:]}.{name[0]}"]
    elif name.startswith("evidence_"):
        stage = name.removeprefix("evidence_")
        paths = [f"evidence_spans.{stage}", f"evidence_regions.{stage}"]
    else:
        paths = [{"action": "response_publishability" if task == "response_audit" else "request_action",
                  "behavior": "safe_response_behavior", "intent": "intent_class"}.get(name, name)]
    return any(path == field or path.startswith(field + ".") for path in paths for field in fields)


def encode_sample(sample: dict[str, Any]) -> dict[str, Any]:
    gold = sample["gold"]
    response_task = sample["task_type"] == "response_audit"
    model_input = {field: sample["model_input"][field] for field in (
        "query", "images", "ocr_text", "policy_deck", "response_candidate",
    )}
    stage_targets = {"R": [], "E": [], "F": []}
    for stage in STAGES:
        state = gold["stage_states"][stage]
        applicable = (stage == "S5") == response_task
        for variable, values in (("R", (0, 1, "uncertain")),
                                 ("E", ("absent", "present", "uncertain")),
                                 ("F", (0, 1, "uncertain"))):
            # S5 requirement follows candidate presence; the method does not train an R5 head.
            target = values.index(state[variable]) if applicable and not (stage == "S5" and variable == "R") \
                else IGNORE_LABEL
            stage_targets[variable].append(target)

    deck = model_input["policy_deck"]
    policy_candidates = [(card["policy_id"], action) for card in deck for action in DECISIONS]
    policy_candidates += [("no_match", "no_match"), ("insufficient_policy", "insufficient_policy")]
    policy_joint = policy_candidates.index((gold["policy_id"], gold["policy_action"])) \
        if not response_task else IGNORE_LABEL
    evidence_stages = {
        stage for stage in STAGES
        if (stage == "S5") == response_task and gold["stage_states"][stage]["E"] == "present"
    }
    targets = {
        "request_action": DECISIONS.index(gold["request_action"])
        if not response_task else IGNORE_LABEL,
        "response_publishability": DECISIONS.index(gold["response_publishability"])
        if response_task else IGNORE_LABEL,
        "stage": stage_targets,
        "intent": INTENTS.index(gold["intent_class"])
        if not response_task else IGNORE_LABEL,
        "policy_joint": policy_joint,
        "safe_response_behavior": BEHAVIORS.index(gold["safe_response_behavior"])
        if sample["task_type"] != "response_audit" else IGNORE_LABEL,
        # Legacy marks can explain missing evidence; they are not positive localization targets.
        "evidence_spans": [span for span in gold["evidence_spans"] if span["stage"] in evidence_stages],
        "evidence_regions": [region for region in gold["evidence_regions"]
                             if region["stage"] in evidence_stages],
    }
    return {"sample_id": sample["sample_id"], "group_id": sample["group_id"],
            "split": sample["split"], "task_type": sample["task_type"],
            "model_input": model_input, "policy_candidates": policy_candidates, "targets": targets}


def load_samples(path: str) -> Iterable[dict[str, Any]]:
    for sample in read_jsonl(path):
        yield encode_sample(sample)


def load_relations(path: str, samples: dict[str, dict[str, Any]]) -> Iterable[dict[str, Any]]:
    """Expose loss targets from reviewed endpoint gold; retain noncausal edges for analysis."""
    for edge in read_jsonl(path):
        left, right = (samples[edge[field]] for field in ("left_id", "right_id"))
        assert left["group_id"] == right["group_id"] == edge["group_id"]
        assert left["split"] == right["split"] == edge["split"]
        assert left["task_type"] == right["task_type"]
        stage = edge["target_stage"]
        verified = edge["causal_status"] == "single_factor_verified"
        a, b = (sample["gold"]["stage_states"][stage]["F"] for sample in (left, right))
        binary = a in (0, 1) and b in (0, 1)
        decision_field = "response_publishability" if left["task_type"] == "response_audit" \
            else "request_action"
        decisions = [DECISIONS.index(sample["gold"][decision_field]) for sample in (left, right)]
        policy_alignment = []
        invariant_policy = verified and stage == "S4" and edge["intervention"]["changed_factor"] == "none"
        if invariant_policy:
            left_deck, right_deck = (sample["model_input"]["policy_deck"] for sample in (left, right))
            assert len(left_deck) == len(right_deck)
            # Reviewed rule clarifications retain IDs; ID/order perturbations retain rule text.
            alignment_field = "policy_id" if edge["intervention"]["operation"] == "policy_rule_replaced" \
                else "rule_text"
            right_indices = {card[alignment_field]: index for index, card in enumerate(right_deck)}
            assert len(right_indices) == len(right_deck)
            assert {card[alignment_field] for card in left_deck} == set(right_indices)
            for index, card in enumerate(left_deck):
                other = right_indices[card[alignment_field]]
                policy_alignment.extend((4 * index + action, 4 * other + action) for action in range(4))
            policy_alignment.extend((4 * len(left_deck) + i, 4 * len(right_deck) + i) for i in range(2))
        yield {"edge_id": edge["edge_id"], "group_id": edge["group_id"], "split": edge["split"],
               "left_id": edge["left_id"], "right_id": edge["right_id"], "target_stage": stage,
               "causal_status": edge["causal_status"], "use_causal_loss": verified,
               "target_flip_direction": int(b - a) if verified and binary and a != b else 0,
               "stable_target_F": verified and binary and a == b,
               "stable_non_target_stages": edge["relation"]["stable_non_target_stages"] if verified else [],
               "decision_targets": decisions, "stable_decision": verified and decisions[0] == decisions[1],
               "decision_contrast": verified and decisions[0] != decisions[1],
               "policy_alignment": policy_alignment}


def load_s2_quartets(path: str, samples: dict[str, dict[str, Any]]) -> Iterable[dict[str, Any]]:
    """Read explicit, semantically verified four-corner indices, not edge-name guesses."""
    for quartet in read_jsonl(path):
        ids = [quartet["members"][corner] for corner in ("00", "01", "10", "11")]
        members = [samples[sample_id] for sample_id in ids]
        assert all(sample["group_id"] == quartet["group_id"] and sample["split"] == quartet["split"]
                   for sample in members)
        states = [sample["gold"]["stage_states"]["S2"]["F"] for sample in members]
        binary = all(value in (0, 1) for value in states)
        delta = states[3] - states[2] - states[1] + states[0] if binary else None
        assert delta == quartet["Delta_gold"]
        yield {"group_id": quartet["group_id"], "split": quartet["split"], "sample_ids": ids,
               "observed_F2": states, "Delta_gold": delta,
               "use_interaction_loss": binary and delta != 0,
               "interaction_direction": (1 if delta > 0 else -1) if binary and delta != 0 else 0}


def load_native_auxiliary(text_path: str, image_path: str) -> Iterable[dict[str, Any]]:
    """Yield only source-native supervision; no Core decisions or stage gold are inferred.

    Multiple items retain the same root_id and must be sampled as one source context.
    Label-origin fields are metadata for analysis, not model inputs.
    """
    for source in read_jsonl(text_path):
        root_id = source["source_dataset"] + ":" + source["source_id"]
        visible = {"query": source["prompt"], "images": [], "ocr_text": None,
                   "policy_deck": [], "response_candidate": None}
        label = source["prompt_harm_label"]
        if label in NATIVE_HARM_LABELS:
            yield {"sample_id": root_id + ":prompt", "root_id": root_id,
                   "task_type": "native_prompt_harm", "model_input": visible,
                   "native_target": NATIVE_HARM_LABELS[label],
                   "native_label_field": "prompt_harm_label"}
        for index, candidate in enumerate(source["response_candidates"]):
            labels = [(field, value) for field, value in candidate.items()
                      if field.startswith("response_harm_label") and value in NATIVE_HARM_LABELS]
            if not labels:
                continue
            assert len(labels) == 1, (root_id, index, labels)
            field, value = labels[0]
            yield {"sample_id": root_id + f":response{index}", "root_id": root_id,
                   "task_type": "native_response_harm",
                   "model_input": {**visible, "response_candidate": candidate["text"]},
                   "native_target": NATIVE_HARM_LABELS[value], "native_label_field": field}
    for source in read_jsonl(image_path):
        root_id = source["source_dataset"] + ":" + source["source_record_id"]
        image_safe = source["native_image_safe"]
        assert isinstance(image_safe, bool), (root_id, image_safe)
        yield {"sample_id": root_id + ":image", "root_id": root_id,
               "task_type": "native_image_harm",
               "model_input": {"query": source.get("native_query", ""), "images": [{"uri": source["image_uri"],
                                                          "sha256": source["image_sha256"]}],
                               "ocr_text": None, "policy_deck": [], "response_candidate": None},
               "native_target": int(not image_safe), "native_label_field": "native_image_safe"}


def messages(model_input: dict[str, Any], system_prompt: str) -> list[dict[str, Any]]:
    content = [{"type": "image", "image": image["uri"]} for image in model_input["images"]]
    content.append({"type": "text", "text": "\nUser query:\n" + model_input["query"] + "\n"})
    if model_input["ocr_text"] is not None:
        content.append({"type": "text", "text": "Visible OCR transcription:\n" + model_input["ocr_text"] + "\n"})
    deck_text = "\n".join(card["policy_id"] + ": " + card["rule_text"]
                          for card in model_input["policy_deck"])
    content.append({"type": "text", "text": "Policy deck:\n" + deck_text + "\n"})
    if model_input["response_candidate"] is not None:
        content.append({"type": "text", "text": "Candidate response:\n" + model_input["response_candidate"]})
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": content}]
