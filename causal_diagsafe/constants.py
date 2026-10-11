from __future__ import annotations

STAGES: tuple[str, ...] = ("S1", "S2", "S3", "S4", "S5")
DECISIONS: tuple[str, ...] = ("allow", "warn", "review", "refuse")
EVIDENCE_STATES: tuple[str, ...] = ("absent", "present", "uncertain")
TERNARY_STATES: tuple[str, ...] = ("no", "yes", "uncertain")
GROUP_ROLES: tuple[str, ...] = ("unsafe", "safe", "helpful")
POLICY_CONDITIONS: tuple[str, ...] = ("correct", "wrong", "missing", "decoy", "unknown")
CHANGED_FACTORS: tuple[str, ...] = (
    "visual_evidence",
    "ocr_text",
    "image_text_binding",
    "intent",
    "policy_rule",
    "response_execution",
    "none",
    "unknown",
)
INTENT_CLASSES: tuple[str, ...] = (
    "malicious",
    "jailbreak_or_evasion",
    "benign_educational",
    "benign_protective",
    "benign_authorized_routine",
    "benign_news_or_discussion",
    "benign_fiction_or_art",
    "ambiguous",
    "unknown",
)
POLICY_SPECIAL_IDS: tuple[str, ...] = ("no_match", "insufficient_policy", "unknown")

DECISION_TO_ID = {name: i for i, name in enumerate(DECISIONS)}
ID_TO_DECISION = {i: name for name, i in DECISION_TO_ID.items()}
EVIDENCE_TO_ID = {name: i for i, name in enumerate(EVIDENCE_STATES)}
ID_TO_EVIDENCE = {i: name for name, i in EVIDENCE_TO_ID.items()}
STAGE_TO_ID = {name: i for i, name in enumerate(STAGES)}
ID_TO_STAGE = {i: name for name, i in STAGE_TO_ID.items()}
INTENT_TO_ID = {name: i for i, name in enumerate(INTENT_CLASSES)}
ID_TO_INTENT = {i: name for name, i in INTENT_TO_ID.items()}
