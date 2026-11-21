"""Save the public audit interface and read metadata from existing adapters."""

from functools import partial
import json
from pathlib import Path

from .short_review import PROTOCOL_VERSION


CHECKPOINT_CONFIG = "judiact_config.json"


def read_checkpoint(adapter):
    directory = Path(adapter)
    current = directory / CHECKPOINT_CONFIG
    if current.is_file():
        config = json.loads(current.read_text(encoding="utf-8"))
        if config["protocol"] != PROTOCOL_VERSION:
            raise ValueError(f"Unsupported JudiAct protocol: {config['protocol']}")
        return config

    # Earlier artifacts can be identified by the trained interface, not a filename.
    candidates = []
    for path in sorted(directory.glob("*_config.json")):
        config = json.loads(path.read_text(encoding="utf-8"))
        if {"short_review", "short_max_new_tokens", "action_objective"} <= config.keys():
            candidates.append(config)
    if len(candidates) != 1:
        raise ValueError(f"Expected one short-review audit configuration in {directory}")
    return {**candidates[0], "protocol": PROTOCOL_VERSION}


def predictor_for_checkpoint(adapter, **execution):
    from .short_predictor import ShortReviewPredictor

    config = read_checkpoint(adapter)
    return partial(ShortReviewPredictor,
                   short_max_new_tokens=config["short_max_new_tokens"],
                   action_objective=config["action_objective"]["objective"],
                   candidate_head_projection=config.get("candidate_head_projection", False),
                   short_compact_after=config.get("short_compact_after", 192),
                   **execution)
