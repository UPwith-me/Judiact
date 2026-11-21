"""Reference-derived W continuation views, without new semantic labels."""

import json
import random

from ..reference_audit import canonical
from .sequence import NodeTarget


def recovery_views(example, encoder, seed, epoch, length_match=False):
    """Keep full fact dictionaries; condition on past tokens and learn the suffix.

    Compare the complete/continue strings and supervise their first differing
    token, conditioned only on the common past. This gives stop and continue
    equal decision mass instead of comparing a two-token stop to a long suffix.
    Full-W SFT still teaches content. No proposal probability is invented here.
    """
    assert example.node.name == "witness" and not example.node.labels
    value = json.loads(example.target) if isinstance(example.target, str) else example.target
    facts = list({canonical(fact): fact for fact in value["facts"]}.values())
    if not facts:
        return []
    rng = random.Random(f"{seed}/{epoch}/{example.sample_id}/witness_recovery")
    rng.shuffle(facts)
    views = []
    for mode, count in (("stop", len(facts)), ("continue", rng.randrange(len(facts)))):
        previous, remaining = facts[:count], facts[count:]
        if previous:
            repeated = rng.choice(previous)
            at = rng.randrange(len(previous) + 1)
            previous = previous[:at] + [repeated] + previous[at:]
            if length_match and mode == "continue":
                # An incomplete history can be as long as the complete one.
                # Otherwise "long enough -> stop" would solve these views.
                while len(previous) < len(facts) + 1:
                    at = rng.randrange(len(previous) + 1)
                    previous.insert(at, rng.choice(facts[:count]))

        def encode(parts):
            prefix = '{"facts":[' + ",".join(map(canonical, parts))
            tail = ("," if parts and remaining else "") + ",".join(map(canonical, remaining)) + "]}"
            text = prefix + tail
            other = prefix + ("," + canonical(facts[0]) if not remaining else "") + "]}"
            ids, alternate = encoder.processor.tokenizer(
                [text, other], add_special_tokens=False)["input_ids"]
            cut = next(i for i, (a, b) in enumerate(zip(ids, alternate)) if a != b)
            return text, ids, cut

        text, ids, cut = encode(previous)
        if len(ids) + 1 > encoder.max_new_tokens:
            # Preserve all information within the declared W support; only the
            # extra duplicate is removed, not a fact or a source condition.
            previous = facts[:count]
            text, ids, cut = encode(previous)
        assert len(ids) + 1 <= encoder.max_new_tokens
        assert {canonical(f) for f in json.loads(text)["facts"]} == set(map(canonical, facts))
        assert cut < len(ids)
        views.append(NodeTarget(
            example.sample_id, example.node, text, view="recovery_" + mode,
            continuation_ids=ids[cut:cut + 1], terminated=False, prefix_ids=ids[:cut]))
    return views
