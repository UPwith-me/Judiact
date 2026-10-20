"""Exact full-support arithmetic correction and the fixed epoch-start LoRA."""

from contextlib import contextmanager
import copy

import torch
from peft import get_peft_model_state_dict, set_peft_model_state_dict


def correction_ce(logits, target_ids, old_logits=None, anchoring=0., hard_mask=None):
    """Per-position CE(q,p), q=(onehot + lambda*p_old)/(1+lambda).

    An anchor-only position has q=p_old, not a zero or fabricated hard target.
    Float64 is retained for mathematical comparisons; training reductions are FP32.
    """
    values = logits if logits.dtype == torch.float64 else logits.float()
    logp = values.log_softmax(-1)
    hard = -logp.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    if not anchoring:
        if hard_mask is not None:
            hard = hard * hard_mask.to(hard)
        return hard
    if old_logits is None or old_logits.shape != logits.shape:
        raise ValueError("anchoring requires the same complete support and context")
    old = old_logits.detach().to(values.dtype).softmax(-1)
    soft = -(old * logp).sum(-1)
    mixed = (hard + anchoring * soft) / (1 + anchoring)
    return mixed if hard_mask is None else torch.where(hard_mask, mixed, soft)


class EpochAnchor:
    """One shared frozen base, two adapters; no second 4B model or quantization.

    Reference calls must finish before building a student autograd graph. Switching
    adapters inside a pending checkpointed graph changes its recomputation.
    """

    name = "v7_epoch_anchor"

    def __init__(self, model):
        self.model = model
        model.add_adapter(self.name, copy.deepcopy(model.peft_config["default"]))
        self.select_student()
        self.refresh()

    def select_student(self):
        self.model.set_adapter("default")
        for name, value in self.model.named_parameters():
            if "." + self.name + "." in name:
                value.requires_grad_(False)

    @torch.no_grad()
    def refresh(self):
        state = {name: value.detach().clone() for name, value in
                 get_peft_model_state_dict(self.model, adapter_name="default").items()}
        set_peft_model_state_dict(self.model, state, adapter_name=self.name)
        self.select_student()

    @contextmanager
    def reference(self):
        training = self.model.training
        self.model.set_adapter(self.name)
        self.model.requires_grad_(False)
        self.model.eval()
        try:
            with torch.no_grad():
                yield self.model
        finally:
            self.select_student()
            self.model.train(training)
