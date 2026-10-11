"""The 2026-09-29 diagnostic method; no legacy router or generated trace."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn


FIELDS = ("image", "query", "ocr_text", "policy_rule", "response_candidate")


class DiagnosticHeads(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        # Uniform initial attention avoids arbitrary evidence concentration on Qwen's outliers.
        self.queries = nn.Parameter(torch.zeros(5, 5, hidden_size))
        self.null_states = nn.Parameter(torch.zeros(5, hidden_size))
        self.state_heads = nn.ModuleDict({
            variable: nn.ModuleList(nn.Linear((7 if k in (1, 3) else 6) * hidden_size, 3)
                                    for k in range(4 if variable == "R" else 5))
            for variable in ("R", "E", "F")
        })
        self.intent = nn.Linear(6 * hidden_size, 8)
        self.behavior = nn.Linear(6 * hidden_size, 5)
        self.policy_actions = nn.Linear(3 * hidden_size + 44, 4)
        self.policy_special = nn.Linear(hidden_size + 44, 2)
        self.router = nn.Linear(36 + 8 + 6 + 36 * 6 + 8 * 6, 4)
        self.publishability = nn.Linear(2 * hidden_size + 6, 4)
        self.native_heads = nn.ModuleDict({
            task: nn.Linear(hidden_size, 2)
            for task in ("native_prompt_harm", "native_response_harm", "native_image_harm")
        })
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.0001)
                nn.init.zeros_(module.bias)

    def locate(self, hidden, mask, query, null):
        scores = torch.einsum("blh,h->bl", hidden, query.to(hidden.dtype)).float()
        null_score = (query * null).sum().expand(hidden.shape[0], 1)
        scores = torch.cat((scores.masked_fill(~mask, -torch.inf), null_score), dim=-1)
        weights = scores.softmax(-1)
        context = torch.einsum("bl,blh->bh", weights[:, :-1].to(hidden.dtype), hidden)
        context = context + weights[:, -1:].to(hidden.dtype) * null.to(hidden.dtype)
        return context, weights

    def forward(self, hidden, attention_mask, field_masks, rule_masks, rule_counts, response_present):
        batch, length, _ = hidden.shape
        positions = torch.arange(length, device=hidden.device).expand(batch, -1)
        anchor_positions = positions.masked_fill(~attention_mask.bool(), -1).max(-1).values
        anchor = hidden[torch.arange(batch, device=hidden.device), anchor_positions]
        contexts, weights, features = [], [], []
        for k in range(5):
            located = [self.locate(hidden, field_masks[:, m], self.queries[k, m],
                                   self.null_states[m]) for m in range(5)]
            context = torch.stack([item[0] for item in located], dim=1)
            contexts.append(context)
            weights.append(torch.stack([item[1] for item in located], dim=1))
            parts = [anchor, *context.unbind(1)]
            if k == 1:
                parts.append(context[:, 0] * context[:, 1])
            if k == 3:
                parts.append(context[:, :3].mean(1) * context[:, 3])
            features.append(torch.cat(parts, dim=-1))
        states = {}
        for variable, heads in self.state_heads.items():
            values = [head(features[k]).float() for k, head in enumerate(heads)]
            if variable == "R":
                r5 = torch.full_like(values[0], -torch.inf)
                r5.scatter_(1, response_present.long()[:, None], 0)
                values.append(r5)
            states[variable] = torch.stack(values, dim=1)
        q_stage = torch.cat([states[v][:, :4].softmax(-1).flatten(1)
                             for v in ("R", "E", "F")], dim=-1)
        intent_logits = self.intent(features[2]).float()
        q_intent = intent_logits.softmax(-1)
        request_evidence = contexts[3][:, :3].mean(1)
        summaries = torch.cat((q_stage, q_intent), dim=-1)
        max_rules = rule_masks.shape[1]
        rule_scores = []
        for p in range(max_rules):
            rule, _ = self.locate(hidden, rule_masks[:, p], self.queries[3, 3], self.null_states[3])
            z = torch.cat((rule, request_evidence, summaries, rule * request_evidence), dim=-1)
            rule_scores.append(self.policy_actions(z).float())
        scores = torch.stack(rule_scores, dim=1) if rule_scores else hidden.new_empty(batch, 0, 4).float()
        valid_rules = torch.arange(max_rules, device=hidden.device)[None] < rule_counts[:, None]
        scores = scores.masked_fill(~valid_rules[:, :, None], -torch.inf)
        special = self.policy_special(torch.cat((request_evidence, summaries), dim=-1)).float()
        joint_logits = torch.cat((scores.flatten(1), special), dim=-1)
        empty_deck = rule_counts == 0
        forced_policy = torch.full_like(joint_logits, -torch.inf)
        forced_policy[:, -1] = 0
        joint_logits = torch.where(empty_deck[:, None], forced_policy, joint_logits)
        joint = joint_logits.softmax(-1)
        q_policy = torch.cat((joint[:, :4 * max_rules].reshape(batch, max_rules, 4).sum(1),
                              joint[:, -2:]), dim=-1)
        phi = torch.cat((q_stage, q_intent, q_policy,
                         (q_stage[:, :, None] * q_policy[:, None, :]).flatten(1),
                         (q_intent[:, :, None] * q_policy[:, None, :]).flatten(1)), dim=-1)
        request_logits = self.router(phi).float()
        forced_request = torch.full_like(request_logits, -torch.inf)
        forced_request[:, 2] = 0
        request_logits = torch.where(empty_deck[:, None], forced_request, request_logits)
        s5_probabilities = torch.cat((states["E"][:, 4].softmax(-1),
                                     states["F"][:, 4].softmax(-1)), dim=-1)
        publish_features = torch.cat((contexts[4][:, 4], contexts[4][:, 3], s5_probabilities), dim=-1)
        return {
            "stage": states, "intent": intent_logits, "policy_joint": joint_logits,
            "request_action": request_logits,
            "response_publishability": self.publishability(publish_features).float(),
            "safe_response_behavior": self.behavior(features[2]).float(),
            "evidence_weights": torch.stack(weights, dim=1),
            "native": {task: head(anchor).float() for task, head in self.native_heads.items()},
        }


class FinalOnlyHeads(nn.Module):
    """Matched-backbone classifiers without stage, policy or evidence supervision."""

    def __init__(self, hidden_size):
        super().__init__()
        self.request = nn.Linear(hidden_size, 4)
        self.publishability = nn.Linear(hidden_size, 4)
        self.native_heads = nn.ModuleDict({
            task: nn.Linear(hidden_size, 2)
            for task in ("native_prompt_harm", "native_response_harm", "native_image_harm")
        })
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.0001)
                nn.init.zeros_(module.bias)

    def forward(self, hidden, attention_mask, field_masks, rule_masks, rule_counts, response_present):
        positions = torch.arange(hidden.shape[1], device=hidden.device).expand(hidden.shape[0], -1)
        last = positions.masked_fill(~attention_mask.bool(), -1).max(-1).values
        anchor = hidden[torch.arange(hidden.shape[0], device=hidden.device), last]
        request = self.request(anchor).float()
        forced = torch.full_like(request, -torch.inf)
        forced[:, 2] = 0
        return {
            "request_action": torch.where((rule_counts == 0)[:, None], forced, request),
            "response_publishability": self.publishability(anchor).float(),
            "native": {task: head(anchor).float() for task, head in self.native_heads.items()},
        }


class DiagSafeV1(nn.Module):
    def __init__(self, model_path: str, adapter_path: str | None = None, variant: str | None = None):
        super().__init__()
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import Qwen3VLForConditionalGeneration

        container, loading = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, dtype=torch.bfloat16, attn_implementation="sdpa",
            local_files_only=True, output_loading_info=True)
        assert not loading["missing_keys"] and not loading["mismatched_keys"], loading
        self.loading_info = loading
        base = container.model
        del container
        base.config.use_cache = False
        base.config.text_config.use_cache = False
        targets = r"language_model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)"
        if adapter_path:
            self.backbone = PeftModel.from_pretrained(base, adapter_path, is_trainable=True)
        else:
            self.backbone = get_peft_model(base, LoraConfig(
                r=16, lora_alpha=32, lora_dropout=0.05, target_modules=targets, bias="none"))
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        saved_config = json.loads((Path(adapter_path) / "diagnostic_config.json").read_text()) if adapter_path else {}
        self.variant = variant or saved_config.get("variant", "method")
        heads = FinalOnlyHeads if self.variant == "final_only" else DiagnosticHeads
        self.heads = heads(base.config.text_config.hidden_size)
        self.config = base.config
        self.model_path = model_path
        if adapter_path:
            self.heads.load_state_dict(torch.load(Path(adapter_path) / "diagnostic_heads.pt",
                                                  map_location="cpu", weights_only=True))

    def forward(self, encoded, field_masks, rule_masks, rule_counts, response_present,
                supervision=None, eval_nll=False):
        output = self.backbone(**encoded, use_cache=False, return_dict=True)
        predictions = self.heads(output.last_hidden_state, encoded["attention_mask"],
                                 field_masks, rule_masks, rule_counts, response_present)
        if supervision is None:
            return predictions
        from ..training.v1_losses import final_only_loss, natural_nll, training_loss

        if eval_nll:
            loss = natural_nll(predictions, supervision["labels"])
        elif self.variant == "final_only":
            loss = final_only_loss(predictions, supervision)
        else:
            loss = training_loss(predictions, supervision, use_relations=self.variant != "no_edge")
        # DDP must trace the loss, not unused task outputs, when finding unused heads.
        return {"loss": loss}

    def save_pretrained(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(directory)
        torch.save(self.heads.state_dict(), directory / "diagnostic_heads.pt")
        (directory / "diagnostic_config.json").write_text(json.dumps({
            "method_version": "20260929", "model_path": self.model_path,
            "F_contrast": "log_P1_minus_log_P0", "fields": FIELDS,
            "head_weight_initial_std": 0.0001, "evidence_query_initialization": "zero",
            "variant": self.variant,
        }, indent=2), encoding="utf-8")

    def load_trainable(self, directory):
        from peft import set_peft_model_state_dict
        from peft.utils.save_and_load import load_peft_weights

        set_peft_model_state_dict(self.backbone, load_peft_weights(str(directory), device="cpu"))
        self.heads.load_state_dict(torch.load(Path(directory) / "diagnostic_heads.pt",
                                              map_location="cpu", weights_only=True))
