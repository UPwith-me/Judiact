"""verl FSDP extension for the existing four-candidate shared audit readout.

This is a model-output/loss adapter, not a standalone training loop. Standard
rollout log-probabilities cannot replace the candidate-normalized audit head.
"""

import torch

from verl.workers.engine.base import EngineRegistry
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead



@EngineRegistry.register(model_type="judiact_shared_audit", backend=["fsdp2"], device=["cuda"])
class JudiActFSDPEngine(FSDPEngineWithLMHead):
    def _build_module(self):
        from transformers import Qwen3VLForConditionalGeneration

        # verl 0.7's PPO Qwen forward drops logits_to_keep and expects supplied
        # mRoPE positions. This audit uses the original HF forward from SFT.
        assert not self.model_config.use_remove_padding
        module = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_config.local_path, torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2", local_files_only=True)
        if self.model_config.enable_gradient_checkpointing:
            module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        return module

    def _build_lora_module(self, module):
        module = super()._build_lora_module(module)
        for parameter in module.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
        # Apply for the whole stage-two update, including checkpoint recomputes.
        # Restoring dropout immediately after forward would change the backward.
        for child in module.modules():
            if isinstance(child, torch.nn.Dropout):
                child.p = 0.
        return module

    def _build_fsdp_module(self, module):
        from torch.distributed.fsdp import MixedPrecisionPolicy
        from verl.utils.fsdp_utils import apply_fsdp2, fsdp2_load_full_state_dict

        assert self.engine_config.strategy == "fsdp2"
        assert not self.engine_config.offload_policy and not self.model_config.enable_activation_offload
        # Upstream defaults cast ALL forward parameters to BF16. Preserve the
        # BF16 base / FP32 adapter distinction instead of silently rounding LoRA.
        policy = MixedPrecisionPolicy(param_dtype=None, reduce_dtype=torch.float32,
                                      cast_forward_inputs=False)
        full_state = module.state_dict()
        apply_fsdp2(module, {"mesh": self.device_mesh, "mp_policy": policy, "offload_policy": None,
                            "reshard_after_forward": self.engine_config.reshard_after_forward},
                    self.engine_config)
        fsdp2_load_full_state_dict(module, full_state, self.device_mesh, None)
        return module
