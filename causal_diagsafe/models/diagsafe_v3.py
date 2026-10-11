"""The complete pretrained Qwen language readout, with one language-only LoRA."""

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


PROCESSOR_SIZE = {"shortest_edge": 65536, "longest_edge": 1003520}


def load_processor_v3(path):
    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    processor.image_processor.size = dict(PROCESSOR_SIZE)
    processor.tokenizer.padding_side = "left"
    return processor


def load_model_v3(path, adapter=None, training=False):
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        path, local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa")
    model.config.use_cache = False
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=training)
    elif training:
        model = get_peft_model(model, LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=.05,
            bias="none", init_lora_weights=True,
            target_modules=r"model\.language_model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)"))
    if training:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        trainable = [name for name, value in model.named_parameters() if value.requires_grad]
        assert trainable and all(".language_model.layers." in name and ".lora_" in name
                                 for name in trainable), trainable
        assert not model.get_output_embeddings().weight.requires_grad
    return model
