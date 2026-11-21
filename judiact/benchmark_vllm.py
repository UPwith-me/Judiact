"""Continuous-batched execution of the unchanged local judge prompts."""

import copy
import time
from .benchmark_response_models import generation_request, open_images


class ContinuousBatcher:
    """Feed a large pending window to vLLM; its scheduler owns active GPU batches."""

    def __init__(self, window=512):
        self.window = window
        self.measurements = []
        self.generation_tokens = 0

    def run(self, entries, infer, on_error=None):
        for start in range(0, len(entries), self.window):
            batch = entries[start:start + self.window]
            begin = time.perf_counter()
            values = infer(batch)
            seconds = time.perf_counter() - begin
            if len(values) != len(batch):
                raise ValueError("Engine outputs must retain every submitted request")
            self.measurements.append({"batch_size": len(batch), "seconds": seconds,
                "records_per_second": len(batch) / seconds,
                "output_tokens_per_second": sum(v.get("output_tokens", 0) for v in values) / seconds})
            for offset, value in enumerate(values):
                yield start + offset, value

    def summary(self):
        return {"engine": "vllm", "submission_window": self.window, "measurements": self.measurements}


class VLLMBenchmarkJudge:
    def __init__(self, path, config, *, tensor_parallel_size=1, memory_utilization=.95,
                 max_model_len=32768, max_num_seqs=64, max_num_batched_tokens=8192,
                 window=512, enforce_eager=False, disable_custom_all_reduce=False):
        from transformers import AutoProcessor, GenerationConfig
        from vllm import LLM, __version__

        path = str(path)
        self.processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        self.profile = GenerationConfig.from_pretrained(path, local_files_only=True)
        self.config = config
        self.capture_prompt_tokens = False
        self.engine_settings = {"backend": "vllm", "version": __version__, "dtype": "bfloat16",
            "kv_cache_dtype": "auto", "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": memory_utilization, "max_model_len": max_model_len,
            "max_num_seqs": max_num_seqs, "max_num_batched_tokens": max_num_batched_tokens,
            "enable_prefix_caching": True, "enable_chunked_prefill": True,
            "enforce_eager": enforce_eager,
            "disable_custom_all_reduce": disable_custom_all_reduce}
        self.engine = LLM(model=path, tokenizer=path, generation_config="vllm", seed=config["seed"],
            limit_mm_per_prompt={"image": 1, "video": 0},
            **{k: v for k, v in self.engine_settings.items() if k not in ("backend", "version")})
        self.eval_batcher = ContinuousBatcher(window)

    def prepare(self, visible, task):
        from vllm import SamplingParams

        text, settings = generation_request(self.processor, visible, task, self.config, True)
        profile = copy.deepcopy(self.profile)
        profile.update(**settings)
        if profile.do_sample or profile.num_beams != 1:
            raise ValueError("This adapter is for the declared greedy local judge, not sampled or beam-search answers")
        eos = profile.eos_token_id
        eos_ids = [eos] if isinstance(eos, int) else list(eos or [])
        params = SamplingParams(temperature=0, top_p=1, top_k=-1,
            max_tokens=profile.max_new_tokens, repetition_penalty=profile.repetition_penalty,
            stop_token_ids=eos_ids, skip_special_tokens=True)
        prompt = {"prompt": text}
        if visible["images"]:
            prompt["multi_modal_data"] = {"image": open_images(visible)}
        return prompt, params, profile

    def predict_batch(self, visibles, tasks=None):
        tasks = tasks or [None] * len(visibles)
        results, prompts, params, profiles, pending = [None] * len(visibles), [], [], [], []
        for index, (visible, task) in enumerate(zip(visibles, tasks)):
            raw = visible.get("source_defined_judgment")
            if raw is not None:
                results[index] = {"status": "ok", "harmful": None, "score": None, "response_candidate": raw,
                    "source_defined_rule": "SIUO actual-empty-response safety, not a missing generation"}
                continue
            prompt, sampling, profile = self.prepare(visible, task)
            prompts.append(prompt)
            params.append(sampling)
            profiles.append(profile)
            pending.append(index)
        if not pending:
            return results
        begin = time.perf_counter()
        outputs = self.engine.generate(prompts, params, use_tqdm=False)
        elapsed = time.perf_counter() - begin
        if len(outputs) != len(pending):
            raise ValueError("vLLM did not return every submitted judge request")
        for index, output, profile in zip(pending, outputs, profiles):
            completion = output.outputs[0]
            tokens = completion.token_ids
            text = self.processor.tokenizer.decode(tokens, skip_special_tokens=True)
            results[index] = {"status": "ok" if text.strip() else "empty_response", "harmful": None, "score": None,
                "response_candidate": text, "input_tokens": len(output.prompt_token_ids), "output_tokens": len(tokens),
                "generation_limit_reached": completion.finish_reason == "length",
                "generation_settings": {"do_sample": False, "temperature": 1., "top_p": 1., "top_k": 50,
                    "num_beams": 1, "max_new_tokens": profile.max_new_tokens,
                    "repetition_penalty": profile.repetition_penalty},
                "engine_sampling_settings": {"temperature": 0., "top_p": 1., "top_k": -1,
                    "max_tokens": profile.max_new_tokens, "repetition_penalty": profile.repetition_penalty},
                "engine_settings": self.engine_settings,
                "timing": {"batch_wall_seconds": elapsed, "batch_size": len(pending),
                           "wall_seconds": elapsed / len(pending)}}
            if self.capture_prompt_tokens:
                results[index]["prompt_token_ids"] = output.prompt_token_ids
        return results
