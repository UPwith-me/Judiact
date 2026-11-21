"""Score delivered answers with the official HarmBench classifier and copyright rubric."""

import importlib.util
import json
import os
from functools import partial
from pathlib import Path

from .benchmark_official import source_literal
from .benchmark_response_models import move_inputs
from .eval_batching import EvalBatcher


class HarmBenchTransport:
    def __init__(self, classifier, references, device="cuda:0"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device
        self.references = Path(references).resolve() / "HarmBench"
        self.templates = source_literal(self.references / "eval_utils.py", "LLAMA2_CLS_PROMPT")
        self.model = AutoModelForCausalLM.from_pretrained(classifier, local_files_only=True,
            dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(classifier, local_files_only=True)
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "right"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.hash_utils = None
        self.eval_batcher = EvalBatcher(devices=[device], initial_batch_size=8, max_batch_size=128, include_policy=False)

    @staticmethod
    def documents(answers):
        docs = []
        for row in answers:
            visible = json.loads(row["visible_json"])
            visible.update(response_candidate=row["response_candidate"],
                           judge_provenance=json.loads(row["provenance_json"]), judge_identifier=row["root_id"])
            docs.append({**row, "sample_id": row["sample_id"] + "::harmbench_test",
                         "visible_json": json.dumps(visible), "audit_target": "response"})
        return docs

    def copyright_match(self, answer, provenance, identifier):
        if self.hash_utils is None:
            from datasketch import MinHash

            spec = importlib.util.spec_from_file_location("harmbench_rubric", self.references / "eval_utils.py")
            self.hash_utils = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.hash_utils)
            self.hash_utils.MinHash = partial(MinHash, scheme="legacy")
        behavior = {"BehaviorID": identifier, "Behavior": provenance["behavior"],
                    "ContextString": provenance["context"], "Tags": provenance["tags"]}
        previous = Path.cwd()
        try:
            os.chdir(self.references)
            return bool(self.hash_utils.compute_results_hashing(behavior, [{"generation": answer}])[0]["label"])
        finally:
            os.chdir(previous)

    def predict_batch(self, visibles, tasks=None):
        import torch

        results, prompts, indices = [None] * len(visibles), [], []
        for i, visible in enumerate(visibles):
            provenance = visible["judge_provenance"]
            tags = {tag.strip() for tag in provenance["tags"].split(",")}
            tokens = self.tokenizer.encode(visible["response_candidate"], max_length=512, truncation=True)
            answer = self.tokenizer.decode(tokens, skip_special_tokens=True)
            if "hash_check" in tags:
                results[i] = "yes" if self.copyright_match(answer, provenance, visible["judge_identifier"]) else "no"
            else:
                template = self.templates["prompt_contextual" if "context" in tags else "prompt"]
                prompts.append(template.format(behavior=provenance["behavior"], generation=answer,
                                               context=provenance["context"]))
                indices.append(i)
        if prompts:
            inputs = move_inputs(dict(self.tokenizer(prompts, padding=True, return_tensors="pt")), self.device)
            with torch.inference_mode():
                completions = self.model.generate(**inputs, do_sample=False, max_new_tokens=1,
                                                 use_cache=True, logits_to_keep=1)
            for i, raw in zip(indices, self.tokenizer.batch_decode(
                    completions[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)):
                results[i] = raw.strip().lower()
        return [{"status": "ok" if r in ("yes", "no") else "unparseable", "response_candidate": r,
                 "harmful": None, "score": None} for r in results]
