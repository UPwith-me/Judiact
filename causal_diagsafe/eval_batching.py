"""Shape-bucketed evaluation batches, sized by measured throughput and CUDA memory."""

import math
import time
from collections import defaultdict
from functools import lru_cache

import torch
from PIL import Image, UnidentifiedImageError


@lru_cache(maxsize=2048)
def image_area(path):
    with Image.open(path) as image:
        return image.width * image.height


def visible_work(visible, image_max_pixels=None, include_policy=True):
    text = sum(len(visible.get(key) or "") for key in ("query", "ocr_text", "response_candidate"))
    if include_policy:
        text += sum(len(card["rule_text"]) for card in visible.get("policy_deck", []))
    pixels = sum(min(image_area(entry["uri"]), image_max_pixels) if image_max_pixels else image_area(entry["uri"])
                 for entry in visible["images"])
    # This is only a sorting estimate, never a tokenizer, resize rule, or input truncation.
    return max(1, text // 3 + pixels // 784)


def completion_tokens(tokens, eos_token_id):
    eos = {eos_token_id} if isinstance(eos_token_id, int) else set(eos_token_id or [])
    for index, token in enumerate(tokens.tolist()):
        if token in eos:
            return tokens[:index + 1]
    return tokens


class EvalBatcher:
    def __init__(self, devices=None, max_batch_size=512, initial_batch_size=1,
                 generation_tokens=0, kv_bytes_per_token=0, image_max_pixels=None, include_policy=True):
        self.devices = list(devices or [torch.cuda.current_device()]) if torch.cuda.is_available() else []
        self.maximum = max_batch_size
        self.initial = min(initial_batch_size, max_batch_size)
        self.generation_tokens, self.kv_bytes_per_token = generation_tokens, kv_bytes_per_token
        self.image_max_pixels, self.include_policy = image_max_pixels, include_policy
        self.plans, self.measurements = {}, []

    def run(self, entries, forward, on_error=None):
        """Yield (original index, result); the caller restores request order and persists each result."""
        buckets = defaultdict(list)
        for index, entry in enumerate(entries):
            try:
                work = visible_work(entry["visible"], self.image_max_pixels, self.include_policy)
            except (FileNotFoundError, UnidentifiedImageError) as exc:
                if on_error is None:
                    raise
                yield index, on_error(entry, exc)
                continue
            key = (entry.get("batch_key", ""), len(entry["visible"]["images"]), int(math.log2(work)))
            buckets[key].append((work, index, entry))
        for key, items in buckets.items():
            items.sort(key=lambda item: item[0], reverse=True)
            plan = self.plans.setdefault(key, {"size": self.initial, "best_size": self.initial,
                "best_rate": 0., "misses": 0, "settled": False})
            offset = 0
            while offset < len(items):
                selected = items[offset:offset + plan["size"]]
                for device in self.devices:
                    torch.cuda.synchronize(device)
                    torch.cuda.reset_peak_memory_stats(device)
                baseline = [torch.cuda.memory_allocated(device) for device in self.devices]
                # Cached allocator blocks are reusable, unlike another process's allocations.
                capacity = [torch.cuda.mem_get_info(device)[0] + torch.cuda.memory_reserved(device) - 3 * 1024**3
                            for device in self.devices]
                started = time.perf_counter()
                failure = None
                try:
                    values = forward([item[2] for item in selected])
                except (torch.cuda.OutOfMemoryError, FileNotFoundError, UnidentifiedImageError) as exc:
                    failure = (type(exc), str(exc))
                except RuntimeError as exc:
                    # Retry the observed Qwen CUDA indexing overflow without changing any input.
                    if str(exc) != "numel: integer multiplication overflow" or len(selected) == 1:
                        raise
                    failure = (type(exc), str(exc))
                if failure is not None:
                    # Release the failed forward's traceback before freeing its tensors.
                    for device in self.devices:
                        with torch.cuda.device(device):
                            torch.cuda.empty_cache()
                    if len(selected) > 1:
                        plan["size"] = max(1, len(selected) // 2)
                        plan["settled"] = True
                        continue
                    if on_error is None:
                        raise failure[0](failure[1])
                    values = [on_error(selected[0][2], failure[0](failure[1]))]
                for device in self.devices:
                    torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - started
                peaks = [torch.cuda.max_memory_allocated(device) for device in self.devices]
                size, rate = len(selected), len(selected) / max(elapsed, 1e-8)
                self.measurements.append({"bucket": str(key), "batch_size": size,
                    "seconds": elapsed, "records_per_second": rate,
                    "peak_allocated_GiB": [peak / 1024**3 for peak in peaks],
                    "error": failure[0].__name__ if failure else None})
                if size == plan["size"] and failure is None and not plan["settled"]:
                    if rate >= plan["best_rate"]:
                        plan["best_rate"], plan["best_size"] = rate, size
                        plan["misses"] = 0
                    else:
                        # One noisy measurement must not prematurely end batch exploration.
                        plan["misses"] += 1
                        plan["settled"] = plan["misses"] >= 2
                    next_size = min(size * 2, self.maximum)
                    emitted = max((value.get("output_tokens", 0) for value in values), default=0)
                    remaining_kv = max(0, self.generation_tokens - emitted) * self.kv_bytes_per_token * next_size
                    fits = all(base + (peak - base) * next_size / size + remaining_kv < limit
                               for base, peak, limit in zip(baseline, peaks, capacity))
                    if not plan["settled"] and next_size > size and fits:
                        plan["size"] = next_size
                    else:
                        plan["size"], plan["settled"] = plan["best_size"], True
                if len(values) != size:
                    raise ValueError("Batched predictions no longer align with their input records")
                for (_, index, _), value in zip(selected, values):
                    yield index, value
                offset += size

    def summary(self):
        return {"policy": "shape buckets; measured throughput; two consecutive non-improvements; 3 GiB physical headroom; no input truncation",
                "max_batch_size": self.maximum, "plans": {str(k): v for k, v in self.plans.items()},
                "measurements": self.measurements}


def kv_bytes_per_token(model):
    config = getattr(model.config, "text_config", model.config)
    heads = config.num_attention_heads
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // heads
    return 2 * config.num_hidden_layers * getattr(config, "num_key_value_heads", heads) \
        * head_dim * 2
