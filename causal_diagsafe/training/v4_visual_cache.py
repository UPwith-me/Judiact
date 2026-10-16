"""Reuse the deterministic frozen Qwen image/deepstack readout, never language states."""

from collections import OrderedDict
from contextlib import contextmanager

import torch


class FrozenVisualCache:
    def __init__(self, max_images=64):
        self.maximum = max_images
        self.values = OrderedDict()
        self.hits = self.misses = self.visual_calls = 0
        self.verified_core = None

    @contextmanager
    def context(self, model, image_uris):
        if not image_uris:
            yield
            return
        raw = model.module if hasattr(model, "module") else model
        raw = raw.get_base_model() if hasattr(raw, "get_base_model") else raw
        core = raw.model
        if self.verified_core is not core:
            assert not any(p.requires_grad for p in core.visual.parameters()), "Only frozen vision may be reused"
            self.verified_core = core
        original = core.get_image_features

        def read(pixel_values, image_grid_thw):
            sizes = (image_grid_thw.prod(-1) // core.visual.spatial_merge_size**2).tolist()
            keys = [(uri, tuple(grid), str(pixel_values.device), str(core.visual.dtype))
                    for uri, grid in zip(image_uris, image_grid_thw.tolist())]
            assert len(keys) == len(image_uris) == len(sizes)
            if all(key in self.values for key in keys):
                self.hits += len(keys)
                features = [self.values[key] for key in keys]
                for key in keys:
                    self.values.move_to_end(key)
            else:
                self.misses += sum(key not in self.values for key in keys)
                with torch.no_grad():
                    embeds, deepstack = original(pixel_values, image_grid_thw)
                self.visual_calls += 1
                stacks = [torch.split(value.detach(), sizes) for value in deepstack]
                features = [(embed.detach().clone(), tuple(stack[index].clone() for stack in stacks))
                            for index, embed in enumerate(embeds)]
                for key, feature in zip(keys, features):
                    self.values[key] = feature
                    self.values.move_to_end(key)
                while len(self.values) > self.maximum:
                    self.values.popitem(last=False)
            return tuple(feature[0] for feature in features), [
                torch.cat([feature[1][index] for feature in features], 0)
                for index in range(len(features[0][1]))]

        core.get_image_features = read
        try:
            yield
        finally:
            core.get_image_features = original

    def summary(self):
        return {"cached_images": len(self.values), "image_hits": self.hits,
                "image_misses": self.misses, "actual_visual_calls": self.visual_calls}
