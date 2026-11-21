"""Load the original RGB pixels for the model processor."""

from functools import lru_cache
from PIL import Image


@lru_cache(maxsize=64)
def open_rgb(uri):
    with Image.open(uri) as image:
        return image.convert("RGB")
