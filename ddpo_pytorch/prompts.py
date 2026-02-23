from importlib import resources
import os
import functools

ASSETS_PATH = resources.files("ddpo_pytorch.assets")


@functools.cache
def _load_lines(path):
    if not os.path.exists(path):
        newpath = ASSETS_PATH.joinpath(path)
    if not os.path.exists(newpath):
        raise FileNotFoundError(f"Could not find {path} or ddpo_pytorch.assets/{path}")
    path = newpath
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]


def from_file_all(path, low=None, high=None):
    prompts = _load_lines(path)[low:high]
    return prompts, {}

def simple_animals():
    return from_file_all("simple_animals.txt")
