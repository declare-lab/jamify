FILLER_REGISTRY = {}

def register_filler(name):
    def wrapper(fn):
        FILLER_REGISTRY[name] = fn
        return fn
    return wrapper

def get_filler(name):
    return FILLER_REGISTRY[name]

from . import fillers

__all__ = [
    'fillers',
    'get_filler',
]
