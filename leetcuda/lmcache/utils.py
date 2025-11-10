import hashlib
from dataclasses import dataclass

from nvtx import annotate  # type: ignore


##### NVTX annotation #####
_NVTX_COLORS = ["green", "blue", "purple", "rapids"]

ENGINE_NAME = "vllm-instance"

def _get_color_for_nvtx(name):
    m = hashlib.sha256()
    m.update(name.encode())
    hash_value = int(m.hexdigest(), 16)
    idx = hash_value % len(_NVTX_COLORS)
    return _NVTX_COLORS[idx]


def _lmcache_nvtx_annotate(func, domain="lmcache"):
    """Decorator for applying nvtx annotations to methods in lmcache."""
    return annotate(
        message=func.__qualname__,
        color=_get_color_for_nvtx(func.__qualname__),
        domain=domain,
    )(func)




@dataclass(order=True)
class CacheEngineKey:
    fmt: str
    model_name: str
    world_size: int
    worker_id: int
    chunk_hash: str

