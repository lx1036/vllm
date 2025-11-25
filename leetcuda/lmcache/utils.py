import asyncio
import hashlib
import os
import threading
from dataclasses import dataclass
from typing import Optional

from nvtx import annotate  # type: ignore
import torch

from leetcuda.lmcache.config import LMCacheEngineConfig
from leetcuda.lmcache.log import init_logger

##### NVTX annotation #####
_NVTX_COLORS = ["green", "blue", "purple", "rapids"]

ENGINE_NAME = "vllm-instance"

logger = init_logger(__name__)


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
    request_configs: Optional[dict] = None

    def to_string(self):
        return f"{self.fmt}@{self.model_name}@{self.world_size}@{self.worker_id}@{self.chunk_hash}"

    @staticmethod
    def from_string(s):
        parts = s.split("@")
        if len(parts) != 5:
            raise ValueError(f"Invalid key string: {s}")
        return CacheEngineKey(parts[0], parts[1], int(parts[2]), int(parts[3]), parts[4])


@dataclass
class DiskCacheMetadata:
    path: str
    size: int  # in bytes
    shape: Optional[torch.Size] = None
    dtype: Optional[torch.dtype] = None

def lmcache_get_config() -> LMCacheEngineConfig:
    config_file = os.environ["LMCACHE_CONFIG_FILE"]
    logger.info(f"Loading LMCache config file {config_file}")
    config = LMCacheEngineConfig.from_file(config_file)

    return config


##### Threading related #####
def thread_safe(func):
    lock = threading.Lock()

    def wrapper(*args, **kwargs):
        with lock:
            return func(*args, **kwargs)

    return wrapper


def init_asyncio_loop():
    async_loop = asyncio.new_event_loop()
    async_thread = threading.Thread(target=async_loop.run_forever)
    async_thread.start()
    return async_loop, async_thread

def close_asyncio_loop(async_loop, async_thread):
    if async_loop.is_running():
        async_loop.call_soon_threadsafe(async_loop.stop)
    if async_thread.is_alive():
        async_thread.join()

def dumb_cache_engine_key():
    return CacheEngineKey("vllm", "test_model", 3, 123, "hash")

def check_mem_obj_equal(left, right, offset=0):
    """
    check whether two memory objects are the same
    """
    for left_mem_obj, right_mem_obj in zip(left, right):
        left_kv, right_kv = left_mem_obj.tensor, right_mem_obj.tensor
        left_k, left_v = left_kv[0], left_kv[1]
        right_k, right_v = right_kv[0], right_kv[1]
        right_k = right_k.to(left_k.device)
        right_v = right_v.to(left_v.device)

        assert len(left_k.shape) == 3
        assert len(left_v.shape) == 3
        assert len(right_k.shape) == 3
        assert len(right_v.shape) == 3

        assert (left_k[:, :, :] == right_k[:, :, :]).all()
        assert (left_v[:, :, :] == right_v[:, :, :]).all()



