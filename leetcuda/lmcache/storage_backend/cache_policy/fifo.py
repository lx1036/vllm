from typing import Any

from leetcuda.lmcache.log import init_logger
from leetcuda.lmcache.storage_backend.cache_policy import BaseCachePolicy
from leetcuda.lmcache.types import CacheEngineKey

# FIFO(First-In-First-Out)

logger = init_logger(__name__)


class FIFOCachePolicy(BaseCachePolicy[dict[CacheEngineKey, Any]]):
    """
    FIFO cache policy.
    """

    def __init__(self):
        logger.info("Initializing FIFOCachePolicy")

    def init_mutable_mapping(self) -> dict[CacheEngineKey, Any]:
        # NOTE: python dict maintains insertion order.
        return {}

    def update_on_hit(self, key: CacheEngineKey, cache_dict: dict[CacheEngineKey, Any]) -> None:
        pass

    def update_on_put(self, key: CacheEngineKey) -> None:
        pass

    def update_on_force_evict(self, key: CacheEngineKey) -> None:
        pass

    def get_evict_candidates(self, cache_dict: dict[CacheEngineKey, Any], num_candidates: int = 1) -> list[CacheEngineKey]:
        evict_keys = []
        for key, cache in cache_dict.items():
            if not cache.can_evict:
                continue
            evict_keys.append(key)
            if len(evict_keys) == num_candidates:
                break

        return evict_keys

def test_get_evict_candidates():
    policy = FIFOCachePolicy()

    key1 = CacheEngineKey("vllm", "test_model", 3, 123, hash("test_key"))
    key2 = CacheEngineKey("abc", "test_model", 3, 123, hash("test_key"))

    cache_dict = policy.init_mutable_mapping()
    # cache_dict[key1] =

    cache_dict: dict[CacheEngineKey, Any] = {
        key1: {
            "can_evict": True,
        },
        key2: {
            "can_evict": True,
        },
    }
    evict_candidates = policy.get_evict_candidates(cache_dict)
    logger.info(f"evict_candidates: {evict_candidates}")

