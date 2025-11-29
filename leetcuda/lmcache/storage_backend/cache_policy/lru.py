from collections import OrderedDict
from typing import Any

from leetcuda.lmcache.log import init_logger
from leetcuda.lmcache.storage_backend.cache_policy import BaseCachePolicy
from leetcuda.lmcache.types import CacheEngineKey

# LRU(Least Recently Used)

logger = init_logger(__name__)

class LRUCachePolicy(BaseCachePolicy[OrderedDict[CacheEngineKey, Any]]):

    def __init__(self):
        logger.info("Initializing LRUCachePolicy")

    def init_mutable_mapping(self) -> OrderedDict[CacheEngineKey, Any]:
        return OrderedDict()

    def update_on_hit(self, key: CacheEngineKey, cache_dict: OrderedDict[CacheEngineKey, Any]) -> None:
        cache_dict.move_to_end(key)

    def update_on_put(self, key: CacheEngineKey) -> None:
        # No action needed for LRU on put, as the key is already at the end.
        pass

    def update_on_force_evict(self, key: CacheEngineKey) -> None:
        pass

    def get_evict_candidates(self, cache_dict: OrderedDict[CacheEngineKey, Any], num_candidates: int = 1) -> list[CacheEngineKey]:
        evict_keys = []
        for key, cache in cache_dict.items():
            if not cache.can_evict:
                continue
            evict_keys.append(key)
            if len(evict_keys) == num_candidates:
                break

        return evict_keys




