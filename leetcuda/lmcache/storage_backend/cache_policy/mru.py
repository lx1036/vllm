from collections import OrderedDict
from typing import Any

from leetcuda.lmcache.log import init_logger
from leetcuda.lmcache.storage_backend.cache_policy import BaseCachePolicy
from leetcuda.lmcache.types import CacheEngineKey

# MRU(Most Recently Used)

logger = init_logger(__name__)

class MRUCachePolicy(BaseCachePolicy[OrderedDict[CacheEngineKey, Any]]):
    """
    MRU cache policy.
    """

    def __init__(self):
        logger.info("Initializing MRUCachePolicy")

    def init_mutable_mapping(self) -> OrderedDict[CacheEngineKey, Any]:
        return OrderedDict()

    def update_on_hit(self, key: CacheEngineKey, cache_dict: OrderedDict[CacheEngineKey, Any]) -> None:
        # since MRU evicts from the back, the logic is same as LRU.
        cache_dict.move_to_end(key, last=True)

    def update_on_put(self, key: CacheEngineKey) -> None:
        # No action needed for MRU on put, as the key is already at the back.
        pass

    def update_on_force_evict(self, key: CacheEngineKey) -> None:
        pass

    def get_evict_candidates(self, cache_dict: OrderedDict[CacheEngineKey, Any], num_candidates: int = 1) -> list[CacheEngineKey]:
        evict_keys = []
        # Since the most recent object is at the end, we reverse the order here
        for key, cache in reversed(cache_dict.items()):
            if not cache.can_evict:
                continue
            evict_keys.append(key)
            if len(evict_keys) == num_candidates:
                break

        return evict_keys
