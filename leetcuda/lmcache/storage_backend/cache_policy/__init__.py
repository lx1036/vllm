from typing import Type, Dict

from leetcuda.lmcache.storage_backend.cache_policy.base_policy import BaseCachePolicy

POLICY_MAPPING: Dict[str, Type[BaseCachePolicy]] = {
    "LRU": LRUCachePolicy,
    "LFU": LFUCachePolicy,
    "FIFO": FIFOCachePolicy,
    "MRU": MRUCachePolicy,
}

def get_cache_policy(policy_name: str) -> BaseCachePolicy:
    if not policy_name:
        raise ValueError("Cache policy name cannot be empty")

    upper_policy_name = policy_name.upper()
    try:
        return POLICY_MAPPING[upper_policy_name]()
    except KeyError:
        raise ValueError(f"Unknown cache policy: {upper_policy_name}. Supported policies are: {list(POLICY_MAPPING.keys())}")
