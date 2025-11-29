from typing import Any

from sortedcontainers import SortedDict

from leetcuda.lmcache.log import init_logger
from leetcuda.lmcache.storage_backend.cache_policy import BaseCachePolicy
from leetcuda.lmcache.types import CacheEngineKey


# LFU(Least Frequently Used)


logger = init_logger(__name__)

class LFUCachePolicy(BaseCachePolicy[dict[CacheEngineKey, Any]]):
    """
    LFU cache policy.
    """
    # NOTE(Jiayi): We use `sorted dict` + `bucket` to implement LFU.
    # NOTE(Jiayi): We use FIFO for entries with the same frequency.
    def __init__(self):
        # TODO(Jiayi): `SortedDict` is log(N).
        # A way to make it O(1) is to use a dict and keep track min freuency.
        # However, this requires us keep another data structures to keep track
        # of the pinned keys.
        self.freq_to_keys = SortedDict()

        # TODO(Jiayi): We can optimize this a bit by using `key_to_val_freq`
        self.key_to_freq = {}

        logger.info("Initializing LFUCachePolicy")

    def init_mutable_mapping(self) -> dict[CacheEngineKey, Any]:
        return {}





def test_sorted_dict():
    result = {'alpha': 1, 'beta': 2}
    sorted_dict = SortedDict({'beta': 2, 'alpha': 1})
    assert result==sorted_dict



