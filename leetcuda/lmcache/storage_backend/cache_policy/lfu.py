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
        # freq: key
        self.freq_to_keys: SortedDict[int, dict[CacheEngineKey, Any]] = SortedDict()

        # key: freq
        self.key_to_freq: dict[CacheEngineKey, int] = {}

        logger.info("Initializing LFUCachePolicy")

    def init_mutable_mapping(self) -> dict[CacheEngineKey, Any]:
        return {}

    def update_on_hit(self, key: CacheEngineKey, cache_dict: dict[CacheEngineKey, Any]) -> None:
        curr_freq = self.key_to_freq[key]
        self.freq_to_keys[curr_freq].pop(key)
        if not self.freq_to_keys[curr_freq]:
            self.freq_to_keys.pop(curr_freq)

        curr_freq += 1
        self.key_to_freq[key] = curr_freq

        if curr_freq not in self.freq_to_keys:
            self.freq_to_keys[curr_freq] = {key: None}
        else:
            self.freq_to_keys[curr_freq][key] = None

    def update_on_put(self, key: CacheEngineKey) -> None:
        # Initialize the frequency for the new key.
        self.key_to_freq[key] = 1
        if 1 not in self.freq_to_keys:
            self.freq_to_keys[1] = {key: None}
        else:
            self.freq_to_keys[1][key] = None

    def update_on_force_evict(self, key: CacheEngineKey) -> None:
        freq = self.key_to_freq.pop(key, None)
        if not freq:
            return
        self.freq_to_keys[freq].pop(key)
        if not self.freq_to_keys[freq]:
            self.freq_to_keys.pop(freq)

    def get_evict_candidates(self, cache_dict: dict[CacheEngineKey, Any], num_candidates: int = 1) -> list[CacheEngineKey]:
        evict_keys = []
        evict_freqs = []
        for curr_min_freq, fifo_keys in self.freq_to_keys.items():
            for key in fifo_keys:
                if not cache_dict[key].can_evict:
                    continue
                evict_keys.append(key)
                evict_freqs.append(curr_min_freq)
                self.key_to_freq.pop(key)
                if len(evict_keys) == num_candidates:
                    break

            if len(evict_keys) == num_candidates:
                break

        for freq, key in zip(evict_freqs, evict_keys, strict=False):
            self.freq_to_keys[freq].pop(key)
            if not self.freq_to_keys[freq]:
                self.freq_to_keys.pop(freq)

        return evict_keys

def test_sorted_dict():
    result = {'alpha': 1, 'beta': 2}
    sorted_dict = SortedDict({'beta': 2, 'alpha': 1})
    assert result==sorted_dict



