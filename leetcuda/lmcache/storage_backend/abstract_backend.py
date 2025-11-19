import abc
from typing import Optional
from concurrent.futures import Future

from leetcuda.lmcache.memory_management import MemoryObj
from leetcuda.lmcache.utils import CacheEngineKey

import torch


class StorageBackendInterface(metaclass=abc.ABCMeta):
    def __init__(self, dst_device: str = "cuda"):
        try:
            torch.device(dst_device)
        except RuntimeError:
            raise

        self.dst_device = dst_device




    @abc.abstractmethod
    def submit_put_task(self, key: CacheEngineKey, obj: MemoryObj) -> Optional[Future]:
        raise NotImplementedError



