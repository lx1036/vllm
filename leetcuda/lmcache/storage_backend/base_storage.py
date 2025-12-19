import abc
from typing import Optional, List
from concurrent.futures import Future

from leetcuda.lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata
from leetcuda.lmcache.memory_management import MemoryObj, MemoryFormat, MemoryAllocatorInterface
from leetcuda.lmcache.utils import CacheEngineKey

import torch


class StorageBackendInterface(metaclass=abc.ABCMeta):
    def __init__(self, dst_device: str = "cuda"):
        """
        Initialize the storage backend.

        :param dst_device: the device where the blocking retrieved KV is stored,
            could be either "cpu", "cuda", or "cuda:0", "cuda:1", etc.

        :raise: RuntimeError if the device is not valid
        """
        try:
            torch.device(dst_device)
        except RuntimeError:
            raise

        self.dst_device = dst_device




    @abc.abstractmethod
    def submit_put_task(self, key: CacheEngineKey, obj: MemoryObj) -> Optional[Future]:
        raise NotImplementedError



class AllocatorBackendInterface(StorageBackendInterface):
    """
    AllocatorBackendInterface extends the StorageBackendInterface with
    the ability to actively allocate the memory objects.
    """

    @abc.abstractmethod
    def initialize_allocator(self, config: LMCacheEngineConfig, metadata: LMCacheEngineMetadata) -> MemoryAllocatorInterface:
        raise NotImplementedError

    @abc.abstractmethod
    def get_memory_allocator(self) -> MemoryAllocatorInterface:
        raise NotImplementedError

    def batched_get_blocking(self, keys: List[CacheEngineKey]) -> List[Optional[MemoryObj]]:
        """
        A blocking function to get the kv cache from the storage backend.

        :param List[CacheEngineKey] keys: The keys of the MemoryObjs.

        :return: a list of memory objects.
        """
        mem_objs = []
        for key in keys:
            mem_objs.append(self.get_blocking(key))
        return mem_objs

    @abc.abstractmethod
    def get_blocking(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """
        A blocking function to get the kv cache from the storage backend.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def allocate(
        self,
        shape: torch.Size,
        dtype: torch.dtype,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        eviction: bool = True,
        busy_loop: bool = True,
    ) -> Optional[MemoryObj]:
        raise NotImplementedError

    @abc.abstractmethod
    def batched_allocate(
        self,
        shape: torch.Size,
        dtype: torch.dtype,
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        eviction: bool = True,
        busy_loop: bool = True,
    ) -> Optional[list[MemoryObj]]:
        raise NotImplementedError

    def calculate_chunk_budget(self) -> int:
        raise NotImplementedError
