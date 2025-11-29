import abc
from typing import Optional
from concurrent.futures import Future

from leetcuda.lmcache.memory_management import MemoryObj
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
    def initialize_allocator(
            self, config: LMCacheEngineConfig, metadata: LMCacheEngineMetadata
    ) -> MemoryAllocatorInterface:
        """
        Create the correct memory allocator for the current storage backend

        Args:
            config: The cache engine config
            metadata: the cache engine metadata

        Returns:
            The memory allocator for this storage backend
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_memory_allocator(self) -> MemoryAllocatorInterface:
        """
        Returns:
            The underlying memory allocator
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
        """
        Allocates memory in the backend to hold a tensor of the given shape.

        :param torch.Size shape: The shape of the tensor to allocate.
        :param torch.dtype dtype: The dtype of the tensor to allocate.
        :param MemoryFormat fmt: The format of the memory to allocate.
        :param bool eviction: whether to enable eviction when allocating.
        :param bool busy_loop: whether to enable a busy loop to wait
            for in-progress store operations to finish and release the
            memory space for retrieve.

        :return: A MemoryObj wrapping the allocated memory. Returns
            None if the allocation failed.

        :rtype: Optional[MemoryObj]
        """
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
        """
        Allocates memory in the backend to hold a tensor of the given shape
        in a batched manner. The allocated memory objects will have the same
        shape, dtype, and format.

        :param torch.Size shape: The shape of the tensor to allocate.
        :param torch.dtype dtype: The dtype of the tensor to allocate.
        :param int batch_size: The number of memory objects to allocate.
        :param MemoryFormat fmt: The format of the memory to allocate.
        :param bool eviction: whether to enable eviction when allocating.
        :param bool busy_loop: whether to enable a busy loop to wait
            for in-progress store operations to finish and release the
            memory space for retrieve.

        :return: A MemoryObj wrapping the allocated memory. Returns
            None if the allocation failed.

        :rtype: Optional[MemoryObj]
        """
        raise NotImplementedError

    def calculate_chunk_budget(self) -> int:
        """
        Calculate the chunk budget for the allocator backend.
        """
        raise NotImplementedError
