import abc
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union, Tuple

import sortedcontainers
import torch

from leetcuda.lmcache.log import init_logger
from leetcuda.lmcache.observability import LMCStatsMonitor

logger = init_logger(__name__)


class MemoryFormat(Enum):
    UNDEFINED = 0
    """[2, num_layers, num_tokens, hidden_dim]
    """
    KV_BLOB = 1
    """Compressed binary array format
    """
    BINARY = 2

    BINARY_BUFFER = 3


@dataclass
class MemoryObjMetadata:
    # The 'logical' shape of the tensor
    shape: torch.Size

    # The 'logical' dtype of the tensor
    dtype: Optional[torch.dtype]

    # The 'physical address' of the tensor
    address: int

    # The 'physical size' in bytes of the allocated memory
    phy_size: int

    # Reference count
    ref_count: int

    # The 'logical' format of the tensor
    fmt: MemoryFormat = MemoryFormat.UNDEFINED

class MemoryObj(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def invalidate(self):
        """
        Invalidate the MemoryObj.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def byte_array(self) -> bytes:
        """
        Get the byte array from the MemoryObj.
        """
        raise NotImplementedError


    @property
    @abc.abstractmethod
    def tensor(self) -> Optional[torch.Tensor]:
        """
        Get the tensor from the MemoryObj.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def metadata(self) -> MemoryObjMetadata:
        """
        Get the metada of the MemoryObj.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_size(self) -> int:
        """
        Get the size of the MemoryObj in bytes.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_physical_size(self) -> int:
        """
        Get the physical size of the MemoryObj in bytes.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_shape(self) -> torch.Size:
        """
        Get the shape of the MemoryObj.
        """
        raise NotImplementedError

    def get_dtype(self) -> Optional[torch.dtype]:
        """
        Get the dtype of the MemoryObj.
        """
        return None

    @abc.abstractmethod
    def get_memory_format(self) -> MemoryFormat:
        """
        Get the memory format of the MemoryObj.
        """
        raise NotImplementedError

@dataclass
class FreeBlock:
    """Metadata class used by the memory allocators
    """
    start: int
    size: int

    def can_be_coalesced(self, succ: "FreeBlock") -> bool:
        return self.start + self.size == succ.start


class TensorMemoryObj(MemoryObj):
    """
    Wraps a raw flat tensor with some metadata
    """

    def __init__(self, raw_data: torch.Tensor, metadata: MemoryObjMetadata):
        self.raw_data = raw_data
        self.meta = metadata
        self.valid = True


class MemoryAllocatorInterface(metaclass=abc.ABCMeta):


    @abc.abstractmethod
    def allocate(
            self,
            shape: Union[torch.Size, Tuple[int, ...]],
            dtype: Optional[torch.dtype],
            fmt: MemoryFormat = MemoryFormat.UNDEFINED,
    ) -> Optional[MemoryObj]:
        """
        Allocates the memory to hold a tensor of the given shape.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def ref_count_up(self, memory_obj: MemoryObj):
        """
        Increase ref count for the given MemoryObj.

        :param MemoryObj memory_obj.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def ref_count_down(self, memory_obj: MemoryObj):
        """
        Decrease ref count for the given MemoryObj.

        :param MemoryObj memory_obj.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_ref_count(self, memory_obj: MemoryObj):
        """
        Get ref count for the given MemoryObj.

        :param MemoryObj memory_obj.
        """
        raise NotImplementedError




class TensorMemoryAllocator(MemoryAllocatorInterface):


    def __init__(self, tensor: torch.Tensor):
        self.buffer = tensor.view(torch.uint8).flatten()

        self.explicit_list = sortedcontainers.SortedList(key=lambda x: x.start)
        self.explicit_list.add(FreeBlock(start=0, size=self.buffer.numel()))

        # For debugging purposes
        self.num_active_allocations = 0
        self.total_allocated_size = 0

        self.stats_monitor = LMCStatsMonitor.GetOrCreate()



    def allocate(self, shape: Union[torch.Size, Tuple[int, ...]], dtype: Optional[torch.dtype], fmt: MemoryFormat = MemoryFormat.KV_BLOB) -> Optional[TensorMemoryObj]:

        # Calculate the size of the tensor
        raw_size = TensorMemoryAllocator._Compute_raw_size(shape, dtype)
        aligned_size = TensorMemoryAllocator._Compute_aligned_size(raw_size)

        # Find the first block that fits the shape
        for block in self.explicit_list:
            if block.size >= aligned_size:
                break
        else:
            logger.warning(f"Failed to allocate memory for tensor({shape}, {dtype}) because no memory is available")
            return None

        # Do not add the block back if `block.size == aligned_size`
        self.explicit_list.remove(block)
        # Update the explicit list
        if block.size > aligned_size:
            self.explicit_list.add(FreeBlock(start=block.start + aligned_size, size=block.size - aligned_size))

        # Update debug status
        self.total_allocated_size += aligned_size
        self.num_active_allocations += 1
        self.stats_monitor.update_local_cache_usage(self.total_allocated_size)

        # Allocate the block
        return TensorMemoryObj(raw_data=self.buffer[block.start:block.start + raw_size], metadata=MemoryObjMetadata(shape, dtype, block.start, aligned_size, 1, fmt))

    def ref_count_up(self, memory_obj: MemoryObj):
        memory_obj.metadata.ref_count += 1

    def ref_count_down(self, memory_obj: MemoryObj):
        memory_obj.metadata.ref_count -= 1
        if memory_obj.metadata.ref_count == 0:
            self.free(memory_obj)

    def get_ref_count(self, memory_obj: MemoryObj):
        return memory_obj.metadata.ref_count

    def free(self, memory_obj: MemoryObj):
        if not memory_obj.is_valid():
            return





class BufferAllocator(MemoryAllocatorInterface):

    def __init__(self, device="cpu"):
        self.device = device




class MixedMemoryAllocator(MemoryAllocatorInterface):

    """
    Allocates (1) memory in the pre-allocated pinned memory.
              (2) byte_array buffer memory.
    """
    def __init__(self, size: int):
        """
        注意 pin_memory=True
        """
        buffer = torch.empty(size, dtype=torch.uint8, pin_memory=True)

        self.pin_allocator = TensorMemoryAllocator(buffer)
        self.buffer_allocator = BufferAllocator("cpu")

        self.host_mem_lock = threading.Lock()



    def allocate(self, shape: Union[torch.Size, Tuple[int, ...]], dtype: Optional[torch.dtype], fmt: MemoryFormat = MemoryFormat.KV_BLOB) -> Optional[MemoryObj]:
        if fmt == MemoryFormat.BINARY_BUFFER:
            return self.buffer_allocator.allocate(shape, dtype, fmt)
        elif fmt == MemoryFormat.KV_BLOB:
            with self.host_mem_lock:
                return self.pin_allocator.allocate(shape, dtype, fmt)
        else:
            raise ValueError(f"Unsupported memory format: {fmt}")


class PinMemoryAllocator(MemoryAllocatorInterface):
    """
    Allocates memory in the pre-allocated pinned memory.
    """

    def __init__(self, size: int):
        """
        :param int size: The size of the pinned memory in bytes.
        """
        buffer = torch.empty(size, dtype=torch.uint8, pin_memory=True)

        self.allocator = TensorMemoryAllocator(buffer)

        self.host_mem_lock = threading.Lock()


    def allocate(self, shape: Union[torch.Size, Tuple[int, ...]], dtype: Optional[torch.dtype], fmt: MemoryFormat = MemoryFormat.KV_BLOB) -> Optional[MemoryObj]:
        with self.host_mem_lock:
            return self.allocator.allocate(shape, dtype, fmt)

    def ref_count_up(self, memory_obj: MemoryObj):
        with self.host_mem_lock:
            self.allocator.ref_count_up(memory_obj)

    def ref_count_down(self, memory_obj: MemoryObj):
        with self.host_mem_lock:
            self.allocator.ref_count_down(memory_obj)

    def get_ref_count(self, memory_obj: MemoryObj):
        with self.host_mem_lock:
            return self.allocator.get_ref_count(memory_obj)
