import abc

import torch





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





class TensorMemoryAllocator(MemoryAllocatorInterface):


    def __init__(self, tensor: torch.Tensor):
        self.buffer = tensor.view(torch.uint8).flatten()


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










