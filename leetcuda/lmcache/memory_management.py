import abc


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
