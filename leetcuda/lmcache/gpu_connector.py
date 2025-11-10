import abc

from memory_management import MemoryObj


class GPUConnectorInterface(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """
        Store the data in the memory object into a GPU buffer.

        :param memory_obj: The memory object to be copied into GPU.
        :param start:
        :param end:
        :param kwargs:
        :return:
        """
        raise NotImplementedError



