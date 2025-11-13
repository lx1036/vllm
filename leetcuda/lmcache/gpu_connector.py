import abc
from typing import List, Optional

import torch

from leetcuda.lmcache.memory_management import MemoryObj, MemoryFormat
from leetcuda.lmcache.utils import _lmcache_nvtx_annotate


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

    @abc.abstractmethod
    def from_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """
        Load the data from a GPU buffer into the memory object.

        :param memory_obj:
        :param start:
        :param end:
        :param kwargs:
        :return:
        """
        raise NotImplementedError


class VLLMPagedMemGPUConnectorV2(GPUConnectorInterface):

    """
    The GPU KV cache should be a nested tuple of K and V tensors.
    More specifically, we have:
    - GPUTensor = Tuple[KVLayer, ...]
    - KVLayer = Tuple[Tensor, Tensor]
    - Tensor: [num_blocks, block_size, num_heads, head_size]

    It will produce / consume memory object with KV_BLOB format
    """

    def __init__(self, hidden_dim_size: int, num_layers: int, use_gpu: bool = False, **kwargs):
        """
        If use_gpu is true, it will create a gpu intermediate buffer. In this
        case, it requires the following kwargs:
        - chunk_size: The MAX size of the chunk to be copied to GPU.
        - dtype: The data type of the intermediate buffer.
        """

        self.hidden_dim_size = hidden_dim_size
        self.num_layers = num_layers



        self.gpu_buffer: Optional[torch.Tensor] = None
        if use_gpu:
            assert "chunk_size" in kwargs, "chunk_size should be provided to create a GPU buffer."
            assert "dtype" in kwargs, "dtype should be provided to create a GPU buffer."
            assert "device" in kwargs, "device should be provided to create a GPU buffer."
            shape = self.get_shape(kwargs["chunk_size"])
            self.gpu_buffer = torch.empty(shape, dtype=kwargs["dtype"], device=kwargs["device"])

    def get_shape(self, num_tokens: int) -> torch.Size:
        return torch.Size([2, self.num_layers, num_tokens, self.hidden_dim_size])


    @_lmcache_nvtx_annotate
    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        assert memory_obj.tensor is not None

        if memory_obj.metadata.fmt != MemoryFormat.KV_BLOB:
            raise ValueError("The memory object should be in KV_BLOB format in order to be processed by VLLMPagedMemGPUConnector")

        if "kvcaches" not in kwargs:
            raise ValueError("'kvcaches' should be provided in kwargs.")

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        kvcaches: List[torch.Tensor] = kwargs["kvcaches"]
        slot_mapping: torch.Tensor = kwargs["slot_mapping"]

        lmc_ops.multi_layer_kv_transfer(memory_obj.tensor, self.kv_cache_pointers, slot_mapping[start:end], kvcaches[0].device, self.page_buffer_size, False)







    @_lmcache_nvtx_annotate
    def from_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        assert memory_obj.tensor is not None

        if "kvcaches" not in kwargs:
            raise ValueError("'kvcaches' should be provided in kwargs.")

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        kvcaches: List[torch.Tensor] = kwargs["kvcaches"]
        slot_mapping: torch.Tensor = kwargs["slot_mapping"]

        if self.gpu_buffer is None or end - start != self.gpu_buffer.shape[2]:
            lmc_ops.multi_layer_kv_transfer(memory_obj.tensor, self.kv_cache_pointers, slot_mapping[start:end], kvcaches[0].device, self.page_buffer_size, True)
        else:
            # kvcaches -> gpu_buffer -> memobj
            assert self.gpu_buffer.device == kvcaches[0].device
            tmp_gpu_buffer = self.gpu_buffer[:, :, :end - start, :]
            lmc_ops.multi_layer_kv_transfer(tmp_gpu_buffer, self.kv_cache_pointers, slot_mapping[start:end], kvcaches[0].device, self.page_buffer_size, True)
            memory_obj.tensor.copy_(tmp_gpu_buffer, non_blocking=True)





class VLLMNestedTupleGPUConnector(GPUConnectorInterface):

    """
    The GPU KV cache should be a nested tuple of K and V tensors.
    More specifically, we have:
    - GPUTensor = Tuple[KVLayer, ...]
    - KVLayer = Tuple[Tensor, Tensor]
    - Tensor: [num_tokens, ...]

    The token dimension is specified by `token_dim` when constructing the
    connector.

    It will produce / consume memory object with KV_BLOB format
    """

    def __init__(self, hidden_dim_size: int, num_layers: int):
        """
        :param int gpu_token_dim: The token dimension of the GPU KV cache in
            the nested tuple.
        """
        self.hidden_dim_size = hidden_dim_size
        self.num_layers = num_layers
