import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

from leetcuda.lmcache.memory_management import MemoryFormat
from leetcuda.lmcache.server.server_storage_backend import LMSBackendInterface

import torch


# TODO(Jiayi): Maybe move the memory management in remote
# cache server to `memory_management.py` as well.
@dataclass
class LMSMemoryObj:
    data: bytearray
    length: int
    fmt: MemoryFormat
    dtype: Optional[torch.dtype]
    shape: torch.Size



class LocalBackend(LMSBackendInterface):
    """
    Cache engine for storing the KV cache of the tokens in the local cpu/gpu
    memory.
    """

    def __init__(self, ):
        """
        Throws:
            RuntimeError if the loaded configuration does not match the current
            configuration
        """
        super().__init__()
        self.dict: OrderedDict[str, bytearray] = OrderedDict()
        self.lock = threading.Lock()



    def put(self, key: str, kv_chunk_bytes: bytearray, blocking: bool = True) -> None:
        with self.lock:
            self.dict[client_meta.key] = LMSMemoryObj(
                kv_chunk_bytes,
                client_meta.length,
                client_meta.fmt,
                client_meta.dtype,
                client_meta.shape,
            )




# TODO(Jiayi): need to optimize disk loading
# current impl. with "naive open read/write" might not be efficient
# (better than torch.load)
class LocalDiskBackend(LMSBackendInterface):




