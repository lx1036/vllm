import threading
from collections import OrderedDict
from typing import Optional, List

from leetcuda.lmcache.protocol import ClientMetaMessage
from leetcuda.lmcache.server.server_storage_backend.abstract_backend import LMSBackendInterface


from leetcuda.lmcache.server.server_storage_backend.utils import LMSMemoryObj
from leetcuda.lmcache.utils import CacheEngineKey, _lmcache_nvtx_annotate


class LocalBackend(LMSBackendInterface):
    """
    Cache engine for storing the KV cache of the tokens in the local cpu/gpu
    memory.
    """

    def __init__(self):
        """
        Throws:
            RuntimeError if the loaded configuration does not match the current
            configuration
        """
        super().__init__()
        self.dict: OrderedDict[CacheEngineKey, LMSMemoryObj] = OrderedDict()
        self.lock = threading.Lock()


    def put(self, client_meta: ClientMetaMessage, kv_chunk_bytes: bytearray) -> None:
        with self.lock:
            self.dict[client_meta.key] = LMSMemoryObj(
                kv_chunk_bytes,
                client_meta.length,
                client_meta.fmt,
                client_meta.dtype,
                client_meta.shape,
            )

    @_lmcache_nvtx_annotate
    def get(self, key: CacheEngineKey) -> Optional[LMSMemoryObj]:
        with self.lock:
            return self.dict.get(key, None)

    def contains(self, key: CacheEngineKey) -> bool:
        with self.lock:
            return key in self.dict


    def list_keys(self) -> List[CacheEngineKey]:
        with self.lock:
            return list(self.dict.keys())

    def close(self):
        pass


