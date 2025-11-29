import asyncio
from concurrent.futures import Future
from typing import Optional

from leetcuda.lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata
from leetcuda.lmcache.lookup_server.abstract_server import LookupServerInterface
from leetcuda.lmcache.memory_management import MemoryAllocatorInterface, MemoryObj
from leetcuda.lmcache.storage_backend.base_backend import StorageBackendInterface
from leetcuda.lmcache.storage_backend.connector import CreateConnector
from leetcuda.lmcache.utils import CacheEngineKey


class RemoteBackend(StorageBackendInterface):

    def __init__(self,
                 config: LMCacheEngineConfig,
                 metadata: LMCacheEngineMetadata,
                 loop: asyncio.AbstractEventLoop,
                 memory_allocator: MemoryAllocatorInterface,
                 dst_device: str = "cuda",
                 lookup_server: Optional[LookupServerInterface] = None,):

        self.connection = CreateConnector(config.remote_url, loop, memory_allocator)



    def submit_put_task(self, key: CacheEngineKey, memory_obj: MemoryObj,) -> Optional[Future]:

        compressed_memory_obj = self.serializer.serialize(memory_obj)

        future = asyncio.run_coroutine_threadsafe(self.connection.put(key, compressed_memory_obj), self.loop)
