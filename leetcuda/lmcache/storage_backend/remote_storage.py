import asyncio
import time
from concurrent.futures import Future
from typing import Optional, List

from leetcuda.lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata
from leetcuda.lmcache.log import init_logger
from leetcuda.lmcache.lookup_server.abstract_server import LookupServerInterface
from leetcuda.lmcache.memory_management import MemoryAllocatorInterface, MemoryObj
from leetcuda.lmcache.storage_backend.base_backend import StorageBackendInterface
from leetcuda.lmcache.storage_backend.connector import CreateConnector
from leetcuda.lmcache.utils import CacheEngineKey


logger = init_logger(__name__)


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


    def batched_get_blocking(self, keys: List[CacheEngineKey]) -> List[Optional[MemoryObj]]:
        if self.connection is None:
            logger.warning("Connection is None in batched_get_blocking, returning None")
            return [None] * len(keys)


        # batched get
        if self.connection.support_batched_get():
            future = asyncio.run_coroutine_threadsafe(self.connection.batched_get(new_keys), self.loop)
            try:
                memory_objs = future.result(self.blocking_timeout_secs)
            except Exception as e:
                if isinstance(e, TimeoutError):
                    logger.warning("batched get blocking timeout, trigger cancel the future task")
                    future.cancel()
                else:
                    logger.warning(f"Error occurred in batched_get_blocking: {e}, returning None list")
                return [None] * len(keys)
        else:
            futures = [
                asyncio.run_coroutine_threadsafe(self.connection.get(key), self.loop)
                for key in new_keys
            ]
            memory_objs = []
            failed = False
            for fut in futures:
                if not failed:
                    try:
                        memory_obj = fut.result(self.blocking_timeout_secs)
                    except Exception as e:
                        failed = True
                        if isinstance(e, TimeoutError):
                            logger.warning("get blocking timeout, trigger cancel the future task")
                            fut.cancel()
                        else:
                            logger.warning(f"Error occurred in get_blocking: {e}, returning None")
                        memory_obj = None
                    memory_objs.append(memory_obj)
                else:
                    memory_objs.append(None) # 失败的，则后面future全部补位 None
                    fut.cancel()

        # 反序列化
        decompressed_memory_objs: list[Optional[MemoryObj]] = []
        for memory_obj in memory_objs:
            if memory_obj is None:
                decompressed_memory_objs.append(None)
            else:
                decompressed_memory_objs.append(self.deserializer.deserialize(memory_obj))

        assert len(decompressed_memory_objs) == len(keys), f"keys length: {len(keys)}, decompressed memory objs length: {len(decompressed_memory_objs)}"
        return decompressed_memory_objs



