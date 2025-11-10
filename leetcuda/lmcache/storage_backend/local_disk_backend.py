import asyncio
import os
import threading
from typing import Optional
from concurrent.futures import Future

import aiofiles

from ..memory_management import MemoryObj, MemoryAllocatorInterface

from .abstract_backend import StorageBackendInterface
from ..utils import CacheEngineKey
from ..log import init_logger

logger = init_logger(__name__)


class LocalDiskBackend(StorageBackendInterface):

    def __init__(
            self,
            config: LMCacheEngineConfig,
            loop: asyncio.AbstractEventLoop,
            memory_allocator: MemoryAllocatorInterface,
            dst_device: str = "cuda",
            lmcache_worker: Optional["LMCacheWorker"] = None,
            lookup_server: Optional[LookupServerInterface] = None,
    ):

        self.disk_lock = threading.Lock()

        assert config.local_disk is not None
        self.path: str = config.local_disk
        if not os.path.exists(self.path):
            os.makedirs(self.path)
            logger.info(f"Created local disk cache directory: {self.path}")


        self.put_tasks: List[CacheEngineKey] = []









    def submit_put_task(
            self,
            key: CacheEngineKey,
            memory_obj: MemoryObj,
    ) -> Optional[Future]:



        self.disk_lock.acquire()
        self.put_tasks.append(key)
        self.disk_lock.release()

        future = asyncio.run_coroutine_threadsafe(self.async_save_bytes_to_disk(key, memory_obj), self.loop)
        return future


    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    async def async_save_bytes_to_disk(
            self,
            key: CacheEngineKey,
            memory_obj: MemoryObj,
    ) -> None:
        """
        Convert KV to bytes and async store bytes to disk.
        """


        byte_array = memory_obj.byte_array

        path = self._key_to_path(key)

        async with aiofiles.open(path, 'wb') as f:
            await f.write(byte_array)

        self.insert_key(key, memory_obj)
        self.memory_allocator.ref_count_down(memory_obj)

        self.disk_lock.acquire()
        self.put_tasks.remove(key)
        self.disk_lock.release()


    def _key_to_path(self, key: CacheEngineKey) -> str:
        return self.path + key.to_string().replace("/", "-") + ".pt"
