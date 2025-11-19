import asyncio
import os
import threading
from collections import OrderedDict
from typing import Optional, List
from concurrent.futures import Future

import aiofiles

import torch

from leetcuda.lmcache.cache_controller.message import KVAdmitMsg, KVEvictMsg
from leetcuda.lmcache.cache_controller.worker import LMCacheWorker
from leetcuda.lmcache.config import LMCacheEngineConfig
from leetcuda.lmcache.log import init_logger
from leetcuda.lmcache.lookup_server.abstract_server import LookupServerInterface
from leetcuda.lmcache.memory_management import MemoryAllocatorInterface, MemoryObj
from leetcuda.lmcache.observability import LMCStatsMonitor
from leetcuda.lmcache.storage_backend.abstract_backend import StorageBackendInterface
from leetcuda.lmcache.storage_backend.evictor.base_evictor import PutStatus
from leetcuda.lmcache.utils import DiskCacheMetadata, CacheEngineKey, _lmcache_nvtx_annotate
from leetcuda.lmcache.storage_backend.evictor.lru_evictor import LRUEvictor

logger = init_logger(__name__)




class LocalDiskBackend(StorageBackendInterface):

    def __init__(
            self,
            config: LMCacheEngineConfig,
            loop: asyncio.AbstractEventLoop,
            memory_allocator: MemoryAllocatorInterface,
            dst_device: str = "cuda",
            lmcache_worker: Optional[LMCacheWorker] = None,
            lookup_server: Optional[LookupServerInterface] = None,
    ):
        if torch.cuda.is_available():
            super().__init__(dst_device)
        else:
            super().__init__("cpu")


        self.loop = loop
        self.lmcache_worker = lmcache_worker
        self.instance_id = config.lmcache_instance_id
        self.memory_allocator = memory_allocator
        self.lookup_server = lookup_server


        self.disk_lock = threading.Lock()
        self.dict: OrderedDict[CacheEngineKey, DiskCacheMetadata] = OrderedDict()

        assert config.local_disk is not None
        self.path: str = config.local_disk
        if not os.path.exists(self.path):
            os.makedirs(self.path)
            logger.info(f"Created local disk cache directory: {self.path}")

        # Initialize the evictor
        self.evictor = LRUEvictor(max_cache_size=config.max_local_disk_size)

        self.put_tasks: List[CacheEngineKey] = []

        self.usage = 0
        self.stats_monitor = LMCStatsMonitor.GetOrCreate()










    def submit_put_task(self, key: CacheEngineKey, memory_obj: MemoryObj) -> Optional[Future]:
        assert memory_obj.tensor is not None

        # Update cache recency
        evict_keys, put_status = self.evictor.update_on_put(self.dict, memory_obj.get_physical_size())
        if put_status == PutStatus.ILLEGAL:
            return None
        # evict caches
        for evict_key in evict_keys:
            self.remove(evict_key)
        if self.lookup_server is not None:
            self.lookup_server.batched_remove(evict_keys)

        self.memory_allocator.ref_count_up(memory_obj)

        self.disk_lock.acquire()
        self.put_tasks.append(key)
        self.disk_lock.release()

        future = asyncio.run_coroutine_threadsafe(self.async_save_bytes_to_disk(key, memory_obj), self.loop)
        return future


    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    async def async_save_bytes_to_disk(self, key: CacheEngineKey, memory_obj: MemoryObj) -> None:
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

    def insert_key(self, key: CacheEngineKey, memory_obj: MemoryObj) -> None:
        path = self._key_to_path(key)
        size = memory_obj.get_size()
        shape = memory_obj.metadata.shape
        dtype = memory_obj.metadata.dtype

        has_stored = False
        with self.disk_lock:
            # Need to do reinsert to update cache recency
            if key in self.dict:
                self.dict.pop(key)
                has_stored = True

            self.dict[key] = DiskCacheMetadata(path, size, shape, dtype)

        # push kv admit msg
        if self.lmcache_worker is not None and not has_stored:
            self.lmcache_worker.put_msg(KVAdmitMsg(self.instance_id, key.worker_id, key.chunk_hash, "disk"))

    def remove(self, key: CacheEngineKey) -> None:
        path = self.dict[key].path

        self.disk_lock.acquire()
        self.dict.pop(key)
        self.disk_lock.release()

        size = os.path.getsize(path)
        self.usage -= size
        self.stats_monitor.update_local_storage_usage(self.usage)

        # 删除文件
        os.remove(path)

        # push kv evict msg
        if self.lmcache_worker is not None:
            self.lmcache_worker.put_msg(KVEvictMsg(self.instance_id, key.worker_id, key.chunk_hash, "disk"))


