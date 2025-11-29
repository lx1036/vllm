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
from leetcuda.lmcache.storage_backend.base_backend import StorageBackendInterface
from leetcuda.lmcache.storage_backend.evictor.base_evictor import PutStatus
from leetcuda.lmcache.utils import DiskCacheMetadata, CacheEngineKey, _lmcache_nvtx_annotate
from leetcuda.lmcache.storage_backend.evictor.lru_evictor import LRUEvictor

logger = init_logger(__name__)




class LocalDiskBackend(StorageBackendInterface):

    def __init__(
            self,
            config: LMCacheEngineConfig,
            loop: asyncio.AbstractEventLoop,
            local_cpu_backend: LocalCPUBackend,
            dst_device: str = "cuda",
            lmcache_worker: Optional["LMCacheWorker"] = None,
    ):
        if torch.cuda.is_available():
            super().__init__(dst_device)
        else:
            super().__init__("cpu")

        self.cache_policy = get_cache_policy(config.cache_policy)
        self.dict = self.cache_policy.init_mutable_mapping()
        self.dst_device = dst_device
        self.local_cpu_backend = local_cpu_backend
        self.disk_lock = threading.Lock()
        assert config.local_disk is not None
        self.path: str = config.local_disk
        if not os.path.exists(self.path):
            os.makedirs(self.path)
            logger.info(f"Created local disk cache directory: {self.path}")
        self.loop = loop
        self.use_local_cpu = config.local_cpu
        # Block size (for file system I/O)
        stat = os.statvfs(self.path)
        self.os_disk_bs = stat.f_bsize
        self.use_odirect = False
        if config.extra_config is not None:
            self.use_odirect = config.extra_config.get("use_odirect", False)
        logger.info("Using O_DIRECT for disk I/O: %s", self.use_odirect)

        self.disk_worker = LocalDiskWorker(loop)
        # TODO(Jiayi): We need a disk space allocator to avoid fragmentation
        # and hide the following details away from the backend.
        self.max_cache_size = int(config.max_local_disk_size * 1024**3)
        self.current_cache_size = 0.0
        # to help maintain suffix -> prefix order in the dict
        # assumption: only one request is looked up at a time
        # (only one worker per cache engine)
        self.keys_in_request: List[CacheEngineKey] = []
        self.lmcache_worker = lmcache_worker
        self.instance_id = config.lmcache_instance_id
        self.stats_monitor = LMCStatsMonitor.GetOrCreate()
        self.usage = 0

    def __str__(self):
        return "LocalDiskBackend"


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



# TODO(Jiayi): handle cases where cache is repetitvely prefetched.
class LocalDiskWorker:
    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop
        self.executor = AsyncPQThreadPoolExecutor(loop, max_workers=4)







