import asyncio
import threading
from typing import Optional, OrderedDict, List, Dict, Tuple, Sequence

from leetcuda.lmcache.cache_controller.worker import LMCacheWorker
from leetcuda.lmcache.config import LMCacheEngineMetadata, LMCacheEngineConfig
from leetcuda.lmcache.lookup_server.abstract_server import LookupServerInterface
from leetcuda.lmcache.memory_management import MemoryAllocatorInterface, MemoryObj, MemoryObjMetadata, MemoryFormat
from leetcuda.lmcache.log import init_logger
from leetcuda.lmcache.storage_backend.base_storage import AllocatorBackendInterface, StorageBackendInterface

import torch

from concurrent.futures import Future

from leetcuda.lmcache.storage_backend.local_disk_storage import LocalDiskBackend
from leetcuda.lmcache.storage_backend.remote_storage import RemoteBackend

logger = init_logger(__name__)

def CreateStorageBackend(
        config: LMCacheEngineConfig,
        metadata: LMCacheEngineMetadata,
        loop: asyncio.AbstractEventLoop,
        memory_allocator: MemoryAllocatorInterface,
        dst_device: str = "cuda",
        lmcache_worker: Optional[LMCacheWorker] = None,
        lookup_server: Optional[LookupServerInterface] = None,
) -> OrderedDict[str, StorageBackendInterface]:
    # Replace 'cuda' with 'cuda:<device id>'
    if dst_device == "cuda":
        dst_device = f"cuda:{torch.cuda.current_device()}"

    storage_backends: OrderedDict[str, StorageBackendInterface] = OrderedDict()

    if config.local_disk and config.max_local_disk_size > 0:
        local_disk_backend = LocalDiskBackend(config, loop, memory_allocator, dst_device, lmcache_worker, lookup_server)
        backend_name = str(local_disk_backend)
        logger.info(f"local_disk_backend name: {backend_name}")
        storage_backends[backend_name] = local_disk_backend

    if config.remote_url is not None:
        remote_backend = RemoteBackend(config, metadata, loop, memory_allocator, dst_device, lookup_server)
        backend_name = str(remote_backend)
        logger.info(f"remote_backend name: {backend_name}")
        storage_backends[backend_name] = remote_backend

    config.enable_blending = False
    assert config.enable_blending is False, "blending is not supported for now"

    return storage_backends


# TODO: extend this class to implement caching policies and eviction policies
class StorageManager:
    """
    The StorageManager is responsible for managing the storage backends.
    """

    def __init__(self,
         config: LMCacheEngineConfig,
         metadata: LMCacheEngineMetadata,
         allocator: MemoryAllocatorInterface,
         lmcache_worker: Optional["LMCacheWorker"] = None,
         lookup_server: Optional[LookupServerInterface] = None
    ):

        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever)
        self.thread.start()

        self.put_tasks: Dict[str, Dict[CacheEngineKey, Tuple[Future, MemoryObj]]] = {}
        #TODO: remove hardcode
        dst_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.storage_backends: OrderedDict[str, StorageBackendInterface] = CreateStorageBackend(config, metadata, self.loop, allocator, dst_device, lmcache_worker, lookup_server)
        for backend_name in self.storage_backends.keys():
            self.put_tasks[backend_name] = {}

        self.memory_allocator = allocator

        self.manager_lock = threading.Lock()

        self.allocator_backend = self.get_allocator_backend(config)
        self.enable_pd = config.enable_pd




    def get_allocator_backend(self, config: LMCacheEngineConfig) -> AllocatorBackendInterface:
        if self.enable_pd:
            allocator_backend = self.storage_backends["PDBackend"]
        else:
            allocator_backend = self.storage_backends["LocalCPUBackend"]
        assert isinstance(allocator_backend, AllocatorBackendInterface)
        return allocator_backend


    def batched_put(
        self,
        keys: Sequence[CacheEngineKey],
        memory_objs: List[MemoryObj],
        transfer_spec=None,
        location: Optional[str] = None,
    ) -> None:
        obj_dict: dict[
            str,
            tuple[Sequence[CacheEngineKey], list[MemoryObj]],
        ] = {}
        obj_dict[self.allocator_backend.__class__.__name__] = (
            keys,
            memory_objs,
        )

        for backend_name, backend in self.storage_backends.items():
            if location and backend_name != location:
                continue

            allocator_backend = backend.get_allocator_backend()
            cname = allocator_backend.__class__.__name__
            if cname not in obj_dict:
                new_keys, new_objs = allocate_and_copy_objects(allocator_backend, keys, memory_objs, self.internal_copy_stream)
                obj_dict[cname] = (new_keys, new_objs)

            # NOTE: the handling of exists_in_put_tasks
            # is done in the backend
            ks, objs = obj_dict[cname]
            backend.batched_submit_put_task(ks, objs, transfer_spec=transfer_spec)

        for cname, (ks, objs) in obj_dict.items():
            for memory_obj in objs:
                memory_obj.ref_count_down()


    def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        return None



    def contains(self, key: CacheEngineKey, search_range: Optional[List[str]] = None) -> bool:
        return False


    def batched_get(
            self,
            keys: List[CacheEngineKey],
            location: Optional[str] = None,
    ) -> Optional[List[Optional[MemoryObj]]]:
        """
        Blocking function to get the memory objects from the storages.
        """
        for backend_name, storage_backend in self.storage_backends.items():
            if location and backend_name != location:
                continue
            memory_objs = storage_backend.batched_get_blocking(keys)
            if memory_objs:
                return memory_objs
        return None


    def allocate(
        self,
        shape: torch.Size,
        dtype: torch.dtype,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        eviction=True,
        busy_loop=True,
    ) -> Optional[MemoryObj]:
        """
        会实际占用内存，如果内存不够，分配不出足够内存存储 memory_obj 对象
        1.MemoryAllocator 分配系统内存(默认max_local_cpu_size=5GB)，8张卡，每张卡占用 5GB.
        2.采用 LRU evictor
        """
        return self.allocator_backend.allocate(shape, dtype, fmt, eviction=eviction, busy_loop=busy_loop)








