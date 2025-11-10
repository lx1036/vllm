import asyncio
import threading
from typing import Optional, OrderedDict, List

from abstract_backend import StorageBackendInterface
from nixl_backend import NixlBackend
import torch
from ..memory_management import MemoryObj, MemoryAllocatorInterface
from ..utils import CacheEngineKey
from ..config import LMCacheEngineConfig, LMCacheEngineMetadata
from ..lookup_server.abstract_server import LookupServerInterface
from remote_backend import RemoteBackend
from local_disk_backend import LocalDiskBackend
from ..cache_controller.worker import LMCacheWorker

def CreateStorageBackends(
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
        storage_backends[backend_name] = local_disk_backend

    if config.remote_url is not None:
        remote_backend = RemoteBackend(config, metadata, loop, memory_allocator, dst_device, lookup_server)
        backend_name = str(remote_backend)
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



        self.storage_backends: OrderedDict[str, StorageBackendInterface] = CreateStorageBackends()
        self.memory_allocator = allocator

        self.manager_lock = threading.Lock()





    def put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        """
        Non-blocking function to put the memory object into the storages.
        Do not store if the same object is being stored (handled here by
        storage manager) or has been stored (handled by storage backend).
        """
        self.manager_lock.acquire()

        for backend_name, backend in self.storage_backends.items():
            put_task = backend.submit_put_task(key, memory_obj)
            if put_task is None:
                continue




    def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:



    def contains(self, key: CacheEngineKey, search_range: Optional[List[str]] = None) -> bool:


    def allocate(self, shape: torch.Size, dtype: torch.dtype, eviction=True) -> Optional[MemoryObj]:
        """
        Allocate memory object with memory allocator.
        Use LRU evictor if eviction is enabled.
        """
        self.manager_lock.acquire()
        memory_obj = self.memory_allocator.allocate(shape, dtype)
        if not eviction or memory_obj is not None:
            self.manager_lock.release()
            return memory_obj


        self.manager_lock.release()
        return memory_obj





class DistributedStorageManager:

    """
    The storage manager for P-D disaggregation setting.

    """


    def __init__(self):

        self.storage_backend: NixlBackend = NixlBackend.CreateNixlBackend(config, metadata)



    def prepare_put(self, keys: list[CacheEngineKey], metadatas: list[MemoryObjMetadata]) -> None:
        self.storage_backend.register_put_tasks(keys, metadatas)





