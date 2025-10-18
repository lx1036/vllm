from typing import Optional, OrderedDict

from abstract_backend import StorageBackendInterface
from nixl_backend import NixlBackend

def CreateStorageBackends() -> OrderedDict[str, StorageBackendInterface]:

    storage_backends: OrderedDict[str, StorageBackendInterface] = OrderedDict()



    return storage_backends


# TODO: extend this class to implement caching policies and eviction policies
class StorageManager:
    """
    The StorageManager is responsible for managing the storage backends.
    """

    def __init__(self):



        self.storage_backends: OrderedDict[str, StorageBackendInterface] = CreateStorageBackends()





    def put(self, key: CacheEngineKey, memory_obj: MemoryObj):


        for backend_name, backend in self.storage_backends.items():
            put_task = backend.submit_put_task(key, memory_obj)

            if put_task is None:
                continue




    def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:



class DistributedStorageManager:

    """
    The storage manager for P-D disaggregation setting.

    """


    def __init__(self):

        self.storage_backend: NixlBackend = NixlBackend.CreateNixlBackend(config, metadata)



    def prepare_put(self, keys: list[CacheEngineKey], metadatas: list[MemoryObjMetadata]) -> None:
        self.storage_backend.register_put_tasks(keys, metadatas)





