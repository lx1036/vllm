import asyncio
from concurrent.futures import Future
from typing import Optional

from .abstract_backend import StorageBackendInterface


from connector.abstract_connector import CreateConnector




class RemoteBackend(StorageBackendInterface):

    def __init__(self):

        self.connection = CreateConnector(config.remote_url, loop, memory_allocator)



    def submit_put_task(self, key: CacheEngineKey, memory_obj: MemoryObj,) -> Optional[Future]:

        compressed_memory_obj = self.serializer.serialize(memory_obj)

        future = asyncio.run_coroutine_threadsafe(self.connection.put(key, compressed_memory_obj), self.loop)
