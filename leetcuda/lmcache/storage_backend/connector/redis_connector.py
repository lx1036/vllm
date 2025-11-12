from typing import Optional

import redis

from leetcuda.lmcache.memory_management import MemoryObj
from leetcuda.lmcache.storage_backend.connector import RemoteConnector
from leetcuda.lmcache.utils import CacheEngineKey


class RedisConnector(RemoteConnector):

    def __init__(self):

        self.connection = redis.Redis(host=host, port=port, decode_responses=False)



    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj):

        redis_metadata_bytes = RedisMetadata().serialize()

        kv_bytes = memory_obj.byte_array


        key_str = key.to_string()
        self.connection.set(key_str + "metadata", redis_metadata_bytes)
        self.connection.set(key_str + "kv_bytes", kv_bytes)



    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        key_str = key.to_string()
        redis_metadata_bytes = self.connection.get(key_str + "metadata")

        redis_metadata = RedisMetadata.deserialize(memoryview(redis_metadata_bytes))


        kv_bytes = self.connection.get(key_str + "kv_bytes")

