from typing import Optional, List, no_type_check

from leetcuda.lmcache.log import init_logger
from leetcuda.lmcache.memory_management import MemoryAllocatorInterface, MemoryObj
from leetcuda.lmcache.storage_backend.connector.abstract_connector import RemoteConnector
from leetcuda.lmcache.utils import CacheEngineKey

logger = init_logger(__name__)


class BlackholeConnector(RemoteConnector):

    def __init__(self, memory_allocator: MemoryAllocatorInterface):
        self.memory_allocator = memory_allocator

    async def exists(self, key: CacheEngineKey) -> bool:
        return False

    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        return None

    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        self.memory_allocator.ref_count_down(memory_obj)

    @no_type_check
    async def list(self) -> List[str]:
        pass

    async def close(self):
        logger.info("Closed the blackhole connection")
