import abc
from typing import List, Optional

from leetcuda.lmcache.log import init_logger
from leetcuda.lmcache.memory_management import MemoryObj
from leetcuda.lmcache.utils import CacheEngineKey

logger = init_logger(__name__)


class RemoteConnector(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        raise NotImplementedError


    @abc.abstractmethod
    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        raise NotImplementedError


    @abc.abstractmethod
    async def exists(self, key: CacheEngineKey) -> bool:
        """
        Check if the remote server contains the key

        Input:
            key: a string

        Returns:
            True if the cache engine contains the key, False otherwise
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def list(self) -> List[str]:
        """
        List all keys in the remote server

        Returns:
            A list of keys in the remote server
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def close(self):
        """
        Close remote server

        """
        raise NotImplementedError




