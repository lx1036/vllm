import abc
from typing import Optional
from concurrent.futures import Future

from leetcuda.lmcache.memory_management import MemoryObj
from leetcuda.lmcache.utils import CacheEngineKey


class StorageBackendInterface(metaclass=abc.ABCMeta):


    @abc.abstractmethod
    def submit_put_task(self, key: CacheEngineKey, obj: MemoryObj) -> Optional[Future]:
        raise NotImplementedError



