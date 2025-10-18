import abc
from typing import Optional
from concurrent.futures import Future


class StorageBackendInterface(metaclass=abc.ABCMeta):


    @abc.abstractmethod
    def submit_put_task(self, key: CacheEngineKey, obj: MemoryObj) -> Optional[Future]:
        raise NotImplementedError



