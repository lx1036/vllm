import abc
from typing import Optional, Tuple


from ..utils import CacheEngineKey

class LookupServerInterface(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def lookup(self, key: CacheEngineKey) -> Optional[Tuple[str, int]]:
        """
        Perform lookup in the lookup server.
        """
        raise NotImplementedError

