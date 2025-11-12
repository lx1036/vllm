from collections import OrderedDict
from typing import Union, List, Tuple

from ...log import init_logger

from base_evictor import BaseEvictor, PutStatus
from ...utils import CacheEngineKey


logger = init_logger(__name__)


class LRUEvictor(BaseEvictor):


    def __init__(self, max_cache_size: float = 10.0):
        # The storage size limit (in bytes)
        self.MAX_CACHE_SIZE = int(max_cache_size * 1024**3)

        # TODO(Jiayi): need a way to avoid fragmentation
        # current storage size (in bytes)
        self.current_cache_size = 0.0



    def update_on_hit(self, key: Union[CacheEngineKey, str], cache_dict: OrderedDict) -> None:
        cache_dict.move_to_end(key)

    def update_on_put(self, cache_dict: OrderedDict, cache_size: int) -> Tuple[List[CacheEngineKey], PutStatus]:





