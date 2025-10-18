import abc
from typing import Union, List, Optional, Tuple, Iterable

import torch

class TokenDatabase(metaclass=abc.ABCMeta):
    """
    TokenDatabase is used to convert input tokens into list of cache engine keys.

    - ChunkedTokenDatabase: It processes tokens into chunks and convert each chunk into a cache engine key using prefix hash.
    - SegmentTokenDatabase: It processes tokens into segments based on special separators and convert each segment into a cache engine key.
    """

    @abc.abstractmethod
    def process_tokens(
            self,
            tokens: Union[torch.Tensor, List[int]],
            mask: Optional[torch.Tensor] = None,
            make_key: bool = True,
    ) -> Iterable[Tuple[int, int, Union[CacheEngineKey, str]]]:

        raise NotImplementedError


class ChunkedTokenDatabase(TokenDatabase):



    def process_tokens(
            self,
            tokens: Union[torch.Tensor, List[int]],
            mask: Optional[torch.Tensor] = None,
            make_key: bool = True,
    ) -> Iterable[Tuple[int, int, Union[CacheEngineKey, str]]]:

