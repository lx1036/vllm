import abc
import array
import hashlib
from typing import Union, List, Optional, Tuple, Iterable

import torch

from leetcuda.lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata
from leetcuda.lmcache.utils import CacheEngineKey


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

    def __init__(self, config: Optional[LMCacheEngineConfig] = None, metadata: Optional[LMCacheEngineMetadata] = None):
        if config is not None:
            self.chunk_size = config.chunk_size
        self.metadata = metadata

    def process_tokens(
            self, tokens: Union[torch.Tensor, List[int]],
            mask: Optional[torch.Tensor] = None,
            make_key: bool = True,
    ) -> Iterable[Tuple[int, int, Union[CacheEngineKey, str]]]:
        """
        Process the tokens and return the corresponding cache engine keys.

        :param tokens:
        :param mask:
        :param make_key:
        :return:
        """


        token_chunks = self._chunk_tokens(tokens)
        prefix_hashes = self._prefix_hash(token_chunks)


        start_idx = 0
        for chunk_id, hash_val in enumerate(prefix_hashes):
            start_idx = chunk_id * self.chunk_size
            end_idx = min(start_idx + self.chunk_size, total_len)
            if start_idx < num_falses:
                continue
            else:
                if make_key:
                    yield start_idx, end_idx, self._make_key_by_hash(hash_val)
                else:
                    yield start_idx, end_idx, hash_val



    def _chunk_tokens(self, tokens: Union[torch.Tensor, List[int]]) -> Iterable[Union[torch.Tensor, List[int]]]:
        """
        Chunk the tokens into chunks of size self.chunk_size.

        :param tokens:
        :return: a generator of chunks of tokens, each with shape [chunk_size]
        """
        for i in range(0, len(tokens), self.chunk_size):
            yield tokens[i: i+self.chunk_size]


    def _prefix_hash(self, token_chunks: Iterable[Union[torch.Tensor, List[int]]]) -> Iterable[str]:
        prefix_hash = self._get_init_hash()
        for token_chunk in token_chunks:
            prefix_hash = self._hash(token_chunk, prefix_hash)
            yield prefix_hash

    def _get_init_hash(self) -> str:
        return ""

    def _hash(self, tokens: Union[torch.Tensor, List[int]], prefix_hash: str) -> str:
        # TODO: change it to a more efficient hash function
        tokens_bytes: bytes = bytes()
        if isinstance(tokens, torch.Tensor):
            tokens_bytes = tokens.cpu().to(torch.uint32).numpy().tobytes()
        elif isinstance(tokens, list):
            tokens_bytes = array.array('I', tokens).tobytes()

        return hashlib.sha256(prefix_hash.encode("ascii") + tokens_bytes).hexdigest()






