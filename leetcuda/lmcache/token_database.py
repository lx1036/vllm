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
        tokens: Optional[Union[torch.Tensor, List[int]]] = None,
        hashes: Optional[List[int]] = None,
        offsets: Optional[List[int]] = None,
        mask: Optional[torch.Tensor] = None,
        make_key: bool = True,
        request_configs: Optional[dict] = None,
    ) -> Iterable[Tuple[int, int, Union[CacheEngineKey, str]]]:
        raise NotImplementedError


class ChunkedTokenDatabase(TokenDatabase):
    def __init__(self, config: Optional[LMCacheEngineConfig] = None, metadata: Optional[LMCacheEngineMetadata] = None):
        if config is not None:
            self.chunk_size = config.chunk_size # 256
        self.metadata = metadata

    # info: 还是每太明白，输入 token_ids，怎么处理成 []CacheEngineKey
    def process_tokens(
        self,
        tokens: Optional[Union[torch.Tensor, List[int]]] = None,
        hashes: Optional[List[int]] = None,
        offsets: Optional[List[int]] = None,
        mask: Optional[torch.Tensor] = None,
        make_key: bool = True,
        request_configs: Optional[dict] = None,
    ) -> Iterable[Tuple[int, int, Union[CacheEngineKey, int]]]:
        """
        Process the tokens and return the corresponding cache engine keys.

        :returns: A iterable of tuples with three elements. The first element
            is the start index of the tokens for the key. The second element
            is the end index of the tokens for the key. The third element is
            the cache engine key (or hash) for the tokens.
        """
        if mask is not None:
            num_falses = mask.numel() - mask.long().sum().item()
        else:
            num_falses = 0
        if num_falses % self.chunk_size != 0:
            raise ValueError("The number of Falses in the mask is not a multiple of the chunk size.")

        if tokens is not None:
            total_len = len(tokens)
            token_chunks = self.chunk_tokens(tokens)
            prefix_hashes = self.prefix_hash(token_chunks) # info: 这里的 prefix hash 是什么?
            start_idx = 0
            for chunk_id, hash_val in enumerate(prefix_hashes):
                start_idx = chunk_id * self.chunk_size
                end_idx = min(start_idx + self.chunk_size, total_len)
                if start_idx < num_falses:
                    continue
                else:
                    if make_key:
                        yield start_idx, end_idx, self.make_key_by_hash(hash_val, request_configs)
                    else:
                        yield start_idx, end_idx, hash_val
        elif hashes is not None:
            assert offsets is not None, "If hashes are provided, offsets must also be provided."
            start_idx = 0
            for hash_val, offset in zip(hashes, offsets, strict=False):
                end_idx = start_idx + offset
                if make_key:
                    yield start_idx, end_idx, self.make_key_by_hash(hash_val, request_configs)
                else:
                    yield start_idx, end_idx, hash_val
                start_idx = end_idx
        else:
            raise ValueError("Either tokens or hashes must be provided.")

    def chunk_tokens(self, tokens: Union[torch.Tensor, List[int]]) -> Iterable[Union[torch.Tensor, List[int]]]:
        """
        Chunk the tokens into chunks of size self.chunk_size.

        :param tokens:
        :return: a generator of chunks of tokens, each with shape [chunk_size]
        """
        for i in range(0, len(tokens), self.chunk_size):
            yield tokens[i: i+self.chunk_size]

    # info: 意思是每一个 chunk 的 hash 值都是基于前一个 chunk 的 hash 值计算出来的？
    def prefix_hash(self, token_chunks: Iterable[Union[torch.Tensor, List[int]]]) -> Iterable[str]:
        prefix_hash = self.get_init_hash()
        for token_chunk in token_chunks:
            prefix_hash = self.hash(token_chunk, prefix_hash)
            yield prefix_hash

    def get_init_hash(self) -> str:
        return ""

    def hash(self, tokens: Union[torch.Tensor, List[int]], prefix_hash: str) -> str:
        # TODO: change it to a more efficient hash function
        tokens_bytes: bytes = bytes()
        if isinstance(tokens, torch.Tensor):
            tokens_bytes = tokens.cpu().to(torch.uint32).numpy().tobytes() # info: 如果是 tensor，则需要获取 tensor 的 bytes
        elif isinstance(tokens, list):
            tokens_bytes = array.array('I', tokens).tobytes() # info: 如果是 []int, 则直接转换成 bytes

        return hashlib.sha256(prefix_hash.encode("ascii") + tokens_bytes).hexdigest()

    def make_key_by_hash(self, chunk_hash: str, request_configs: Optional[dict] = None):
        assert self.metadata is not None
        return CacheEngineKey(self.metadata.fmt, self.metadata.model_name, self.metadata.world_size, self.metadata.worker_id, chunk_hash, request_configs)










