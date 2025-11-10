from typing import Optional, Dict, List, Union

import torch

from log import init_logger
from storage_backend.storage_manager import StorageManager, DistributedStorageManager
from token_database import TokenDatabase, ChunkedTokenDatabase
from memory_management import MixedMemoryAllocator, MemoryAllocatorInterface
from config import LMCacheEngineConfig, LMCacheEngineMetadata
from utils import _lmcache_nvtx_annotate
from lookup_server.abstract_server import LookupServerInterface
from gpu_connector import GPUConnectorInterface


logger = init_logger(__name__)

class LMCacheEngine:
    """
    KVCache -> MemoryObjs -> async StorageBackends

    MemoryObjs from Engine -> (GPUConnectors) -> KVCache
    """

    def __init__(self, config: LMCacheEngineConfig,
                 metadata: LMCacheEngineMetadata,
                 memory_allocator: MemoryAllocatorInterface,
                 token_database: TokenDatabase,
                 gpu_connector: GPUConnectorInterface,):
        logger.info(f"Creating LMCacheEngine with config: {config}")
        self.config = config
        self.token_database = token_database
        self.memory_allocator = memory_allocator
        self.lookup_server: Optional[LookupServerInterface] = None
        self.gpu_connector = gpu_connector


        self.storage_manager = StorageManager(config, metadata, self.memory_allocator, self.lmcache_worker, self.lookup_server)



    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def store(self, tokens: torch.Tensor, mask: Optional[torch.Tensor] = None, **kwargs):


        for start, end, key in self.token_database.process_tokens(tokens, mask):
            if self.storage_manager.contains(key):
                continue

            memory_obj = self.storage_manager.allocate(kv_shape, kv_dtype)
            if memory_obj is None:
                logger.warning("Failed to allocate memory for the KV cache. The KV cache will not be stored.")
                break

            self.storage_manager.put(key, memory_obj)

    def store_distributed(self,tokens: torch.Tensor, mask: Optional[torch.Tensor] = None, **kwargs) -> None:


        self.storage_manager.commit_put()



    def retrieve(self, tokens: torch.Tensor):


    def lookup(self, tokens: Union[torch.Tensor, List[int]], search_range: Optional[List[str]] = None) -> int:
        """
        Checks the existence of KV cache of the tokens from the cache engine.

        :param tokens: the input tokens, with shape [seq_len]
        :param search_range:
        :return: An int indicating how many prefix tokens are cached.
        """
        end = 0
        for start, end, key in self.token_database.process_tokens(tokens):
            assert isinstance(key, CacheEngineKey)
            if not self.storage_manager.contains(key, search_range):
                return start

        return end




    def close(self) -> None:


class LMCacheEngineBuilder:
    _instances: Dict[str, LMCacheEngine] = {}
    _cfgs: Dict[str, LMCacheEngineConfig] = {}
    _metadatas: Dict[str, LMCacheEngineMetadata] = {}


    @classmethod
    def get_or_create(
            cls,
            instance_id: str,
            config: LMCacheEngineConfig,
            metadata: LMCacheEngineMetadata,
            gpu_connector:
            GPUConnectorInterface,  # gpu connectors is from outside
    ) -> LMCacheEngine:
        logger.info(f"Creating LMCacheEngine instance {instance_id}")
        if instance_id not in cls._instances:
            memory_allocator = cls._Create_memory_allocator(config, metadata)
            token_database = cls._Create_token_database(config, metadata)

            engine = LMCacheEngine(config, metadata, memory_allocator, token_database, gpu_connector)
            cls._instances[instance_id] = engine


            return engine

        else:
            if cls._cfgs[instance_id] != config or cls._metadatas[instance_id] != metadata:
                raise ValueError(f"Instance {instance_id} already exists with a different configuration or metadata.")

            return cls._instances[instance_id]

    @classmethod
    def get(cls, instance_id: str) -> Optional[LMCacheEngine]:
        return cls._instances.get(instance_id)



    @staticmethod
    def _Create_memory_allocator(config: LMCacheEngineConfig, metadata: LMCacheEngineMetadata) -> MemoryAllocatorInterface:
        max_local_cpu_size = config.max_local_cpu_size
        return MixedMemoryAllocator(int(max_local_cpu_size * 1024**3))


    @staticmethod
    def _Create_token_database(config: LMCacheEngineConfig, metadata: LMCacheEngineMetadata) -> TokenDatabase:
        return ChunkedTokenDatabase(config, metadata)

