from typing import Optional

import torch

from log import init_logger
from .storage_backend.storage_manager import StorageManager, DistributedStorageManager
from token_database import TokenDatabase

logger = init_logger(__name__)

class LMCacheEngine:
    def __init__(self, token_database: TokenDatabase,):
        logger.info(f"Creating LMCacheEngine with config: {config}")

        self.token_database = token_database


        if config.enable_nixl:
            self.storage_manager = DistributedStorageManager()
        else:
            self.storage_manager = StorageManager()



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



    def close(self) -> None:
