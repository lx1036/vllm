from typing import Union, Optional

from leetcuda.lmcache.lookup_client.abstract_client import LookupClientInterface
from leetcuda.lmcache.token_database import ChunkedTokenDatabase
from leetcuda.lmcache.types import CacheEngineKey

import torch


"""
1. 如果是 mooncake lookup client，则 server 则是 mooncake-store，无需创建 AsyncLookupServer/LookupServer
"""
class MooncakeLookupClient(LookupClientInterface):
    def __init__(self, vllm_config: "VllmConfig", master_addr: str):
        from mooncake.store import MooncakeDistributedStore

        self.store = MooncakeDistributedStore()
        self.store.setup(
            "localhost",
            "P2PHANDSHAKE", # P2P handshake (no HTTP metadata). info: 这里不从 metadata server 里查询，表示什么
            0, # 0MB segment size
            16 * 1024 * 1024, # 16MB local buffer
            "tcp",
            "", # Leave empty; Mooncake auto-picks RDMA devices when needed
            master_addr,
        )

        self.token_database = ChunkedTokenDatabase(config, metadata)


    def lookup(self, token_ids: Union[torch.Tensor, list[int]], lookup_id: Optional[str] = None, request_configs: Optional[dict] = None) -> Optional[int]:
        # 1. token_ids -> keys:[]CacheEngineKey
        keys = []
        ends = []
        for start, end, key in self.token_database.process_tokens(token_ids):
            assert isinstance(key, CacheEngineKey)
            keys.append(key.to_string())
            ends.append(end)

        # Use batch_is_exist to check all keys at once
        # rets is list of int: 1 = found, 0 = not found, -1 = error
        rets = self.store.batch_is_exist(keys)

        # Find the first key that doesn't exist (ret != 1)
        # This follows the same logic as cache engine's lookup method
        for i, ret in enumerate(rets):
            if ret != 1:  # Not found or error
                # Return the end position of the previous chunk
                # If i == 0, no chunks were found, return 0
                return ends[i - 1] if i > 0 else 0

        # All keys were found, return the last end position
        # info: ???
        return ends[-1] if ends else 0
