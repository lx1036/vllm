from dataclasses import dataclass
from typing import Optional

import torch

@dataclass(order=True)
class CacheEngineKey:
    fmt: str
    model_name: str
    world_size: int
    worker_id: int
    chunk_hash: int
    request_configs: Optional[dict] = None

    def to_string(self):
        return f"{self.fmt}@{self.model_name}@{self.world_size}@{self.worker_id}@{self.chunk_hash}"

    @staticmethod
    def from_string(s):
        parts = s.split("@")
        if len(parts) != 5:
            raise ValueError(f"Invalid key string: {s}")
        return CacheEngineKey(parts[0], parts[1], int(parts[2]), int(parts[3]), parts[4])


@dataclass
class DiskCacheMetadata:
    path: str
    size: int  # in bytes
    shape: Optional[torch.Size] = None
    dtype: Optional[torch.dtype] = None
