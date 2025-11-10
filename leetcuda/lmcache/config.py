from dataclasses import dataclass
from typing import Optional


@dataclass
class LMCacheEngineConfig:
    chunk_size: int

    local_cpu: bool
    # need to be assigned a non-zero
    # value even if local_cpu is disabled
    local_disk: Optional[str]
    max_local_disk_size: float  # in GB

    remote_url: Optional[str]
    remote_serde: Optional[str]  # Can be "naive" or "cachegen"





@dataclass
class LMCacheEngineMetadata:
    model_name: str

