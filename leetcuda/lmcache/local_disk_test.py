import torch

from cache_engine import LMCacheEngine, LMCacheEngineBuilder
from utils import ENGINE_NAME, lmcache_get_config
from config import LMCacheEngineMetadata
from vllm.config import CacheConfig, ModelConfig, ParallelConfig
from gpu_connector import VLLMPagedMemGPUConnectorV2


data = torch.tensor([1,2,3], dtype=torch.float32)


config = lmcache_get_config()
model = "Qwen/Qwen3-0.6B"
world_size = 1*2 # 单机两张卡
rank = 0
kv_dtype = data.dtype
num_layer = 32
chunk_size = 2
num_kv_head = 32
head_size = 512
kv_shape = (num_layer, 2, chunk_size, num_kv_head, head_size)
metadata = LMCacheEngineMetadata(model, world_size, rank, "vllm", kv_dtype, kv_shape)

# cuda
# device = torch.device(f"cuda:{rank}")
device = torch.device('cpu')
hidden_dim_size = num_kv_head * head_size
vllm_gpu_connector = VLLMPagedMemGPUConnectorV2(hidden_dim_size, num_layer, use_gpu=use_gpu, chunk_size=chunk_size, dtype=kv_dtype, device=device)
engine = LMCacheEngineBuilder.get_or_create(ENGINE_NAME, config, metadata, vllm_gpu_connector)
engine.store(data)
