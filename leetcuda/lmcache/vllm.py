from enum import Enum
from typing import List

from cache_engine import LMCacheEngineBuilder
from utils import ENGINE_NAME
from config import LMCacheEngineConfig
from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata
from vllm.attention.backends.flash_attn import FlashAttentionMetadata

import torch
from log import init_logger


logger = init_logger(__name__)


class StoreStatus(Enum):
    PREFILL = 1
    CHUNK_PREFILL = 2
    DECODE = 3
    SUFFIX_PREFILL = 4
    NONE = 5


def init_lmcache_engine(
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        cache_config: CacheConfig,
) -> Optional[LMCacheEngine]:




    return LMCacheEngineBuilder.get_or_create(ENGINE_NAME, config, metadata, vllm_gpu_connector)












def lmcache_store_kv(

        model_input: ModelInputForGPUWithSamplingMetadata,

        store_status: List[StoreStatus],

):
    """

    :param model_input:
    :param store_status: Indicate whether and how KV cache of each req is stored
    :return:
    """

    engine = LMCacheEngineBuilder.get(ENGINE_NAME)
    assert engine is not None, "LMCache engine is not initialized."

    assert isinstance(model_input.attn_metadata, FlashAttentionMetadata), "Only FlashAttention backend is supported for now."
    seq_lens = model_input.attn_metadata.seq_lens
    assert seq_lens is not None

    seq_group_list = model_input.sampling_metadata.seq_groups
    assert seq_group_list is not None

    for seq_group_idx, seq_group in enumerate(seq_group_list):
        for seqid, seq_data in seq_group.seq_data.items():
            status = store_status[seq_data_idx]
            if status in [StoreStatus.NONE]:
                continue
            elif status in [StoreStatus.SUFFIX_PREFILL, StoreStatus.CHUNK_PREFILL]:
                seq_len = seq_lens[seq_data_idx]
            else:
                seq_len = seq_data.get_len()
                if status == StoreStatus.DECODE:
                    if seq_len % engine.config.chunk_size != 0:
                        continue

            current_tokens = torch.tensor(seq_data.get_token_ids()[:seq_len], device="cpu")
            skip_leading_tokens = engine.lookup(current_tokens)
            assert skip_leading_tokens <= seq_len


            if skip_leading_tokens < seq_len:
                stored_token_num = seq_len - skip_leading_tokens


                engine.store(current_tokens.cpu(), kv_tensors_mask, kvcaches=kv_caches, slot_mapping=slot_mapping_req_full, offset=skip_leading_tokens)

            else:
                stored_token_num = 0
                skip_leading_tokens = seq_len


            logger.debug(f"Store skips {skip_leading_tokens} tokens and then stores {stored_token_num} tokens")
            seq_data_idx += 1






def lmcache_retrieve_kv():




def need_gpu_interm_buffer(lmcache_config: LMCacheEngineConfig):
    if lmcache_config.local_cpu:
        return True
    else:
        return False
