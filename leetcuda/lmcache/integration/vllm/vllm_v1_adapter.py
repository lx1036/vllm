from typing import Optional

from leetcuda.lmcache.integration.vllm.utils import lmcache_get_or_create_config
from leetcuda.lmcache.log import init_logger
from leetcuda.lmcache.lookup_client.factory import LookupClientFactory
from leetcuda.lmcache.utils import _lmcache_nvtx_annotate

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)


logger = init_logger(__name__)


class LMCacheConnectorV1Impl:
    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        parent: KVConnectorBase_V1,
    ):
        config = lmcache_get_or_create_config()



        if role == KVConnectorRole.SCHEDULER:
            self.lookup_client = LookupClientFactory.create_lookup_client(vllm_config, config)

        else:

            # Create lookup server using factory
            assert self.lmcache_engine is not None
            self.lookup_server = LookupClientFactory.create_lookup_server(self.lmcache_engine, vllm_config)







    @_lmcache_nvtx_annotate
    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs,
    ) -> None:

        """
        Start saving the a layer of KV cache from vLLM's paged buffer to the connector.
        """



        # Storing KV cache for 24 out of 24 tokens (skip_leading_tokens=0) for request b10134c40d5c422d9c2d7f3c95179293
        logger.info(
            "Storing KV cache for %d out of %d tokens "
            "(skip_leading_tokens=%d) for request %s",
            len(token_ids) - skip_leading_tokens,
            len(token_ids),
            skip_leading_tokens,
            request.req_id,
            )



    @_lmcache_nvtx_annotate
    def wait_for_save(self):
        """Blocking until the KV cache is saved to the connector buffer."""

        # Storing KV cache for 24 out of 24 tokens (skip_leading_tokens=0) for request b10134c40d5c422d9c2d7f3c95179293
        logger.info(
            "Storing KV cache for %d out of %d tokens "
            "(skip_leading_tokens=%d) for request %s",
            len(token_ids) - skip_leading_tokens,
            len(token_ids),
            skip_leading_tokens,
            request.req_id,
            )



    @_lmcache_nvtx_annotate
    def get_num_new_matched_tokens(self, request: "Request", num_computed_tokens: int) -> Optional[int]:
        """
        Check for external KV cache hit.
        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally computed tokens for this request

        Returns:
            the number of tokens that can be loaded from the external KV cache beyond what is already computed.
        """

        # 1. 从 reqs_status 缓存里查询 [lookup_id, num_hit_toks]，避免每次都要从 mooncake-store 里费时操作 lookup()
        # mooncake-store 好像不支持
        if cached_num_hit_toks := self.lookup_client.lookup_cache(lookup_id=req_id):
            return cached_num_hit_toks



        # 2. 从 mooncake-store(本地的 store) 里查询
        num_external_hit_tokens = self.lookup_client.lookup(token_ids, lookup_id=req_id, request_configs=request_configs)
        if num_external_hit_tokens is None:
            logger.debug("Reqid: %s, Total tokens %d, LMCache hit tokens: None.", req_id, request.num_tokens)
            return None

        # When prompt length is divisible by the block size and all
        # blocks are cached, we need to recompute the last token.
        # This will be removed in the future if vLLM's scheduler provides
        # a better support for this case.
        need_to_allocate = num_external_hit_tokens - num_computed_tokens
        # In, full-prompt-hit case, we need to recompute the last token
        if num_external_hit_tokens == request.num_tokens:
            need_to_allocate -= 1

        # Reqid: 00-c7b504418919983f30fc505a46c6ada4-8eea7d07b1b8cc64-00, Total tokens 261, LMCache hit tokens: 261, need to load: 4
        logger.info("Reqid: %s, Total tokens %d, LMCache hit tokens: %d, need to load: %d",request.request_id, request.num_tokens, num_external_hit_tokens, need_to_allocate)

        self.load_specs[req_id] = LoadSpec(vllm_cached_tokens=num_computed_tokens, lmcache_cached_tokens=num_external_hit_tokens, can_load=False)
        if need_to_allocate <= 0:
            return 0

        return need_to_allocate
