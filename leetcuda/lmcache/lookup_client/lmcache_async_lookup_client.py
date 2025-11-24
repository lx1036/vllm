




# NOTE(Jiayi): Prefetch could load extra redundant cache if multiple
# workers has different hit tokens.
class LMCacheAsyncLookupClient(LookupClientInterface):
    """
    ZMQ-based lookup client that communicates with a lookup server.

    Related extra_config:
    - create_lookup_server_only_on_worker_0_for_mla:
        is a flag to control whether to create lookup server only on worker 0.
    """

    def __init__(
            self,
            vllm_config: "VllmConfig",
    ):





class LMCacheAsyncLookupServer:
