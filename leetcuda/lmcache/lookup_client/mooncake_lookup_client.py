


class MooncakeLookupClient(LookupClientInterface):
    def __init__(
            self,
            vllm_config: "VllmConfig",
            master_addr: str,
    ):
        from mooncake.store import MooncakeDistributedStore


