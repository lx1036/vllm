


from connector.nixl_connector import NixlChannel

class NixlBackend(StorageBackendInterface):
    """
    Currently, the put is synchronized and blocking, to simplify the
    implementation.

    At the sender side, it will never save anything but directly write the data
    to the receiver side.
    """

    def __init__(self, nixl_config: NixlConfig):
        """
        Initialize the Nixl storage backend.
        """
        self._nixl_channel = NixlChannel(nixl_config)
        self._registered_keys: list[CacheEngineKey] = []
        self._registered_metadatas: list[MemoryObjMetadata] = []


    @staticmethod
    def CreateNixlBackend(config: LMCacheEngineConfig, metadata: LMCacheEngineMetadata) -> "NixlBackend":
        # Create the Nixl config
        nixl_config = NixlConfig.from_cache_engine_config(config, metadata)
        # Create the Nixl backend
        backend = NixlBackend(nixl_config)
        return backend

    def register_put_tasks(self, keys: list[CacheEngineKey], metadatas: list[MemoryObjMetadata]) -> None:
        if len(self._registered_keys) > 0:
            raise RuntimeError("The backend has already registered put tasks.")

        self._registered_keys = keys
        self._registered_metadatas = metadatas
        self._nixl_channel.prepare_send(keys=keys, metadatas=metadatas)



