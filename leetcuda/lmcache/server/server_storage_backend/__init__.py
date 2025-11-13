from leetcuda.lmcache.server.server_storage_backend.abstract_backend import LMSBackendInterface
from leetcuda.lmcache.log import init_logger
from leetcuda.lmcache.server.server_storage_backend.local_backend import LocalBackend, LocalDiskBackend

logger = init_logger(__name__)


def CreateStorageBackend(device: str) -> LMSBackendInterface:
    match device:
        case "cpu":
            # cpu only
            logger.info("Initializing cpu-only cache server")
            return LocalBackend()

        case _:
            # cpu only
            logger.info("Initializing disk-only cache server")
            return LocalDiskBackend(path=device)



