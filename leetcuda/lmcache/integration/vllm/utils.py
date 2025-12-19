import os
import threading
from typing import Union

from leetcuda.lmcache.config import LMCacheEngineConfig
from leetcuda.lmcache.log import init_logger

# Thread-safe singleton storage
_config_instance: LMCacheEngineConfig = None
_config_lock = threading.Lock()


logger = init_logger(__name__)


def lmcache_get_or_create_config() -> LMCacheEngineConfig:
    """
    Get the LMCache configuration from the environment variable `LMCACHE_CONFIG_FILE`.
    If the environment variable is not set, this function will return the default configuration.
    """

    global _config_instance

    # Double-checked locking for thread-safe singleton
    if _config_instance is None:
        with _config_lock:
            if "LMCACHE_CONFIG_FILE" not in os.environ:
                logger.warning("No LMCache configuration file is set. Trying to read configurations from the environment variables. You can set the configuration file through the environment variable: LMCACHE_CONFIG_FILE" )
                _config_instance = LMCacheEngineConfig.from_env()
            else:
                config_file = os.environ["LMCACHE_CONFIG_FILE"]
                logger.info(f"Loading LMCache config file {config_file}")
                _config_instance = LMCacheEngineConfig.from_file(config_file)
                # Update config from environment variables
                _config_instance.update_config_from_env()

    return _config_instance
