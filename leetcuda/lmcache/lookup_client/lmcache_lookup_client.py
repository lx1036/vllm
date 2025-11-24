import threading

import msgspec

from leetcuda.lmcache.log import init_logger
from leetcuda.lmcache.lookup_client.abstract_client import LookupClientInterface
from vllm.utils import make_zmq_socket

logger = init_logger(__name__)


class LMCacheLookupClient(LookupClientInterface):
    """
    ZMQ-based lookup client that communicates with a lookup server.

    Related extra_config:
    - create_lookup_server_only_on_worker_0_for_mla:
        is a flag to control whether to create lookup server only on worker 0.
    """




class LMCacheLookupServer:
    """ZMQ-based lookup server that handles lookup requests using LMCacheEngine."""

    def __init__(self, lmcache_engine: LMCacheEngine, vllm_config: "VllmConfig"):
        self.decoder = msgspec.msgpack.Decoder()
        self.ctx = zmq.Context()  # type: ignore[attr-defined]
        rpc_port = vllm_config.kv_transfer_config.get_from_extra_config("lmcache_rpc_port", 0)
        socket_path = get_zmq_rpc_path_lmcache(vllm_config, "lookup", rpc_port, vllm_config.parallel_config.rank)
        self.socket = make_zmq_socket(
            self.ctx,
            socket_path,
            zmq.REP,  # type: ignore[attr-defined]
            bind=True,
        )

        self.lmcache_engine = lmcache_engine
        self.running = True
        self.enable_blending = lmcache_engine.config.enable_blending

        def process_request():
            while self.running:



        logger.info(f"lmcache lookup server start on {socket_path}")
        self.thread = threading.Thread(target=process_request, daemon=True)
        self.thread.start()


    def close(self):
        self.socket.close(linger=0)
        # TODO: close the thread!
