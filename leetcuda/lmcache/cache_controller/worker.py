import asyncio
import threading
from typing import TYPE_CHECKING

import msgspec
import zmq

from leetcuda.lmcache.cache_controller.message import WorkerMsg
from leetcuda.lmcache.cache_controller.rpc import get_zmq_socket, get_zmq_context
from leetcuda.lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata
from leetcuda.lmcache.log import init_logger

if TYPE_CHECKING: # fix circular import
    from leetcuda.lmcache.cache_engine import LMCacheEngine


logger = init_logger(__name__)


class LMCacheWorker:
    """
    LMCache Worker class to handle the execution of cache operations.
    This class is responsible for receiving requests from the executor and
    executing the corresponding operations on the LMCache engine.
    Each worker is associated with a specific LMCache instance and a worker id.
    """

    def __init__(
            self,
            config: LMCacheEngineConfig,
            metadata: LMCacheEngineMetadata,
            lmcache_engine: "LMCacheEngine", # fix circular import
    ):
        self.worker_id = metadata.worker_id
        self.lmcache_instance_id = config.lmcache_instance_id



        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()
        asyncio.run_coroutine_threadsafe(self.start_all(), self.loop)
        self.msg_queue: asyncio.Queue[WorkerMsg] = asyncio.Queue()

        self.context = get_zmq_context()
        self.push_socket = get_zmq_socket(self.context, config.controller_url, protocol="tcp", role=zmq.PUSH)




    def put_msg(self, msg: WorkerMsg):
        """
        Put a message into the message queue.
        """
        self.loop.call_soon_threadsafe(self.msg_queue.put_nowait, msg)

    async def start_all(self):
        try:
            logger.info(f"Starting lmcache worker {self.worker_id} for instance {self.lmcache_instance_id}")
            await asyncio.gather(self.push(), self.handle_request())
        except Exception as e:
            logger.error(f"Instance {self.lmcache_instance_id}, worker {self.worker_id} error: {e}")



    async def push(self):
        while True:
            try:
                msgs = await self.batched_get_msg()
                logger.debug(f"Sending {len(msgs)} messages")
                self.push_socket.send_multipart([msgspec.msgpack.encode(msg) for msg in msgs])

            except Exception as e:
                logger.error(f"Push error: {e}")
