import asyncio
import threading
from typing import TYPE_CHECKING

import msgspec
import zmq

from leetcuda.lmcache.cache_controller.message import WorkerMsg, RegisterMsg, DeRegisterMsg, HeartbeatMsg
from leetcuda.lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata
from leetcuda.lmcache.log import init_logger
from leetcuda.lmcache.rpc_utils import get_ip, get_zmq_context, get_zmq_socket

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
        """
        LMCacheWorker register/deregister/heartbeat to RegisterController
        """
        self.config = config
        self.worker_id = metadata.worker_id
        self.lmcache_instance_id = config.lmcache_instance_id
        lmcache_worker_port = config.lmcache_worker_ports[self.worker_id]
        self.lmcache_worker_ip = get_ip()
        self.lmcache_worker_port = lmcache_worker_port

        self.context = get_zmq_context()
        assert config.controller_pull_url is not None
        self.push_socket = get_zmq_socket(
            self.context,
            config.controller_pull_url,
            protocol="tcp",
            role=zmq.PUSH,  # type: ignore[attr-defined]
            bind_or_connect="connect",
        )
        if config.controller_reply_url is not None:
            controller_rep_url = config.controller_reply_url
            self.req_socket = get_zmq_socket(
                self.context,
                controller_rep_url,
                protocol="tcp",
                role=zmq.REQ,  # type: ignore[attr-defined]
                bind_or_connect="connect",
            )

        self.p2p_host = config.p2p_host
        self.p2p_init_port = config.p2p_init_ports[self.worker_id]
        self.p2p_init_url = f"{self.p2p_host}:{self.p2p_init_port}"

        self.lmcache_worker_internal_url = f"*:{lmcache_worker_port}"
        self.reply_socket = get_zmq_socket(
            self.context,
            self.lmcache_worker_internal_url,
            protocol="tcp",
            role=zmq.REP,  # type: ignore[attr-defined]
            bind_or_connect="bind",
        )

        # logger.info(f"Reply socket established at {self.lmcache_worker_internal_url}")

        # go self.start_all()
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()
        asyncio.run_coroutine_threadsafe(self.start_all(), self.loop)

        self.msg_queue: asyncio.Queue[WorkerMsg] = asyncio.Queue()

        self.register()

    async def start_all(self):
        try:
            logger.info(f"Starting lmcache worker {self.worker_id} for instance {self.lmcache_instance_id}")
            await asyncio.gather(
                self.push(),
                # self.handle_request(),
                self.heartbeat(),
            )
        except Exception as e:
            logger.error(f"Instance {self.lmcache_instance_id}, worker {self.worker_id} error: {e}")

    # server: LMCacheControllerManager.handle_batched_push_request()
    async def push(self):
        while True:
            try:
                msgs = await self.batched_get_msg()
                logger.debug(f"Sending {len(msgs)} messages")
                self.push_socket.send_multipart([msgspec.msgpack.encode(msg) for msg in msgs])
            except Exception as e:
                logger.error(f"Push error: {e}")

    async def batched_get_msg(self, max_bsz: int = 50) -> list[WorkerMsg]:
        """
        Get a batch of messages from the message queue.
        """
        batch = []
        # use blocking get for the first msg
        try:
            item = await self.msg_queue.get() # block
            batch.append(item)
        except asyncio.CancelledError:
            return batch  # shutdown path

        for _ in range(max_bsz - 1):
            try:
                item = self.msg_queue.get_nowait()
                batch.append(item)
            except asyncio.QueueEmpty:
                break

        return batch

    def put_msg(self, msg: WorkerMsg):
        """
        Put a message into the message queue.
        """
        self.loop.call_soon_threadsafe(self.msg_queue.put_nowait, msg)

    def register(self):
        """
        Register the lmcache worker with the controller.
        """
        assert self.lmcache_instance_id is not None
        logger.info(f"Registering lmcache instance-worker: {(self.lmcache_instance_id, self.worker_id)}")
        self.put_msg(
            RegisterMsg(
                instance_id=self.lmcache_instance_id,
                worker_id=self.worker_id,
                ip=self.lmcache_worker_ip,
                port=self.lmcache_worker_port,
                distributed_url=self.p2p_init_url,
            )
        )

    def deregister(self):
        """
        De-register the lmcache worker from the controller.
        """
        assert self.lmcache_instance_id is not None
        self.put_msg(
            DeRegisterMsg(
                instance_id=self.lmcache_instance_id,
                worker_id=self.worker_id,
                ip=self.lmcache_worker_ip,
                port=self.lmcache_worker_port,
            )
        )

    async def heartbeat(self):
        enable_heartbeat = (
            self.config.lmcache_worker_heartbeat_time is not None
            and self.config.lmcache_worker_heartbeat_time > 0
        )
        if enable_heartbeat:
            logger.info(
                f"Start heartbeat in {self.lmcache_instance_id} : {self.worker_id}, "
                f"delay time: {self.config.lmcache_worker_heartbeat_delay_time}s, "
                f"heartbeat time: {self.config.lmcache_worker_heartbeat_time}s"
            )

            await asyncio.sleep(self.config.lmcache_worker_heartbeat_delay_time)

            while True:
                self.put_msg(
                    HeartbeatMsg(
                        instance_id=self.lmcache_instance_id,
                        worker_id=self.worker_id,
                        ip=self.lmcache_worker_ip,
                        port=self.lmcache_worker_port,
                        distributed_url=self.p2p_init_url,
                    )
                )

                await asyncio.sleep(self.config.lmcache_worker_heartbeat_time)

def test_lmcache_worker():
    logger.info("test_lmcache_worker")

