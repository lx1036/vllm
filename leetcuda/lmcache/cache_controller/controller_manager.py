import asyncio
import json
from typing import Optional

import msgspec
import zmq

from leetcuda.lmcache.cache_controller.controllers.kv_controller import KVController
from leetcuda.lmcache.cache_controller.controllers.registration_controller import RegistrationController
from leetcuda.lmcache.cache_controller.executor import LMCacheClusterExecutor
from leetcuda.lmcache.cache_controller.message import LookupMsg, HealthMsg, QueryInstMsg, ClearMsg, PinMsg, CompressMsg, \
    DecompressMsg, MoveMsg, CheckFinishMsg, OrchRetMsg, OrchMsg, ErrorMsg, WorkerReqMsg, Msg, MsgBase, WorkerReqRetMsg, \
    BatchedP2PLookupMsg, WorkerMsg, KVAdmitMsg, KVEvictMsg, HeartbeatMsg, RegisterMsg, DeRegisterMsg
from leetcuda.lmcache.log import init_logger
from leetcuda.lmcache.rpc_utils import get_zmq_context, get_zmq_socket


logger = init_logger(__name__)


class LMCacheControllerManager:
    def __init__(self, controller_urls: dict[str, str]):
        self.zmq_context = get_zmq_context()
        self.controller_urls = controller_urls
        # TODO: We might need multiple sockets if there are more
        # controllers. For now, we use a single socket to receive messages
        # for all controllers.
        # Similarly we might need more sockets to handle different control
        # messages. For now, we use one socket to handle all control messages.

        # TODO: Another thing is that we might need to decoupe the
        # interactions among `handle_worker_message`, `handle_control_message`
        # and `handle_orchestration_message`. For example, in
        # `handle_orchestration_message`, we might need to call
        # `issue_control_message`. This will make the system less concurrent.

        # Micro controllers
        self.controller_pull_socket = get_zmq_socket(
            self.zmq_context,
            self.controller_urls["pull"],
            protocol="tcp",
            role=zmq.PULL,  # type: ignore[attr-defined]
            bind_or_connect="bind",
        )
        if self.controller_urls["reply"] is not None:
            self.controller_rep_socket = get_zmq_socket(
                self.zmq_context,
                self.controller_urls["reply"],
                protocol="tcp",
                role=zmq.REP,  # type: ignore[attr-defined]
                bind_or_connect="bind",
            )

        self.kv_controller = KVController()
        self.reg_controller = RegistrationController()
        # Cluster executor
        self.cluster_executor = LMCacheClusterExecutor(reg_controller=self.reg_controller)
        # post initialization of controllers
        self.kv_controller.post_init(reg_controller=self.reg_controller, cluster_executor=self.cluster_executor)
        self.reg_controller.post_init(kv_controller=self.kv_controller, cluster_executor=self.cluster_executor)

    async def start_all(self):
        tasks = []
        if self.controller_urls["reply"] is not None:
            tasks.append(self.handle_batched_req_request(self.controller_rep_socket))

        tasks.append(self.handle_batched_push_request(self.controller_pull_socket))

        await asyncio.gather(*tasks, return_exceptions=True)

    async def handle_batched_req_request(self, socket) -> Optional[MsgBase]:
        while True:
            try:
                response = await socket.recv()
                # Parse message based on format
                if response.startswith(b"{"):
                    # JSON format - typically from external systems like Mooncake
                    msg_dict = json.loads(response)
                    msg = msgspec.convert(msg_dict, type=Msg)
                else:
                    # MessagePack format - internal LMCache communication
                    msg = msgspec.msgpack.decode(response, type=Msg) # bytes => Msg

                if isinstance(msg, WorkerReqMsg):
                    ret_msg = await self.handle_worker_req_message(msg)
                    await socket.send(msgspec.msgpack.encode(ret_msg))
                else:
                    logger.error(f"Unknown message type: {type(msg)}")
                    err_msg = ErrorMsg(error=f"Unknown message type: {type(msg)}")
                    await socket.send(msgspec.msgpack.encode(err_msg))
            except Exception as e:
                logger.error(f"Controller Manager error: {e}")

    async def handle_worker_req_message(self, msg: WorkerReqMsg) -> WorkerReqRetMsg:
        if isinstance(msg, BatchedP2PLookupMsg):
            ret_msg = await self.kv_controller.batched_p2p_lookup(msg)
        return ret_msg

    async def handle_batched_push_request(self, socket) -> Optional[MsgBase]:
        while True:
            try:
                responses = await socket.recv_multipart() # socket.recv()
                for response in responses:
                    # Parse message based on format
                    if response.startswith(b"{"):
                        # JSON format - typically from external systems like Mooncake
                        msg_dict = json.loads(response)
                        msg = msgspec.convert(msg_dict, type=Msg)
                    else:
                        # MessagePack format - internal LMCache communication
                        msg = msgspec.msgpack.decode(response, type=Msg)

                    if isinstance(msg, WorkerMsg):
                        await self.handle_worker_message(msg)
                    elif isinstance(msg, OrchMsg):
                        await self.handle_orchestration_message(msg)
                    else:
                        logger.error(f"Unknown message type: {type(msg)}")
            except Exception as e:
                logger.error(f"Controller Manager error: {e}")

    async def handle_worker_message(self, msg: WorkerMsg) -> None:
        if isinstance(msg, HeartbeatMsg):
            await self.reg_controller.heartbeat(msg)
        elif isinstance(msg, RegisterMsg):
            await self.reg_controller.register(msg)
        elif isinstance(msg, DeRegisterMsg):
            await self.reg_controller.deregister(msg)
        elif isinstance(msg, KVAdmitMsg):
            await self.kv_controller.admit(msg)
        elif isinstance(msg, KVEvictMsg):
            await self.kv_controller.evict(msg)
        else:
            logger.error(f"Unknown worker message type: {msg}")

    async def handle_orchestration_message(self, msg: OrchMsg) -> OrchRetMsg:
        if isinstance(msg, LookupMsg):
            return await self.kv_controller.lookup(msg)
        elif isinstance(msg, HealthMsg):
            return await self.reg_controller.health(msg)
        elif isinstance(msg, QueryInstMsg):
            return await self.reg_controller.get_instance_id(msg)
        elif isinstance(msg, ClearMsg):
            return await self.kv_controller.clear(msg)
        elif isinstance(msg, PinMsg):
            return await self.kv_controller.pin(msg)
        elif isinstance(msg, CompressMsg):
            return await self.kv_controller.compress(msg)
        elif isinstance(msg, DecompressMsg):
            return await self.kv_controller.decompress(msg)
        elif isinstance(msg, MoveMsg):
            return await self.kv_controller.move(msg)
        elif isinstance(msg, CheckFinishMsg):
            # FIXME(Jiayi): This `check_finish` thing
            # shouldn't be implemented in kv_controller.
            return await self.kv_controller.check_finish(msg)
        else:
            logger.error(f"Unknown orchestration message type: {msg}")
            raise RuntimeError(f"Unknown orchestration message type: {msg}")










