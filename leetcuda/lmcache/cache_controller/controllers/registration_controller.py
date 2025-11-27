import time
from dataclasses import dataclass

from leetcuda.lmcache.cache_controller.message import HealthMsg, HealthRetMsg, RegisterMsg, HeartbeatMsg
from leetcuda.lmcache.log import init_logger
from leetcuda.lmcache.rpc_utils import get_zmq_context, get_zmq_socket

import zmq
import zmq.asyncio


logger = init_logger(__name__)


@dataclass
class WorkerInfo:
    instance_id: str
    worker_id: int
    ip: str
    port: int
    distributed_url: str
    registration_time: float
    last_heartbeat_time: float

class RegistrationController:

    def __init__(self):
        # Mapping from `instance_id` -> `worker_ids`
        self.worker_mapping: dict[str, list[int]] = {}

        # Mapping from `(instance_id, worker_id)` -> `socket`
        self.socket_mapping: dict[tuple[str, int], zmq.asyncio.Socket] = {}

        # Mapping from `(instance_id, worker_id)` -> `distributed_url`
        # NOTE(Jiayi): `distributed_url` is used for actual KV cache transfer.
        # It's not the lmcache_worker_url
        self.distributed_url_mapping: dict[tuple[str, int], str] = {}

        # Mapping from `ip` -> `instance_id`
        self.instance_mapping: dict[str, str] = {}

        # Mapping from `(instance_id, worker_id)` -> `WorkerInfo`
        self.worker_info_mapping: dict[tuple[str, int], WorkerInfo] = {}

    def post_init(self, kv_controller, cluster_executor):
        """
        Post initialization of the Registration Controller.
        """
        self.kv_controller = kv_controller
        self.cluster_executor = cluster_executor


    def get_workers(self, instance_id: str) -> list[int]:
        """
        Get worker ids given an instance id.
        """
        return self.worker_mapping.get(instance_id, [])

    async def health(self, msg: HealthMsg) -> HealthRetMsg:
        """
        Check the health of the lmcache worker.
        """
        return await self.cluster_executor.execute("health", msg)

    async def register(self, msg: RegisterMsg) -> None:
        """
        Register a new instance-worker connection mapping.
        """
        instance_id = msg.instance_id
        worker_id = msg.worker_id
        ip = msg.ip
        port = msg.port
        url = f"{ip}:{port}"
        distributed_url = msg.distributed_url
        self.distributed_url_mapping[(instance_id, worker_id)] = distributed_url
        self.instance_mapping[ip] = instance_id

        socket = get_zmq_socket(
            get_zmq_context(),
            url,
            protocol="tcp",
            role=zmq.REQ,  # type: ignore[attr-defined]
            bind_or_connect="connect",
        )
        self.socket_mapping[(instance_id, worker_id)] = socket

        self.worker_info_mapping[(instance_id, worker_id)] = WorkerInfo(
            instance_id, worker_id, ip, port, distributed_url, time.time(), time.time()
        )

        if instance_id not in self.worker_mapping:
            self.worker_mapping[instance_id] = []
        self.worker_mapping[instance_id].append(worker_id)
        self.worker_mapping[instance_id].sort()

        logger.info(f"Registered instance-worker {(instance_id, worker_id)} with URL {url}")

    async def heartbeat(self, msg: HeartbeatMsg) -> None:
        """
        Heartbeat from lmcache worker.
        """
        instance_id = msg.instance_id
        worker_id = msg.worker_id
        worker_key = (instance_id, worker_id)
        if worker_key not in self.worker_info_mapping:
            logger.warning(f"{worker_key} has not been registered, re-register the worker.")
            # re-register the worker
            await self.register(msg)
        else:
            # update worker info
            self.worker_info_mapping[worker_key].last_heartbeat_time = time.time()

        now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        logger.info(f"heartbeat from instance-worker {(instance_id, worker_id)} at {now}")

