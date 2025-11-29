import asyncio
import socket
from typing import Optional, List, no_type_check

from leetcuda.lmcache.log import init_logger
from leetcuda.lmcache.memory_management import MemoryAllocatorInterface, MemoryFormat, MemoryObj
from leetcuda.lmcache.protocol import ClientMetaMessage, ServerMetaMessage, Constants
from leetcuda.lmcache.storage_backend.connector.base_connector import RemoteConnector
from leetcuda.lmcache.utils import CacheEngineKey, _lmcache_nvtx_annotate

import torch


logger = init_logger(__name__)


# for communication + deserialization
# lmc client
class LMCServerConnector(RemoteConnector):


    def __init__(self, host: str, port: int, loop: asyncio.AbstractEventLoop, memory_allocator: MemoryAllocatorInterface):
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((host, port))

        self.memory_allocator = memory_allocator
        self.loop = loop
        self.async_socket_lock = asyncio.Lock()


    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        logger.debug("Async call to put()!")

        kv_bytes = memory_obj.byte_array
        kv_shape = memory_obj.get_shape()
        kv_dtype = memory_obj.get_dtype()
        memory_format = memory_obj.get_memory_format()

        async with self.async_socket_lock:
            await self.loop.sock_sendall(self.client_socket, ClientMetaMessage(Constants.CLIENT_PUT, key, len(kv_bytes), memory_format, kv_dtype, kv_shape).serialize())
            await self.loop.sock_sendall(self.client_socket, kv_bytes)

        self.memory_allocator.ref_count_down(memory_obj)


    @_lmcache_nvtx_annotate
    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        async with self.async_socket_lock:
            self.client_socket.sendall(ClientMetaMessage(Constants.CLIENT_GET, key, 0, MemoryFormat(1), torch.float16, torch.Size([0, 0, 0, 0])).serialize())
            data = self.client_socket.recv(ServerMetaMessage.packlength())

        meta = ServerMetaMessage.deserialize(data)
        if meta.code != Constants.SERVER_SUCCESS:
            return None

        async with self.async_socket_lock:
            memory_obj = self.receive_all(meta)

        return memory_obj

    def receive_all(self, meta: ServerMetaMessage) -> Optional[MemoryObj]:
        memory_obj = self.memory_allocator.allocate(meta.shape, meta.dtype, meta.fmt)
        if memory_obj is None:
            logger.warning("Failed to allocate memory during remote receive")
            return None

        buffer = memory_obj.byte_array
        view = memoryview(buffer)

        received = 0
        n = meta.length
        while received < n:
            num_bytes = self.client_socket.recv_into(view[received:], n - received)
            if num_bytes == 0:
                return None
            received += num_bytes

        return memory_obj


    async def exists(self, key: CacheEngineKey) -> bool:
        async with self.async_socket_lock:
            self.client_socket.sendall(ClientMetaMessage(Constants.CLIENT_EXIST, key, 0, MemoryFormat(1), torch.float16, torch.Size([0, 0, 0, 0])).serialize())
            response = self.client_socket.recv(ServerMetaMessage.packlength())

        return ServerMetaMessage.deserialize(response).code == Constants.SERVER_SUCCESS


    @no_type_check
    async def list(self) -> List[str]:
        pass


    async def close(self):
        async with self.async_socket_lock:
            self.client_socket.close()
        logger.info("Closed the lmserver connection")













