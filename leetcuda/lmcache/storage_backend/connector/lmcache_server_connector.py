import asyncio
import socket

from leetcuda.lmcache.memory_management import MemoryAllocatorInterface, MemoryFormat
from leetcuda.lmcache.protocol import ClientMetaMessage, ServerMetaMessage, Constants
from leetcuda.lmcache.storage_backend.connector.abstract_connector import RemoteConnector
from leetcuda.lmcache.utils import CacheEngineKey

import torch


# TODO: performance optimization for this class, consider using C/C++/Rust
# for communication + deserialization
class LMCServerConnector(RemoteConnector):


    def __init__(self, host: str, port: int, loop: asyncio.AbstractEventLoop, memory_allocator: MemoryAllocatorInterface):
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((host, port))


        self.memory_allocator = memory_allocator
        self.loop = loop
        self.async_socket_lock = asyncio.Lock()




    async def exists(self, key: CacheEngineKey) -> bool:
        async with self.async_socket_lock:
            self.client_socket.sendall(ClientMetaMessage(Constants.CLIENT_EXIST, key, 0, MemoryFormat(1), torch.float16, torch.Size([0, 0, 0, 0])).serialize())
            response = self.client_socket.recv(ServerMetaMessage.packlength())

        return ServerMetaMessage.deserialize(response).code == Constants.SERVER_SUCCESS







