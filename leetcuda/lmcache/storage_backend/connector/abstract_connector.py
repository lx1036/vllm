import abc
import asyncio



from redis_connector import RedisConnector

logger = init_logger(__name__)


class RemoteConnector(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        raise NotImplementedError


    @abc.abstractmethod
    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        raise NotImplementedError




def CreateConnector(url: str, loop: asyncio.AbstractEventLoop, memory_allocator: MemoryAllocatorInterface) -> RemoteConnector:

    match connector_type:
        case "redis":
            connector = RedisConnector(host, port, loop, memory_allocator)

        case "lm":


        case "infinistore":


        case "mooncakestore":

        case _:
            raise ValueError(f"Unknown connector type {connector_type} (url is: {url})")


    logger.info(f"Created connector {connector} for {connector_type}")
    return connector






