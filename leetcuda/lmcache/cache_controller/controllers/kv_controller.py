from dataclasses import dataclass

from leetcuda.lmcache.cache_controller.executor import LMCacheClusterExecutor
from leetcuda.lmcache.cache_controller.message import LookupMsg, LookupRetMsg, ClearMsg, ClearRetMsg
from leetcuda.lmcache.token_database import ChunkedTokenDatabase


# TODO: Need more efficient data structures (e.g., trie)
# to handle these operations (e.g., evict, deregister)
# more efficiently.

@dataclass
class KVChunkMetadata:
    """
    A class representing a KV chunk metadata.
    """

    instance_id: str
    worker_id: int
    location: str

class KVController:
    def __init__(self) -> None:
        # NOTE: Even if we offload kv_pool to
        # redis. We might need a local cache for handling
        # messages like `check_finish`. Or everything should be
        # written to redis.
        self.kv_pool: dict[int, list[KVChunkMetadata]] = {}

        # TODO: remove this hardcode
        self.token_database = ChunkedTokenDatabase()


    def post_init(self, reg_controller, cluster_executor: LMCacheClusterExecutor):
        """
        Post initialization of the KV controller.
        """
        self.reg_controller = reg_controller
        self.cluster_executor = cluster_executor


    # TODO(Jiayi): The current implementation does not handle
    # the case where the prefix chunks are evicted while the
    # suffix chunk is still in the system. LMCache should guarantee
    # this does not happen.
    # TODO(Jiayi): The current implementation does not consider
    # the location of the kv chunks. It simply returns the
    # `instance_id` with longest prefix.
    # TODO(Jiayi): Need to get rid of the hash somehow
    async def lookup(self, msg: LookupMsg) -> LookupRetMsg:
        tokens = msg.tokens
        layout_info = {}
        for start, end, key in self.token_database.process_tokens(tokens, make_key=False):
            if key not in self.kv_pool:
                break
            matched_instance = self.kv_pool[key][0].instance_id
            matched_location = self.kv_pool[key][0].location
            layout_info[matched_instance] = (matched_location, end)
        return LookupRetMsg(layout_info=layout_info, event_id=msg.event_id)

    async def clear(self, msg: ClearMsg) -> ClearRetMsg:
        """
        Clear kv chunks of instance-worker(s).
        """
        return await self.cluster_executor.execute("clear", msg)
