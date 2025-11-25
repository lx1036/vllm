from typing import Union

from leetcuda.lmcache.cache_controller.message import MsgBase, ErrorMsg, ClearMsg, ClearRetMsg


# NOTE (Jiayi): `LMCacheClusterExecutor` might need to be in different processes
# in the future for the sake of performance.
# NOTE (Jiayi): Also, consider scaling up the number of cluster executors
# in the future.
# TODO (Jiayi): need better error handling
class LMCacheClusterExecutor:
    """
    LMCache Cluster Executor class to handle the execution of cache operations.
    """

    def __init__(self, reg_controller):
        self.reg_controller = reg_controller


    # TODO(Jiayi): need to make the types more specific
    async def execute(self, operation: str, msg: MsgBase) -> MsgBase:
        """
        Execute a cache operation with error handling.

        :param operation: The operation to execute
        (e.g., 'clear').
        :param msg: The message containing the operation details.
        :return: The result of the operation or an error message.
        """
        try:
            method = getattr(self, operation)
            return await method(msg)
        except AttributeError:
            return ErrorMsg(error=f"Operation '{operation}' is not supported.")
        except Exception as e:
            return ErrorMsg(error=str(e))

    async def clear(self, msg: ClearMsg) -> Union[ClearRetMsg, ErrorMsg]:




