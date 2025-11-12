

import msgspec



class MsgBase(msgspec.Struct, tag=True):  # type: ignore
    """Base class for all messages"""

    def describe(self) -> str:
        return ""




class WorkerMsg(MsgBase):
    """Message between LMCache and Controller"""

    def describe(self) -> str:
        return ""


class KVAdmitMsg(WorkerMsg):
    """Message for KV chunk admission"""
    # TODO(Jiayi): instance_id can be replaced with url
    instance_id: str
    worker_id: int
    key: str
    location: str

    def describe(self) -> str:
        return f"kv_admit {self.key} to {self.instance_id}"



class KVEvictMsg(WorkerMsg):
    """Message for KV chunk eviction"""
    # TODO(Jiayi): instance_id can be replaced with url
    instance_id: str
    worker_id: int
    key: str
    location: str

    def describe(self) -> str:
        return f"kv_evict {self.key} from {self.instance_id}"
