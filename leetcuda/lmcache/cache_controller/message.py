from typing import Optional, Tuple, Dict

import msgspec


class MsgBase(msgspec.Struct, tag=True):  # type: ignore
    """Base class for all messages"""

    def describe(self) -> str:
        return ""

class ErrorMsg(MsgBase):
    """Control Error Message"""

    error: str

    def describe(self) -> str:
        return f"Error: {self.error}"


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


"""Orchestration Message from Ochestrator to LMCache"""
class OrchMsg(MsgBase):
    """Message from Ochestrator to Controller"""

    def describe(self) -> str:
        return ""

class QueryInstMsg(OrchMsg):
    """Query instance message"""

    event_id: str
    ip: str

    def describe(self) -> str:
        return f"Query instance id of ip {self.ip}"

class OrchRetMsg(MsgBase):
    """Return message from Controller to Ochestrator"""

    def describe(self) -> str:
        return ""

class QueryInstRetMsg(OrchRetMsg):
    """Query instance return message"""

    event_id: str
    instance_id: Optional[str]

    def describe(self) -> str:
        return f"The instance id is {self.instance_id}"

class LookupMsg(OrchMsg):
    """Lookup message"""

    event_id: str
    tokens: list[int]

    def describe(self) -> str:
        return f"Lookup tokens {self.tokens}"

class LookupRetMsg(OrchRetMsg):
    """Lookup return message"""
    event_id: str
    layout_info: Dict[str, Tuple[str, int]]

    def describe(self) -> str:
        return f"The layout info is {self.layout_info}"

class HealthMsg(OrchMsg):
    """Health message"""

    event_id: str
    instance_id: str

    def describe(self) -> str:
        return f"Health check for instance {self.instance_id}"

class ClearMsg(OrchMsg):
    """Clear message"""

    event_id: str
    instance_id: str
    location: str

    def describe(self) -> str:
        return f"Clear tokens in instance {self.instance_id} and locations {self.location}"

class ClearRetMsg(OrchRetMsg):
    """Clear return message"""

    event_id: str
    num_tokens: int

    def describe(self) -> str:
        return f"Number of cleared tokens: {self.num_tokens}"

class PinMsg(OrchMsg):
    """Pin message"""

    event_id: str
    instance_id: str
    location: str
    tokens: list[int]

    def describe(self) -> str:
        return (
            f"Pin tokens {self.tokens} in instance "
            f"{self.instance_id} and "
            f"location {self.location}"
        )

class CompressMsg(OrchMsg):
    """Compress message"""

    event_id: str
    instance_id: str
    method: str
    location: str
    tokens: Optional[list[int]] = None  # `None` means compress all tokens

    def describe(self) -> str:
        return (
            f"Compress tokens {self.tokens} in instance "
            f"{self.instance_id} and "
            f"locations {self.location} with "
            f"method {self.method}"
        )

class DecompressMsg(OrchMsg):
    """Decompress message"""

    event_id: str
    instance_id: str
    method: str
    location: str
    tokens: Optional[list[int]] = None  # `None` means compress all tokens

    def describe(self) -> str:
        return (
            f"Decompress tokens {self.tokens} in instance "
            f"{self.instance_id} and "
            f"locations {self.location} with "
            f"method {self.method}"
        )

class MoveMsg(OrchMsg):
    """Move message"""

    event_id: str
    old_position: Tuple[str, str]
    new_position: Tuple[str, str]
    tokens: Optional[list[int]] = None
    copy: Optional[bool] = False

    def describe(self) -> str:
        return (
            f"Move tokens {self.tokens} from {self.old_position} to {self.new_position}"
        )

class CheckFinishMsg(OrchMsg):
    """Check finish message"""

    event_id: str

    def describe(self) -> str:
        return f"Checking finish for event {self.event_id}"
