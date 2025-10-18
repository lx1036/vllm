


import enum

import torch

from nixl._api import nixl_agent, nixl_agent_config
from nixl.logging import get_logger

logger = get_logger(__name__)

class NixlRole(enum.Enum):
    """
    Enum to represent the role of the Nixl connection.
    """
    SENDER = "sender"
    RECEIVER = "receiver"


# target


if __name__ == "__main__":
    receiver_role = str(NixlRole.RECEIVER)
    sender_role = str(NixlRole.SENDER)

    torch.set_default_device("cpu")
    # torch.set_default_device("cuda:0")

    listen_port = 5555
    config = nixl_agent_config(True, True, listen_port)
    agent = nixl_agent(receiver_role, config)
    # data
    tensors = [torch.ones(10, dtype=torch.float32) for _ in range(2)]

    logger.info("Running test with %s tensors in mode %s", tensors, receiver_role)

    # 1. 申请一块内存区域 memory section
    reg_descs = agent.register_memory(tensors)
    if not reg_descs:
        logger.error("Memory registration failed.")
        exit(1)
    # <nixl._bindings.nixlRegDList object at 0x7efff589d7f0>
    # <nixl._bindings.nixlRegDList object at 0x7f9482a3a170>
    logger.info(f"reg_descs: {reg_descs}")

    # 序列化
    target_desc = agent.get_serialized_descs(reg_descs.trim())

    # Send desc list to initiator when metadata is ready
    ready = False
    while not ready:
        # block, 等待 sender 进程起来
        ready = agent.check_remote_metadata(sender_role)

    # 告诉 sender 往内存地址 target_desc 发数据。
    # 等待 sender 发送数据
    logger.info("Waiting for transfer")
    agent.send_notif(sender_role, target_desc)
    logger.info("%s has transfer data", sender_role)

    # For now the notification is just UUID, could be any python bytes.
    # Also can have more than UUID, and check_remote_xfer_done returns
    # the full python bytes, here it would be just UUID.

    # UUIDs 和 sender 里 initialize_xfer("READ", local_descs, remote_descs, receiver_role, "UUIDs") 对应。
    # 经过测试，不对应也能运行???
    while not agent.check_remote_xfer_done(sender_role, b"abcdef"):
        continue

    # Verify data
    logger.info(f"tensors: {tensors}")
    # tensors: [tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]), tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])]

    agent.deregister_memory(reg_descs)
    logger.info("Test Complete.")


