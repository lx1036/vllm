




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


if __name__ == "__main__":
    receiver_role = str(NixlRole.RECEIVER)
    sender_role = str(NixlRole.SENDER)

    torch.set_default_device("cpu")
    # torch.set_default_device("cuda:0")

    # sender role 的 listen_port 是 0
    listen_port = 0
    config = nixl_agent_config(True, True, listen_port)
    agent = nixl_agent(sender_role, config)
    # data
    tensors = [torch.zeros(10, dtype=torch.float32) for _ in range(2)]

    logger.info("Running test with %s tensors in mode %s", tensors, sender_role)

    # 1. 申请一块内存区域 memory section
    reg_descs = agent.register_memory(tensors)
    if not reg_descs:
        logger.error("Memory registration failed.")
        exit(1)
    # <nixl._bindings.nixlRegDList object at 0x7efff589d7f0>
    logger.info(f"reg_descs: {reg_descs}")

    local_ip = "127.0.0.1"
    local_port = 5555
    remote_ip = "127.0.0.1"
    remote_port = 5555
    agent.fetch_remote_metadata(receiver_role, remote_ip, remote_port)
    agent.send_local_metadata(local_ip, local_port)

    notifs = agent.get_new_notifs()
    while len(notifs) == 0:
        notifs = agent.get_new_notifs()

    remote_descs = agent.deserialize_descs(notifs[receiver_role][0])
    # <nixl._bindings.nixlXferDList object at 0x7f9bb4fd4570>
    logger.info(f"remote_descs: {remote_descs}")
    local_descs = reg_descs.trim()

    # Ensure remote metadata has arrived from fetch
    ready = False
    while not ready:
        ready = agent.check_remote_metadata(receiver_role)

    logger.info("Ready for transfer")

    # Initialize a transfer operation.
    # READ/WRITE
    # xfer_handle = agent.initialize_xfer("READ", local_descs, remote_descs, receiver_role, "abcdef")
    # tensors: [tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]), tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])]
    xfer_handle = agent.initialize_xfer("WRITE", local_descs, remote_descs, receiver_role, "abcdef")
    # tensors: [tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]), tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])]
    if not xfer_handle:
        logger.error("Creating transfer failed.")
        exit()

    state = agent.transfer(xfer_handle)
    if state == "ERR":
        logger.error("Posting transfer failed.")
        exit()
    while True:
        # Check the state of a transfer operation
        state = agent.check_xfer_state(xfer_handle)
        if state == "ERR":
            logger.error("Transfer got to Error state.")
            exit()
        elif state == "DONE":
            break

    # Verify data
    logger.info(f"tensors: {tensors}")
    # tensors: [tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]), tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])]
    for i, tensor in enumerate(tensors):
        if not torch.allclose(tensor, torch.ones(10)): # initiator 收到 target 发送的数据 [torch.zeros(10, dtype=torch.float32) for _ in range(2)]
            logger.error("Data verification failed for tensor %d.", i)
            exit()
    logger.info("%s Data verification passed", sender_role)

    agent.remove_remote_agent(receiver_role)
    agent.release_xfer_handle(xfer_handle)
    agent.invalidate_local_metadata(local_ip, local_port)

    agent.deregister_memory(reg_descs)
    logger.info("Test Complete.")
