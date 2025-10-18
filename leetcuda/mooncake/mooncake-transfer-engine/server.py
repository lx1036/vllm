import argparse
import logging
import os
import time

import numpy as np
import zmq

import torch

# https://kvcache-ai.github.io/Mooncake/python-api-reference/transfer-engine.html#quick-start

from mooncake.engine import TransferEngine
from safetensors.torch import load as safetensors_load
from safetensors.torch import save as safetensors_save

# vllm/distributed/kv_transfer/kv_pipe/mooncake_pipe.py

logger = logging.getLogger(__name__)

def main(local_hostname: str):
    print(f"local_hostname: {local_hostname}")

    # Initialize ZMQ context and socket
    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    socket.bind("tcp://*:5555")  # Bind to port 5555 for buffer info

    HOSTNAME = local_hostname # "10.252.112.16" # localhost for simple demo
    METADATA_SERVER = "P2PHANDSHAKE" # [ETCD_SERVER_URL, P2PHANDSHAKE, ...]
    PROTOCOL = "rdma" # [rdma, tcp, ...]
    DEVICE_NAME = "mlx5_ib1" # auto discovery if empty

    # 注册 memory region
    engine = TransferEngine()
    # mooncake-integration/transfer_engine/transfer_engine_py.cpp::TransferEnginePy::initialize(local_hostname, metadata_server, protocol, device_name)
    ret_value = engine.initialize(
        HOSTNAME,
        METADATA_SERVER,
        PROTOCOL,
        DEVICE_NAME
    )
    if ret_value != 0:
        logger.error("Mooncake Transfer Engine initialization failed.")
        raise RuntimeError("Mooncake Transfer Engine initialization failed.")

    # server data
    # torch.set_default_device("cpu")
    # torch.set_default_device("cuda:0")
    # tensors = [torch.ones(10, dtype=torch.float32) for _ in range(2)]
    server_buffer = torch.zeros(10, device="cuda:0")
    # server_data = safetensors_save({"tensor": server_buffer})
    # server_len = len(server_data)
    # server_ptr = engine.allocate_managed_buffer(server_len)
    # if server_ptr <= 0:
    #     print("Allocation Return Error")
    #     raise Exception("Allocation Return Error")

    server_ptr = server_buffer.data_ptr()
    # server_len = server_buffer.size().numel()
    server_len = server_buffer.numel() * server_buffer.element_size()

    # server_buffer = np.zeros(1024*1024, dtype=np.uint8)
    # server_ptr = server_buffer.ctypes.data # 获取这个数组的内存地址
    # server_len = server_buffer.nbytes
    ret_value = engine.register_memory(server_ptr, server_len)
    if ret_value != 0:
        print("Mooncake memory registration failed.")
        raise RuntimeError("Mooncake memory registration failed.")

    # get_rpc_port(): Gets the RPC port that the transfer engine is listening on
    session_id = f"{HOSTNAME}:{engine.get_rpc_port()}"
    # Send buffer info to client
    buffer_info = {
        "session_id": session_id,
        "ptr": server_ptr,
        "len": server_len
    }
    print(f"Server initialized with session ID: {session_id}")
    print(f"Server buffer address: {server_ptr}, length: {server_len}")
    socket.send_json(buffer_info)
    print("Buffer information sent to client")

    print(f"old tensors: {server_buffer}")
    # Keep server running
    try:
        while True:
            # if np.array_equal(server_buffer, np.ones(1024*1024, dtype=np.uint8))
            if torch.ones(10, device="cuda:0").equal(server_buffer):
            # if np.allclose(server_buffer, np.ones(1024*1024, dtype=np.uint8)):
                print(f"new tensors: {server_buffer}")
                print("received data from client successfully")
                break
            else:
                # print(f"wrong tensors: {server_buffer}")
                print("wait for receiving data from client, sleep 3s...")
                time.sleep(3)
            # input("Press Enter to exit...")
    except KeyboardInterrupt:
        print("\nShutting down server...")
    finally:

        # Cleanup
        ret_value = engine.unregister_memory(server_ptr)
        if ret_value != 0:
            print("Mooncake memory unregister_memory failed.")
            raise RuntimeError("Mooncake memory unregister_memory failed.")
        socket.close()
        context.term()


# 多机：python3 server.py --local_hostname=10.252.112.16
# 单机：python3 server.py --local_hostname=localhost
if __name__ == "__main__":
    os.environ['MC_LOG_LEVEL'] = 'INFO'
    os.environ['MC_TE_METRIC'] = '1'
    os.environ['MC_CUSTOM_TOPO_JSON'] = ''

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-local",
        "--local_hostname",
        help="The server local hostname, .",
        default="localhost"
    )
    args = parser.parse_args()

    main(args.local_hostname)


'''
root@edge-l40s-1:/home/liuxiang/mooncake/multi-node# python3 server_tensor.py --local_hostname=10.252.112.16
local_hostname: 10.252.112.16
WARNING: Logging before InitGoogleLogging() is written to STDERR
I1018 00:45:47.942695 1844453 transfer_engine.cpp:91] Transfer Engine parseHostNameWithPort. server_name: 10.252.112.16 port: 12001
I1018 00:45:47.942726 1844453 transfer_engine.cpp:146] Transfer Engine RPC using P2P handshake, listening on 10.252.112.16:15546
I1018 00:45:47.942741 1844531 transfer_engine.cpp:493] Metrics reporting thread started (interval: 5s)
I1018 00:45:47.942777 1844453 transfer_engine.cpp:185] Auto-discovering topology...
I1018 00:45:47.942785 1844453 transfer_engine.cpp:188] Using custom topology from:
W1018 00:45:47.942795 1844453 transfer_engine.cpp:193] Failed to load custom topology from , falling back to auto-detect.
I1018 00:45:47.943269 1844453 transfer_engine.cpp:200] Topology discovery complete. Found 1 HCAs.
I1018 00:45:47.968216 1844453 rdma_context.cpp:491] Find best gid index: 0 on mlx5_ib1/
I1018 00:45:47.969202 1844453 rdma_context.cpp:129] RDMA device: mlx5_ib1, LID: 470, GID: (GID_Index 0) fe:80:00:00:00:00:00:00:a0:88:c2:03:00:17:87:71
Server initialized with session ID: 10.252.112.16:15546
Server buffer address: 140632108564480, length: 40
Buffer information sent to client
old tensors: tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], device='cuda:0')
wait for receiving data from client, sleep 3s...
new tensors: tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.], device='cuda:0')
received data from client successfully
I1018 00:46:08.998579 1844453 transfer_engine.cpp:535] Waiting for metrics reporting thread to join...
I1018 00:46:09.944137 1844531 transfer_engine.cpp:528] Metrics reporting thread stopped
I1018 00:46:09.944181 1844453 transfer_engine.cpp:537] Metrics reporting thread joined
root@edge-l40s-1:/home/liuxiang/mooncake/multi-node#

'''
