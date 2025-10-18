import argparse
import logging
import os

import numpy as np
import zmq

from mooncake.engine import TransferEngine, TransferOpcode
import torch
from safetensors.torch import load as safetensors_load
from safetensors.torch import save as safetensors_save

# vllm/distributed/kv_transfer/kv_pipe/mooncake_pipe.py


logger = logging.getLogger(__name__)


def main(local_hostname: str, remote_hostname: str):
    print(f"local_hostname: {local_hostname}, remote_hostname: {remote_hostname}")

    # Initialize ZMQ context and socket
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.connect(f"tcp://{remote_hostname}:5555")
    print("Waiting for server buffer information...")
    buffer_info = socket.recv_json()
    server_session_id = buffer_info["session_id"]
    server_ptr = buffer_info["ptr"]
    server_len = buffer_info["len"]
    print(f"Received server info - Session ID: {server_session_id}")
    print(f"Server buffer address: {server_ptr}, length: {server_len}")

    # Initialize client engine
    HOSTNAME = local_hostname #"10.252.112.17" # localhost for simple demo
    METADATA_SERVER = "P2PHANDSHAKE" # [ETCD_SERVER_URL, P2PHANDSHAKE, ...]
    PROTOCOL = "rdma" # [rdma, tcp, ...]
    DEVICE_NAME = "mlx5_ib1" # auto discovery if empty
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

    # Allocate and initialize client buffer (1MB)
    client_buffer = torch.ones(10, device="cuda:0")
    # user_data = safetensors_save({"tensor": client_buffer})
    # client_len = len(user_data)
    # client_ptr = engine.allocate_managed_buffer(client_len)
    # if client_ptr <= 0:
    #     print("Allocation Return Error")
    #     raise Exception("Allocation Return Error")

    client_ptr = client_buffer.data_ptr()
    # client_len = client_buffer.size().numel()
    client_len = client_buffer.numel() * client_buffer.element_size() # float32 * 10=40B
    print(f"Client buffer address: {client_ptr}, length: {client_len}")

    # client_buffer = np.ones(1024 * 1024, dtype=np.uint8)  # Fill with ones
    # client_ptr = client_buffer.ctypes.data
    # client_len = client_buffer.nbytes

    # Register memory with Mooncake
    engine.register_memory(client_ptr, client_len)

    # session_id = f"localhost:{engine.get_rpc_port()}"
    # print(f"Client initialized with session ID: {session_id}")

    # Transfer data from client to server
    print("Transferring data to server...")
    # tensors: tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.], device='cuda:0')
    print(f"tensors: {client_buffer}")
    # transfer_sync_write()
    ret = engine.transfer_sync(
        server_session_id,
        client_ptr, # client_ptr 指向的内存数据 赋值给 server_ptr 的数据
        server_ptr,
        min(client_len, server_len),  # Transfer minimum of both lengths
        TransferOpcode.Write
    )
    if ret >= 0:
        print("Transfer successful!")
    else:
        print("Transfer failed!")

    # Cleanup
    engine.unregister_memory(client_ptr)
    socket.close()
    context.term()


# python3 client.py --local_hostname=10.252.112.17 --remote_hostname=10.252.112.16
# python3 client.py --local_hostname=localhost --remote_hostname=localhost
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
    parser.add_argument(
        "-remote",
        "--remote_hostname",
        help="The server local hostname, .",
        default="localhost"
    )
    args = parser.parse_args()

    main(args.local_hostname, args.remote_hostname)


'''
root@edge-l40s-2:/home/liuxiang/mooncake/multi-node# python3 client_tensor.py --local_hostname=10.252.112.17 --remote_hostname=10.252.112.16
local_hostname: 10.252.112.17, remote_hostname: 10.252.112.16
Waiting for server buffer information...
Received server info - Session ID: 10.252.112.16:15546
Server buffer address: 140632108564480, length: 40
WARNING: Logging before InitGoogleLogging() is written to STDERR
I1018 00:46:05.891112 2913040 transfer_engine.cpp:91] Transfer Engine parseHostNameWithPort. server_name: 10.252.112.17 port: 12001
I1018 00:46:05.891155 2913040 transfer_engine.cpp:146] Transfer Engine RPC using P2P handshake, listening on 10.252.112.17:15892
I1018 00:46:05.891147 2913114 transfer_engine.cpp:493] Metrics reporting thread started (interval: 5s)
I1018 00:46:05.891203 2913040 transfer_engine.cpp:185] Auto-discovering topology...
I1018 00:46:05.891208 2913040 transfer_engine.cpp:188] Using custom topology from:
W1018 00:46:05.891216 2913040 transfer_engine.cpp:193] Failed to load custom topology from , falling back to auto-detect.
I1018 00:46:05.891713 2913040 transfer_engine.cpp:200] Topology discovery complete. Found 1 HCAs.
I1018 00:46:05.914466 2913040 rdma_context.cpp:491] Find best gid index: 0 on mlx5_ib1/
I1018 00:46:05.915460 2913040 rdma_context.cpp:129] RDMA device: mlx5_ib1, LID: 519, GID: (GID_Index 0) fe:80:00:00:00:00:00:00:a0:88:c2:03:00:6b:24:c9
Client buffer address: 140073192390656, length: 40
Transferring data to server...
tensors: tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.], device='cuda:0')
Transfer successful!
I1018 00:46:07.274418 2913040 transfer_engine.cpp:535] Waiting for metrics reporting thread to join...
I1018 00:46:07.891306 2913114 transfer_engine.cpp:528] Metrics reporting thread stopped
I1018 00:46:07.891341 2913040 transfer_engine.cpp:537] Metrics reporting thread joined
root@edge-l40s-2:/home/liuxiang/mooncake/multi-node#
'''
