


import sys
import os
from mooncake.engine import TransferEngine

def main():
    os.environ['MC_LOG_LEVEL'] = 'INFO'
    os.environ['MC_TE_METRIC'] = '1'
    os.environ['MC_CUSTOM_TOPO_JSON'] = ''
    engine = TransferEngine()
    print('Local topology: ', engine.get_local_topology())

if __name__ == "__main__":
    sys.exit(main())


'''
root@edge-l40s-1:/home/liuxiang/mooncake# python3 transfer_engine_topology.py
WARNING: Logging before InitGoogleLogging() is written to STDERR
I1017 11:38:42.199270 1316125 transfer_engine.cpp:91] Transfer Engine parseHostNameWithPort. server_name:  port: 12001
I1017 11:38:42.199313 1316125 transfer_engine.cpp:146] Transfer Engine RPC using P2P handshake, listening on :16683
I1017 11:38:42.199306 1316127 transfer_engine.cpp:493] Metrics reporting thread started (interval: 5s)
I1017 11:38:42.199374 1316125 transfer_engine.cpp:185] Auto-discovering topology...
I1017 11:38:42.199380 1316125 transfer_engine.cpp:188] Using custom topology from:
W1017 11:38:42.199389 1316125 transfer_engine.cpp:193] Failed to load custom topology from , falling back to auto-detect.
I1017 11:38:42.200111 1316125 transfer_engine.cpp:200] Topology discovery complete. Found 8 HCAs.
I1017 11:38:42.227402 1316125 rdma_context.cpp:491] Find best gid index: 0 on mlx5_ib2/
I1017 11:38:42.228183 1316125 rdma_context.cpp:129] RDMA device: mlx5_ib2, LID: 468, GID: (GID_Index 0) fe:80:00:00:00:00:00:00:a0:88:c2:03:00:17:8c:70
I1017 11:38:42.251226 1316125 rdma_context.cpp:491] Find best gid index: 0 on mlx5_ib3/
I1017 11:38:42.251937 1316125 rdma_context.cpp:129] RDMA device: mlx5_ib3, LID: 469, GID: (GID_Index 0) fe:80:00:00:00:00:00:00:a0:88:c2:03:00:17:8c:71
I1017 11:38:42.275285 1316125 rdma_context.cpp:491] Find best gid index: 3 on mlx5_bond_0/
I1017 11:38:42.276250 1316125 rdma_context.cpp:129] RDMA device: mlx5_bond_0, LID: 0, GID: (GID_Index 3) 00:00:00:00:00:00:00:00:00:00:ff:ff:0a:fc:70:10
I1017 11:38:42.299132 1316125 rdma_context.cpp:491] Find best gid index: 0 on mlx5_ib0/
I1017 11:38:42.299819 1316125 rdma_context.cpp:129] RDMA device: mlx5_ib0, LID: 467, GID: (GID_Index 0) fe:80:00:00:00:00:00:00:a0:88:c2:03:00:17:87:70
I1017 11:38:42.323100 1316125 rdma_context.cpp:491] Find best gid index: 0 on mlx5_ib1/
I1017 11:38:42.323796 1316125 rdma_context.cpp:129] RDMA device: mlx5_ib1, LID: 470, GID: (GID_Index 0) fe:80:00:00:00:00:00:00:a0:88:c2:03:00:17:87:71
I1017 11:38:42.347209 1316125 rdma_context.cpp:491] Find best gid index: 0 on mlx5_storage0/
I1017 11:38:42.347985 1316125 rdma_context.cpp:129] RDMA device: mlx5_storage0, LID: 16, GID: (GID_Index 0) fe:80:00:00:00:00:00:00:a0:88:c2:03:00:2c:c9:10
W1017 11:38:42.371016 1316125 rdma_context.cpp:433] Device mlx5_0 port not active
E1017 11:38:42.371371 1316125 rdma_context.cpp:70] Failed to open device mlx5_0 on port  with GID 0
W1017 11:38:42.371390 1316125 rdma_transport.cpp:420] Disable device mlx5_0
W1017 11:38:42.375350 1316125 rdma_context.cpp:433] Device mlx5_1 port not active
E1017 11:38:42.375675 1316125 rdma_context.cpp:70] Failed to open device mlx5_1 on port  with GID 0
W1017 11:38:42.375686 1316125 rdma_transport.cpp:420] Disable device mlx5_1
I1017 11:38:42.375751 1316125 transfer_engine.cpp:535] Waiting for metrics reporting thread to join...
I1017 11:38:43.199393 1316127 transfer_engine.cpp:528] Metrics reporting thread stopped
I1017 11:38:43.199424 1316125 transfer_engine.cpp:537] Metrics reporting thread joined
Local topology:  {
	"cpu:0" :
	[
		[
			"mlx5_ib0",
			"mlx5_ib1",
			"mlx5_storage0"
		],
		[
			"mlx5_ib2",
			"mlx5_ib3",
			"mlx5_bond_0"
		]
	],
	"cpu:1" :
	[
		[
			"mlx5_ib2",
			"mlx5_ib3",
			"mlx5_bond_0"
		],
		[
			"mlx5_ib0",
			"mlx5_ib1",
			"mlx5_storage0"
		]
	]
}
'''
