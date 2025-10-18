

# 安装

```shell
# cuda 环境下安装，cpu没法安装
pip3 install ucx-py-cu12
```



```shell
ibdev2netdev
show_gids

CUDA_VISIBLE_DEVICES=0 UCX_NET_DEVICES=mlx5_ib0:1 UCX_TLS=rc,cuda_copy ucx_perftest -t tag_bw -m cuda -s 10000000 -n 10 -p 9999 & \
CUDA_VISIBLE_DEVICES=1 UCX_NET_DEVICES=mlx5_ib0:1 UCX_TLS=rc,cuda_copy ucx_perftest `hostname` -t tag_bw -m cuda -s 100000000 -n 10 -p 9999
CUDA_VISIBLE_DEVICES=1 UCX_NET_DEVICES=mlx5_ib2:1 UCX_TLS=rc,cuda_copy ucx_perftest `hostname` -t tag_bw -m cuda -s 100000000 -n 10 -p 9999


CUDA_VISIBLE_DEVICES=0 UCX_NET_DEVICES=mlx5_ib0:1 UCX_TLS=rc,cuda_copy ucx_perftest -t tag_bw -m cuda -s 10000000 -n 10 -p 9999 & \
CUDA_VISIBLE_DEVICES=1 UCX_NET_DEVICES=mlx5_ib1:1 UCX_TLS=rc,cuda_copy ucx_perftest `hostname` -t tag_bw -m cuda -s 100000000 -n 10 -p 9999
CUDA_VISIBLE_DEVICES=1 UCX_NET_DEVICES=mlx5_ib3:1 UCX_TLS=rc,cuda_copy ucx_perftest `hostname` -t tag_bw -m cuda -s 100000000 -n 10 -p 9999


```

