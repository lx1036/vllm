from dataclasses import dataclass
from typing import Optional
import yaml
import re
import torch

@dataclass
class LMCacheEngineConfig:
    chunk_size: int
    local_cpu: bool
    max_local_cpu_size: float  # in GB

    # need to be assigned a non-zero
    # value even if local_cpu is disabled
    # local_disk: Optional[str]
    max_local_disk_size: float  # in GB

    remote_url: Optional[str]
    remote_serde: Optional[str]  # Can be "naive" or "cachegen"

    save_decode_cache: bool  # whether to store decode kv cache

    # Blending related configurations
    enable_blending: bool  # whether to enable blending
    blend_recompute_ratio: float  # the ratio of blending recompute
    blend_min_tokens: int  # the minimum number of tokens for blending
    blend_special_str: str = " # # "  # the separator for blending

    # P2P related configurations
    enable_p2p: bool = False  # whether to enable peer-to-peer sharing
    lookup_url: Optional[str] = None  # the url of the lookup server
    distributed_url: Optional[str] = None  # the url of the distributed server
    p2p_host: Optional[str] = None # the host of the lookup server
    p2p_init_ports: Optional[list[int]] = None

    # Error handling related configurations
    error_handling: bool = False  # whether to enable error handling

    # Controller related configurations
    enable_controller: Optional[bool] = False  # whether to enable controller
    # the id of the lmcache instance
    lmcache_instance_id: str = "lmcache_default_instance"
    # controller url
    controller_pull_url: Optional[str] = None
    controller_reply_url: Optional[str] = None
    lmcache_worker_ports: Optional[list[int]] = None
    # lmcache worker url
    # NOTE: port number will add `worker_id`
    lmcache_worker_url: Optional[str] = None
    # the lmcache_worker_heartbeat_time means that sending heartbeat periodically.
    lmcache_worker_heartbeat_time: Optional[int] = None
    lmcache_worker_heartbeat_delay_time: int = 10

    # (Optional) Nixl configurations
    # whether to enable Nixl
    enable_nixl: Optional[bool] = False
    # Role: sender or receiver
    nixl_role: Optional[str] = None
    # The url of the nixl peer
    nixl_peer_host: Optional[str] = None
    # The BASE port of the nixl peer, real port is nixl_peer_port + WORKER_RANK
    nixl_peer_port: Optional[int] = None
    # The transport buffer size of nixl in bytes
    nixl_buffer_size: Optional[int] = None
    # The device that nixl uses
    nixl_buffer_device: Optional[str] = None
    # HACK: explicit option to enable/disable nixl GC before it's mature enough
    nixl_enable_gc: Optional[bool] = False

    @staticmethod
    def from_file(file_path: str) -> "LMCacheEngineConfig":
        """
        Load the config from a yaml file
        """
        with open(file_path, "r") as fin:
            config = yaml.safe_load(fin)

        chunk_size = config.get("chunk_size", 256)

        local_cpu = config.get("local_cpu", True)
        max_local_cpu_size = config.get("max_local_cpu_size", 5)

        local_disk = config.get("local_disk", None)
        max_local_disk_size = config.get("max_local_disk_size", 5)

        remote_url = config.get("remote_url", None)
        remote_serde = config.get("remote_serde", "naive")

        save_decode_cache = config.get("save_decode_cache", False)

        enable_blending = config.get("enable_blending", False)
        blend_recompute_ratio = config.get("blend_recompute_ratio", 0.15)
        blend_min_tokens = config.get("blend_min_tokens", 256)
        blend_special_str = config.get("blend_special_str", " # # ")

        enable_p2p = config.get("enable_p2p", False)
        lookup_url = config.get("lookup_url", None)
        distributed_url = config.get("distributed_url", None)

        error_handling = config.get("error_handling", False)

        enable_controller = config.get("enable_controller", False)
        lmcache_instance_id = config.get("lmcache_instance_id","lmcache_default_instance")
        controller_url = config.get("controller_url", None)
        controller_pull_url = config.get("controller_pull_url", None)
        controller_reply_url = config.get("controller_reply_url", None)
        lmcache_worker_url = config.get("lmcache_worker_url", None)
        lmcache_worker_ports = config.get("lmcache_worker_ports", None)

        enable_nixl = config.get("enable_nixl", False)
        nixl_role = config.get("nixl_role", None)
        nixl_peer_host = config.get("nixl_peer_host", None)
        nixl_peer_port = config.get("nixl_peer_port", None)
        nixl_buffer_size = config.get("nixl_buffer_size", None)
        nixl_buffer_device = config.get("nixl_buffer_device", None)
        nixl_enable_gc = config.get("nixl_enable_gc", False)

        p2p_host = config.get("p2p_host", None)
        p2p_init_ports = config.get("p2p_init_ports", None)


        local_disk_path = None
        match local_disk:
            case None:
                local_disk_path = None
            case path if re.match(r"file://(.*)/", path):  # local disk directory
                local_disk_path = path[7:]

        match remote_url:
            case None:
                pass
            case url if re.match(r"(.*)://(.*):(\d+)", url):
                pass
            case _:
                raise ValueError(f"Invalid remote storage url: {remote_url}")

        return LMCacheEngineConfig(
            chunk_size=chunk_size,
            local_cpu=local_cpu,
            max_local_cpu_size=max_local_cpu_size,
            # local_disk_path=local_disk_path,
            max_local_disk_size=max_local_disk_size,
            remote_url=remote_url,
            remote_serde=remote_serde,
            save_decode_cache=save_decode_cache,
            enable_blending=enable_blending,
            blend_recompute_ratio=blend_recompute_ratio,
            blend_min_tokens=blend_min_tokens,
            blend_special_str=blend_special_str,
            enable_p2p=enable_p2p,
            p2p_host=p2p_host,
            p2p_init_ports=p2p_init_ports,
            lookup_url=lookup_url,
            distributed_url=distributed_url,
            error_handling=error_handling,
            enable_controller=enable_controller,
            lmcache_instance_id=lmcache_instance_id,
            # controller_url=controller_url,
            controller_pull_url=controller_pull_url,
            controller_reply_url=controller_reply_url,
            lmcache_worker_url=lmcache_worker_url,
            lmcache_worker_ports=lmcache_worker_ports,
            enable_nixl=enable_nixl,
            nixl_role=nixl_role,
            nixl_peer_host=nixl_peer_host,
            nixl_peer_port=nixl_peer_port,
            nixl_buffer_size=nixl_buffer_size,
            nixl_buffer_device=nixl_buffer_device,
            nixl_enable_gc=nixl_enable_gc,
        ).validate()

    @staticmethod
    def from_legacy(
            chunk_size: int = 256,
            backend: str = "cpu",
            remote_url: Optional[str] = "lm://localhost:65432",
            remote_serde: str = "naive",
            save_decode_cache: bool = False,
            enable_blending: bool = False,
            blend_recompute_ratio: float = 0.15,
            blend_min_tokens: int = 256,
            blend_special_str: str = " # # ",
            max_local_disk_size: float = 0.0,
            enable_p2p: bool = False,
            lookup_url: Optional[str] = None,
            distributed_url: Optional[str] = None,
            error_handling: bool = False,
    ) -> "LMCacheEngineConfig":
        # TODO (ApostaC): Add nixl config
        if backend == "cpu":
            local_cpu = True
            max_local_cpu_size = 5
            local_disk = None
            max_local_disk_size = 0
            remote_url = None
        elif backend == "local_disk":
            local_cpu = False
            max_local_cpu_size = 5
            local_disk = "/local/disk_test/local_disk/"
            max_local_disk_size = 5
            remote_url = None
        elif backend == "local_cpu_disk":
            local_cpu = True
            max_local_cpu_size = 5
            local_disk = "/local/disk_test/local_disk/"
            max_local_disk_size = 5
            remote_url = None
        elif backend == "remote":
            local_cpu = False
            max_local_cpu_size = 5
            local_disk = None
        elif backend == "local_cpu_remote":
            local_cpu = True
            max_local_cpu_size = 5
            local_disk = None
        elif backend == "local_disk_remote":
            local_cpu = False
            max_local_cpu_size = 5
            local_disk = "/local/disk_test/local_disk/"
            max_local_disk_size = 5
        elif backend == "local_cpu_disk_remote":
            local_cpu = True
            max_local_cpu_size = 5
            local_disk = "/local/disk_test/local_disk/"
            max_local_disk_size = 5
        else:
            raise ValueError(f"Invalid backend: {backend}")
        return LMCacheEngineConfig(chunk_size, local_cpu, max_local_cpu_size,
                                   local_disk, max_local_disk_size, remote_url,
                                   remote_serde, save_decode_cache,
                                   enable_blending, blend_recompute_ratio,
                                   blend_min_tokens, blend_special_str,
                                   enable_p2p, lookup_url, distributed_url,
                                   error_handling).validate()

    def validate(self) -> 'LMCacheEngineConfig':
        """Validate the config
        """
        if self.enable_p2p:
            assert self.lookup_url is not None
            assert self.distributed_url is not None

        if self.enable_nixl:
            assert self.nixl_role is not None
            assert self.nixl_peer_host is not None
            assert self.nixl_peer_port is not None
            assert self.nixl_buffer_size is not None
            assert self.nixl_buffer_device is not None
            assert self.nixl_enable_gc is not None
            assert self.local_cpu is False, "Nixl only supports local_cpu=False"
            assert self.max_local_cpu_size == 0, "Nixl only supports max_local_cpu_size=0"
            assert self.local_disk is None, "Nixl only supports local_disk=None"
            assert self.remote_url is None, "Nixl only supports remote_url=None"
            assert self.save_decode_cache is False, "Nixl only supports save_decode_cache=False"
            assert self.enable_p2p is False, "Nixl only supports enable_p2p=False"

        return self


@dataclass
class LMCacheEngineMetadata:
    """ name of the LLM model """
    model_name: str
    """ world size when running under a distributed setting """
    world_size: int
    """ worker id when running under a distributed setting """
    worker_id: int
    """ the format of kv tensors """
    fmt: str
    """ the data type of kv tensors """
    kv_dtype: torch.dtype
    """ the shape of kv tensors """
    """ (num_layer, 2, chunk_size, num_kv_head, head_size) """
    """
    num_layer: 模型层数
    2: k 和 v
    chunk_size: 当输入文本超过模型最大上下文长度（或为了优化计算效率）时，会将文本分块处理，chunk_size即每个块的 token 数量。常见于滑动窗口注意力、长上下文扩展等场景。
    num_kv_head: k 和 v 注意力头数量，即多少个 heads
    head_size: 每个注意力头的维度（即每个头中 Q/K/V 向量的长度）。模型的总隐藏维度（hidden_dim）通常等于 num_q_head × head_size（或 num_kv_head × head_size，取决于注意力机制）。
    """
    kv_shape: tuple[int, int, int, int, int]

# for test
def create_engine_metadata(kv_shape=(32, 2, 256, 8, 128)) -> LMCacheEngineMetadata:
    return LMCacheEngineMetadata(
        model_name="test_model",
        world_size=3,
        worker_id=1,
        fmt="vllm",
        kv_dtype=torch.bfloat16,
        kv_shape=kv_shape,
    )
