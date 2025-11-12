from dataclasses import dataclass
from typing import Optional
import yaml
import re
import torch

@dataclass
class LMCacheEngineConfig:


    chunk_size: int

    max_local_cpu_size: float  # in GB
    local_cpu: bool

    # need to be assigned a non-zero
    # value even if local_cpu is disabled
    local_disk: Optional[str]
    max_local_disk_size: float  # in GB

    remote_url: Optional[str]
    remote_serde: Optional[str]  # Can be "naive" or "cachegen"


    # Controller related configurations
    enable_controller: Optional[bool] = False  # whether to enable controller
    # the id of the lmcache instance
    lmcache_instance_id: str = "lmcache_default_instance"
    # controller url
    controller_url: Optional[str] = None

    # P2P related configurations
    enable_p2p: bool = False  # whether to enable peer-to-peer sharing
    lookup_url: Optional[str] = None  # the url of the lookup server
    distributed_url: Optional[str] = None  # the url of the distributed server

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
        lmcache_instance_id = config.get("lmcache_instance_id",
                                         "lmcache_default_instance")
        controller_url = config.get("controller_url", None)
        lmcache_worker_url = config.get("lmcache_worker_url", None)

        enable_nixl = config.get("enable_nixl", False)
        nixl_role = config.get("nixl_role", None)
        nixl_peer_host = config.get("nixl_peer_host", None)
        nixl_peer_port = config.get("nixl_peer_port", None)
        nixl_buffer_size = config.get("nixl_buffer_size", None)
        nixl_buffer_device = config.get("nixl_buffer_device", None)
        nixl_enable_gc = config.get("nixl_enable_gc", False)

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
            chunk_size,
            local_cpu,
            max_local_cpu_size,
            local_disk_path,
            max_local_disk_size,
            remote_url,
            remote_serde,
            save_decode_cache,
            enable_blending,
            blend_recompute_ratio,
            blend_min_tokens,
            blend_special_str,
            enable_p2p,
            lookup_url,
            distributed_url,
            error_handling,
            enable_controller,
            lmcache_instance_id,
            controller_url,
            lmcache_worker_url,
            enable_nixl,
            nixl_role,
            nixl_peer_host,
            nixl_peer_port,
            nixl_buffer_size,
            nixl_buffer_device,
            nixl_enable_gc,
        ).validate()

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

            assert self.local_cpu is False, \
                "Nixl only supports local_cpu=False"
            assert self.max_local_cpu_size == 0, \
                "Nixl only supports max_local_cpu_size=0"

            assert self.local_disk is None, \
                "Nixl only supports local_disk=None"

            assert self.remote_url is None, \
                "Nixl only supports remote_url=None"

            assert self.save_decode_cache is False, \
                "Nixl only supports save_decode_cache=False"
            assert self.enable_p2p is False, \
                "Nixl only supports enable_p2p=False"

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
    kv_shape: tuple[int, int, int, int, int]

