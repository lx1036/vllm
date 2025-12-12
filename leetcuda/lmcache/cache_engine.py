import time
from typing import Optional, Dict, List, Union, Tuple, Generator

import torch

from leetcuda.lmcache.cache_controller.worker import LMCacheWorker
from leetcuda.lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata
from leetcuda.lmcache.gpu_connector import GPUConnectorInterface
from leetcuda.lmcache.log import init_logger
from leetcuda.lmcache.lookup_server.abstract_server import LookupServerInterface
from leetcuda.lmcache.memory_management import MemoryAllocatorInterface, MixedMemoryAllocator
from leetcuda.lmcache.observability import LMCStatsMonitor, LMCacheStatsLogger
from leetcuda.lmcache.storage_backend.storage_manager import StorageManager
from leetcuda.lmcache.token_database import TokenDatabase, ChunkedTokenDatabase
from leetcuda.lmcache.utils import _lmcache_nvtx_annotate, CacheEngineKey

logger = init_logger(__name__)

class LMCacheEngine:
    """
    KVCache -> MemoryObjs -> async StorageBackends

    MemoryObjs from Engine -> (GPUConnectors) -> KVCache
    """

    def __init__(self, config: LMCacheEngineConfig,
                 metadata: LMCacheEngineMetadata,
                 memory_allocator: MemoryAllocatorInterface,
                 token_database: TokenDatabase,
                 gpu_connector: GPUConnectorInterface,):
        logger.info(f"Creating LMCacheEngine with config: {config}")
        self.config = config
        self.token_database = token_database
        self.memory_allocator = memory_allocator
        self.lookup_server: Optional[LookupServerInterface] = None
        self.gpu_connector = gpu_connector

        self.lmcache_worker: Optional[LMCacheWorker] = None
        if self.config.enable_controller:
            self.lmcache_worker = LMCacheWorker(config, metadata, self)

        self.storage_manager = StorageManager(config, metadata, self.memory_allocator, self.lmcache_worker, self.lookup_server)



    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def store(
        self,
        tokens: torch.Tensor,
        hashes: Optional[List[int]] = None,
        offsets: Optional[List[int]] = None,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Store the tokens/hashes and mask into the cache engine.
        Format: either 'huggingface' or 'vllm'

                For huggingface,
                it should have the shape of
                [num_heads, num_tokens, head_size]

                For vllm,
                it should have the shape of
                [num_tokens, num_heads, head_size]

        :param tokens:
        :param mask:
        :param kwargs:
        :return:
        """
        if self.is_passive():
            logger.debug(f"rank={self.metadata.worker_id} ignore store")
            return

        if mask is not None:
            num_to_store_tokens = torch.sum(mask).item()
        elif tokens is not None:
            num_to_store_tokens = len(tokens)
        elif hashes is not None:
            assert offsets is not None, "Offsets should be set when hashes are provided during store"
            num_to_store_tokens = sum(offsets)
            kwargs["slot_mapping"] = torch.tensor(kwargs["slot_mapping"], dtype=torch.long, device="cuda")

        assert tokens is not None or hashes is not None, "Either 'tokens' or 'hashes' must be provided."

        monitor_req_id = self.stats_monitor.on_store_request(num_to_store_tokens)
        starts = []
        ends = []
        keys = []
        memory_objs = []
        offload_time = 0.0
        put_time = 0.0
        tot_kv_size = 0
        tot_token_num = 0
        t = time.perf_counter()

        request_configs = kwargs.get("request_configs")
        if request_configs is not None and len(request_configs) != 0:
            assert isinstance(request_configs, dict)

        for start, end, key in self.token_database.process_tokens(tokens, hashes, offsets, mask, request_configs=request_configs,):
            assert isinstance(key, CacheEngineKey)
            # Allocate the memory object
            num_tokens = end - start
            kv_shape = self.gpu_connector.get_shape(num_tokens)
            kv_dtype = self.metadata.kv_dtype
            # TODO (Jiayi): should be batched in the future
            memory_obj = self.storage_manager.allocate(kv_shape, kv_dtype, busy_loop=self.force_store_wait)
            if memory_obj is None:
                logger.warning("Local cpu memory under pressure so choosing to not store the KV cache.")
                break

            starts.append(start)
            ends.append(end)
            keys.append(key)
            memory_objs.append(memory_obj)
            tot_kv_size += memory_obj.get_size()
            tot_token_num += num_tokens

        # memory_objs might be empty, directly return to avoid sending tokens
        if not memory_objs:
            return
        self.gpu_connector.batched_from_gpu(memory_objs, starts, ends, **kwargs)
        offload_time += time.perf_counter() - t
        t = time.perf_counter()
        transfer_spec = kwargs.get("transfer_spec", None)
        self.storage_manager.batched_put(keys, memory_objs, transfer_spec=transfer_spec)
        put_time += time.perf_counter() - t
        tot_time = offload_time + put_time
        logger.info(
            "Stored %d out of total %d tokens. size: %.4f gb, cost %.4f ms, "
            "throughput: %.4f GB/s; offload_time: %.4f ms, put_time: %.4f ms",
            tot_token_num,
            num_to_store_tokens,
            tot_kv_size / 1024**3,
            tot_time * 1000,
            tot_kv_size / tot_time / 1024**3,
            offload_time * 1000,
            put_time * 1000,
        )

        self.stats_monitor.on_store_finished(monitor_req_id, tot_token_num)

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def store_layer(
            self,
            tokens: Union[torch.Tensor, list[int]],
            mask: Optional[torch.Tensor] = None,
            **kwargs,
    ) -> Generator[None, None, None]:
        """
        Store the KV cache in a layerwise manner.

        :param tokens:
        :param mask:
        :param kwargs:
        :return: A generator that yields None. In the first iteration, the
            generator allocates the memory objects for all layers and moves
            the KV cache of the first layer from GPU to CPU. In the next
            iterations, it moves the KV cache of layer i from GPU to the memory
            objects (on CPU) and puts the memory objects of layer i-1 to the
            storage backends. In the last iteration, it puts the memory objects
            of the last layer to the storage backends.
        """
        if mask is not None:
            num_to_store_tokens = torch.sum(mask).item()
        else:
            num_to_store_tokens = len(tokens)
        monitor_req_id = self.stats_monitor.on_store_request(num_to_store_tokens)

        starts = []
        ends = []
        keys = []
        memory_objs = []
        tot_token_num = 0
        kv_dtype = self.metadata.kv_dtype
        request_configs = kwargs.get("request_configs")
        if request_configs is not None and len(request_configs) != 0:
            assert isinstance(request_configs, dict)

        for start, end, key in self.token_database.process_tokens(tokens=tokens, mask=mask, request_configs=request_configs):
            assert isinstance(key, CacheEngineKey)

            keys_multi_layer = key.split_layers(self.num_layers)
            # Only check the first layer
            if self.storage_manager.contains(keys_multi_layer[0]):
                continue

            # Allocate the memory object
            num_tokens = end - start
            kv_shape_single_layer = self.gpu_connector.get_shape(num_tokens)
            memory_objs_multi_layer = self.storage_manager.batched_allocate(
                kv_shape_single_layer,
                kv_dtype,
                batch_size=self.num_layers,
                fmt=self.fmt,
                busy_loop=self.force_store_wait,
            )

            if memory_objs_multi_layer is None:
                logger.warning("Local cpu memory under pressure so choosing to not store the KV cache.")
                break

            starts.append(start)
            ends.append(end)
            keys.append(keys_multi_layer)
            memory_objs.append(memory_objs_multi_layer)
            tot_token_num += num_tokens

        if keys:
            # Transpose the keys and memory objects into layer major format
            memory_objs = [list(row) for row in zip(*memory_objs, strict=False)]
            keys = [list(row) for row in zip(*keys, strict=False)]

            assert isinstance(
                self.gpu_connector,
                (
                    VLLMPagedMemLayerwiseGPUConnector,
                    VLLMBufferLayerwiseGPUConnector,
                    SGLangLayerwiseGPUConnector,
                ),
            )

            mem_obj_generator = self.gpu_connector.batched_from_gpu(memory_objs, starts, ends, **kwargs)
            next(mem_obj_generator)
            for layer_id in range(self.num_layers):
                yield
                next(mem_obj_generator)
                self.storage_manager.batched_put(keys[layer_id], memory_objs[layer_id])
        else:
            # If no cache are found, we still need to yield to avoid
            # `StopIteration`
            for layer_id in range(self.num_layers):
                yield

        self.stats_monitor.on_store_finished(monitor_req_id, tot_token_num)
        logger.debug(f"Stored {tot_token_num} out of total {len(tokens)} tokens")
        yield


    def lookup(self, tokens: Union[torch.Tensor, List[int]], search_range: Optional[List[str]] = None) -> int:
        """
        Checks the existence of KV cache of the tokens from the cache engine.

        :param tokens: the input tokens, with shape [seq_len]
        :param search_range:
        :return: An int indicating how many prefix tokens are cached.
        """
        end = 0
        for start, end, key in self.token_database.process_tokens(tokens):
            assert isinstance(key, CacheEngineKey)
            if not self.storage_manager.contains(key, search_range):
                return start

        return end


    def is_passive(self):
        """
        A 'passive' CacheEngine means that the node itself will not store/retrieve
        the data directly, but from the "active" worker (i.e., rank 0 in MLA)
        """
        return self.save_only_first_rank and not self.metadata.is_first_rank()

    def close(self) -> None:
        logger.info("LMCacheEngine closed.")



class LMCacheEngineBuilder:
    _instances: Dict[str, LMCacheEngine] = {}
    _cfgs: Dict[str, LMCacheEngineConfig] = {}
    _metadatas: Dict[str, LMCacheEngineMetadata] = {}
    _stat_loggers: Dict[str, LMCacheStatsLogger] = {}


    @classmethod
    def get_or_create(
            cls,
            instance_id: str,
            config: LMCacheEngineConfig,
            metadata: LMCacheEngineMetadata,
            gpu_connector:
            GPUConnectorInterface,  # gpu connectors is from outside
    ) -> LMCacheEngine:
        logger.info(f"Creating LMCacheEngine instance {instance_id}")
        if instance_id not in cls._instances:
            memory_allocator = cls.create_memory_allocator(config, metadata)
            token_database = cls.create_token_database(config, metadata)
            engine = LMCacheEngine(config, metadata, memory_allocator, token_database, gpu_connector)
            stat_logger = LMCacheStatsLogger(metadata, log_interval=10)
            cls._instances[instance_id] = engine
            cls._cfgs[instance_id] = config
            cls._metadatas[instance_id] = metadata
            cls._stat_loggers[instance_id] = stat_logger

            return engine

        else:
            if cls._cfgs[instance_id] != config or cls._metadatas[instance_id] != metadata:
                raise ValueError(f"Instance {instance_id} already exists with a different configuration or metadata.")

            return cls._instances[instance_id]

    @classmethod
    def get(cls, instance_id: str) -> Optional[LMCacheEngine]:
        return cls._instances.get(instance_id)



    @staticmethod
    def create_memory_allocator(config: LMCacheEngineConfig, metadata: LMCacheEngineMetadata) -> MemoryAllocatorInterface:
        max_local_cpu_size = config.max_local_cpu_size
        return MixedMemoryAllocator(int(max_local_cpu_size * 1024**3))


    @staticmethod
    def create_token_database(config: LMCacheEngineConfig, metadata: LMCacheEngineMetadata) -> TokenDatabase:
        return ChunkedTokenDatabase(config, metadata)


    @classmethod
    def destroy(cls, instance_id: str) -> None:
        """Close and delete the LMCacheEngine instance by the instance ID"""
        # TODO: unit test for this
        if instance_id in cls._instances:
            stat_logger = cls._stat_loggers[instance_id]
            stat_logger.shutdown()
            engine = cls._instances[instance_id]
            engine.close()
            cls._instances.pop(instance_id, None)
            cls._cfgs.pop(instance_id, None)
            cls._metadatas.pop(instance_id, None)
            cls._stat_loggers.pop(instance_id, None)
            LMCStatsMonitor.DestroyInstance()
