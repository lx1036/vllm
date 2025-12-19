import time
from collections import defaultdict
from typing import Optional, Dict, List, Union, Tuple, Generator

import torch

from leetcuda.lmcache.cache_controller.worker import LMCacheWorker
from leetcuda.lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata
from leetcuda.lmcache.gpu_connector import GPUConnectorInterface
from leetcuda.lmcache.log import init_logger
from leetcuda.lmcache.lookup_server.abstract_server import LookupServerInterface
from leetcuda.lmcache.memory_management import MemoryAllocatorInterface, MixedMemoryAllocator, MemoryFormat, MemoryObj
from leetcuda.lmcache.observability import LMCStatsMonitor, LMCacheStatsLogger
from leetcuda.lmcache.storage_backend.storage_manager import StorageManager
from leetcuda.lmcache.token_database import TokenDatabase, ChunkedTokenDatabase
from leetcuda.lmcache.types import CacheEngineKey
from leetcuda.lmcache.utils import _lmcache_nvtx_annotate

logger = init_logger(__name__)

class LMCacheEngine:
    """
    KVCache -> MemoryObjs -> async StorageBackends

    MemoryObjs from Engine -> (GPUConnectors) -> KVCache
    """

    def __init__(
        self, config: LMCacheEngineConfig,
        metadata: LMCacheEngineMetadata,
        memory_allocator: MemoryAllocatorInterface,
        token_database: TokenDatabase,
        gpu_connector: GPUConnectorInterface,
    ):
        logger.info(f"Creating LMCacheEngine with config: {config}")
        self.config = config
        self.token_database = token_database
        self.memory_allocator = memory_allocator
        self.lookup_server: Optional[LookupServerInterface] = None
        self.gpu_connector = gpu_connector
        self.async_loading = config.enable_async_loading


        self.lmcache_worker: Optional[LMCacheWorker] = None
        if self.config.enable_controller:
            self.lmcache_worker = LMCacheWorker(config, metadata, self)

        self.storage_manager = StorageManager(config, metadata, self.memory_allocator, self.lmcache_worker, self.lookup_server)

        self.use_layerwise = config.use_layerwise
        self.fmt = None
        if self.use_layerwise:
            if config.enable_blending:
                self.fmt = MemoryFormat.KV_2TD
            else:
                self.fmt = MemoryFormat.KV_T2D
        if metadata.use_mla:
            self.fmt = MemoryFormat.KV_MLA_FMT



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
        Store the tokens/hashes and mask into the local storage(local memory) or remote storage(redis, mooncake).

        :param Optional[torch.Tensor] tokens: The tokens of the corresponding KV caches.

        :param Optional[List[int]] hashes: The hashes of the corresponding KV caches.

        :param **kwargs: The additional arguments for the storage backend which
            will be passed into the gpu_connector.
            Should include KV cache specific information (e.g., paged KV buffer
            and the page tables).

        used:
        lmcache_engine.store(
            token_ids,
            mask=store_mask,
            kvcaches=kvcaches,
            slot_mapping=slot_mapping,
            offset=skip_leading_tokens,
            transfer_spec=request.disagg_spec,
            request_configs=request.request_configs,
        )
        """
        if mask is not None:
            num_to_store_tokens = torch.sum(mask).item()
        elif tokens is not None:
            num_to_store_tokens = len(tokens)
        elif hashes is not None:
            assert offsets is not None, "Offsets should be set when hashes are provided during store"
            num_to_store_tokens = sum(offsets)
            kwargs["slot_mapping"] = torch.tensor(kwargs["slot_mapping"], dtype=torch.long, device="cuda")

        assert tokens is not None or hashes is not None, "Either 'tokens' or 'hashes' must be provided."

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

            # 1. create local memory(memory_objs)
            memory_obj = self.storage_manager.allocate(kv_shape, kv_dtype, busy_loop=self.force_store_wait, fmt=self.fmt)
            if memory_obj is None: # 测试时有该报错，需要调大 max_local_cpu_size 值由默认5GB为20GB
                logger.warning("Local cpu memory under pressure so choosing to not store the KV cache.")
                break

            starts.append(start)
            ends.append(end)
            keys.append(key)
            memory_objs.append(memory_obj)
            tot_kv_size += memory_obj.get_size()
            tot_token_num += num_tokens

        if not memory_objs:
            return

        # 2. offload: GPU(HBM) -> local memory(memory_objs)
        self.gpu_connector.batched_from_gpu(memory_objs, starts, ends, **kwargs)
        offload_time += time.perf_counter() - t

        t = time.perf_counter()
        # 3. put: local memory(memory_objs) -> local/remote storage
        transfer_spec = kwargs.get("transfer_spec", None)
        self.storage_manager.batched_put(keys, memory_objs, transfer_spec=transfer_spec)
        put_time += time.perf_counter() - t
        tot_time = offload_time + put_time

        # Stored 24 out of total 24 tokens. size: 0.0015 gb, cost 1.0393 ms, throughput: 1.4094 GB/s; offload_time: 0.8446 ms, put_time: 0.1947 ms
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


    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def retrieve(
        self,
        tokens: Union[torch.Tensor, list[int]], # token_ids: list[int]
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Retrieve KVCaches from local/remote storage(local-memory/redis/mooncake)
        """

        ret_mask = torch.zeros(len(tokens), dtype=torch.bool, device="cpu")


        # 1. Retrieve KVCache from local/remote storage to local memory(memory_objs)
        reordered_chunks: List[Tuple[CacheEngineKey, MemoryObj, int, int]] = []
        if self.async_loading:
            reordered_chunks, tot_kv_size = self.async_process_tokens_internal(tokens, mask, ret_mask, **kwargs)
        else:
            reordered_chunks, tot_kv_size = self.process_tokens_internal(tokens, mask, ret_mask, **kwargs)

        # 2. Move KVCache from local memory to GPU(HBM)
        # For example, disk->gpu is faster than disk->cpu->gpu.
        # RDMA is another example.
        if len(reordered_chunks) > 0:
            _, memory_objs, starts, ends = zip(*reordered_chunks, strict=False)
            self.gpu_connector.batched_to_gpu(
                list(memory_objs), list(starts), list(ends), **kwargs
            )




        # Retrieved 133 out of 133 required tokens (from 133 total tokens). size: 0.0000 gb, cost 3.2318 ms, throughput: 0.0000 GB/s;
        logger.info(
            "Retrieved %d out of %d required tokens (from %d total tokens)."
            " size: %.4f gb,"
            " cost %.4f ms, throughput: %.4f GB/s;",
            retrieved_tokens,
            num_required_tokens,
            len(tokens),
            tot_kv_size / 1024**3,
            onload_time * 1000,
            tot_kv_size / onload_time / 1024**3 if onload_time > 0 else 0,
            )

        return ret_mask


    def process_tokens_internal(self, tokens, mask, ret_mask, **kwargs) -> tuple[list[tuple[CacheEngineKey, MemoryObj, int, int]], int]:

        reordered_chunks: list[tuple[CacheEngineKey, MemoryObj, int, int]] = []
        tot_kv_size = 0
        # location -> [(CacheEngineKey, start, end)]
        block_mapping: dict[str, list[tuple[CacheEngineKey, int, int]]] = defaultdict(list)

        for start, end, key in self.token_database.process_tokens(
                tokens=tokens,
                mask=mask,
                request_configs=request_configs,
        ):
            assert isinstance(key, CacheEngineKey)
            location = None
            if key in self.lookup_cache:
                # TODO(Jiayi): we can reduce the number of `contains` calls
                # by checking the lookup cache first (should be updated in `lookup`)
                pass
            else:
                location = self.storage_manager.contains(key)
                if location is None:
                    break

                # NOTE: Here we make the assumption that the underlying
                # storage backend support pin operation, and the memory
                # object is already pinned in the storage backend.
                ret_mask[start:end] = True

            assert location is not None
            block_mapping[location].append((key, start, end))





        for location, blocks in block_mapping.items():
            keys = [key for key, _, _ in blocks]
            memory_objs = self.storage_manager.batched_get(keys=keys, location=location)
            assert memory_objs is not None, "Failed to get memory objects from storage backend"

            for (key, start, end), memory_obj in zip(blocks, memory_objs, strict=False):
                if memory_obj is None:
                    logger.warning("The cache block is in the storage, but it can't be retrieved")
                    if last_failed_block_start is None or last_failed_block_start < start:
                        last_failed_block_start = start
                    break
                reordered_chunks.append((key, memory_obj, start, end))
                tot_kv_size += memory_obj.get_size()




        return reordered_chunks, tot_kv_size




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
