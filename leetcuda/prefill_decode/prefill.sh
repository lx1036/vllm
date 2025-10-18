



MODEL=/work/cache/Qwen3-0.6B

# Prefiller listens on port 8100


#prefill_config_file=lmcache-prefiller-config.yaml
#
#UCX_TLS=cuda_ipc,cuda_copy,tcp \
#    LMCACHE_CONFIG_FILE=$prefill_config_file \
#    VLLM_ENABLE_V1_MULTIPROCESSING=1 \
#    VLLM_WORKER_MULTIPROC_METHOD=spawn \
#    CUDA_VISIBLE_DEVICES=0 \
#    vllm serve $MODEL \
#    --port 8100 \
#    --disable-log-requests \
#    --enforce-eager \
#    --kv-transfer-config \
#    '{"kv_connector":"LMCacheConnector","kv_role":"kv_producer","kv_connector_extra_config": {"discard_partial_chunks": false, "lmcache_rpc_port": "producer1"}}'



LMCACHE_CONFIG_FILE=lmcache-prefiller-config.yaml \
LMCACHE_USE_EXPERIMENTAL=True \
CUDA_VISIBLE_DEVICES=0 \
vllm serve $MODEL \
--served-model-name qwen \
--port 8100 \
--kv-transfer-config \
'{"kv_connector":"LMCacheConnector","kv_role":"kv_producer","kv_connector_extra_config": {"discard_partial_chunks": false, "lmcache_rpc_port": "producer1"}}'
