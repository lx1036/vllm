


MODEL=/work/cache/Qwen3-0.6B

# Decoder listens on port 8200
#decode_config_file=lmcache-decoder-config.yaml
#
#UCX_TLS=cuda_ipc,cuda_copy,tcp \
#    LMCACHE_CONFIG_FILE=$decode_config_file \
#    VLLM_ENABLE_V1_MULTIPROCESSING=1 \
#    VLLM_WORKER_MULTIPROC_METHOD=spawn \
#    CUDA_VISIBLE_DEVICES=1 \
#    vllm serve $MODEL \
#    --port 8200 \
#    --disable-log-requests \
#    --enforce-eager \
#    --kv-transfer-config \
#    '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_consumer","kv_connector_extra_config": {"discard_partial_chunks": false, "lmcache_rpc_port": "consumer1"}}'


LMCACHE_CONFIG_FILE=lmcache-decoder-config.yaml \
LMCACHE_USE_EXPERIMENTAL=True \
CUDA_VISIBLE_DEVICES=1 \
vllm serve $MODEL \
--served-model-name qwen \
--port 8200 \
--kv-transfer-config \
'{"kv_connector":"LMCacheConnector","kv_role":"kv_consumer","kv_connector_extra_config": {"discard_partial_chunks": false, "lmcache_rpc_port": "consumer1"}}'
