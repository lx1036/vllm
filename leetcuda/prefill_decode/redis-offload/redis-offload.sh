

MODEL=/work/cache/Qwen3-0.6B



LMCACHE_CONFIG_FILE=redis-offload.yaml \
LMCACHE_USE_EXPERIMENTAL=True \
CUDA_VISIBLE_DEVICES=0 \
vllm serve $MODEL \
--served-model-name qwen \
--port 8100 \
--kv-transfer-config \
'{"kv_connector":"LMCacheConnector","kv_role":"kv_both"}'


