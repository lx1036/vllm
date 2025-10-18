



MODEL=/work/cache/Qwen3-0.6B

MOONCAKE_CONFIG_PATH=mooncake-config.json \
LMCACHE_CONFIG_FILE=lmcache-mooncake-config.yaml \
LMCACHE_USE_EXPERIMENTAL=True \
CUDA_VISIBLE_DEVICES=0 \
vllm serve $MODEL \
--served-model-name qwen \
--port 8100 \
--kv-transfer-config \
'{"kv_connector":"LMCacheConnector","kv_role":"kv_both"}'



# mooncake store service
python3 -m mooncake.mooncake_store_service --config=mooncake-config.json
