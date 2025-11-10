




MODEL=/work/cache/Qwen3-0.6B

CUDA_VISIBLE_DEVICES=0 \
vllm serve $MODEL \
--served-model-name qwen \
--port 8100 \
--kv-events-config \
'{"enable_kv_cache_events": True, "publisher": "zmq", "replay_endpoint": "tcp://*:5558", "topic": "qwen-kvcache-event"}"}'
