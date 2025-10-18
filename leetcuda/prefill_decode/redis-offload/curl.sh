


python benchmark_serving.py --port 9000 --seed $(date +%s) \
    --model qwen \
    --dataset-name random --random-input-len 7500 --random-output-len 200 \
    --num-prompts 30 --burstiness 100 --request-rate 1 --ignore-eos



curl http://127.0.0.1:8100/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{
         "model": "qwen",
         "messages": [
             {"role": "system", "content": "You are a helpful assistant."},
             {"role": "user", "content": "你是谁？"}
         ]
     }'
