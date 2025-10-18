



curl http://127.0.0.1:8100/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{
         "model": "qwen",
         "messages": [
             {"role": "system", "content": "You are a helpful assistant."},
             {"role": "user", "content": "你是谁？"}
         ]
     }'
