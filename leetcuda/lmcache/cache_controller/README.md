

# KVCache Management

## 运行测试
```shell
# 启动 lmcache kvcache management server
python3 -m lmcache.v1.api_server
```


## health api

```shell
#PYTHONHASHSEED=123 python3 -m lmcache.v1.api_server --host localhost --port 9000 --monitor-port 9001
PYTHONHASHSEED=123 python3 -m lmcache.v1.api_server --host localhost --port 9000 --monitor-ports '{"pull": 8300, "reply": 8400}'

curl -X POST http://localhost:9000/health \
  -H "Content-Type: application/json" \
  -d '{"instance_id": "lmcache_default_instance"}'

#{"event_id": "health47ce328d-f27e-48ae-ab0c-c2218aabce95", "error_codes": {"0": 0, "1": 0}}
{"event_id":"health09fde91c-d95d-438c-9e84-c93440ba769d","error_codes":{}}
```
