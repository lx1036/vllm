


#python3 disagg_proxy_server_first_token_from_prefiller.py \
#        --host localhost \
#        --port 9000 \
#        --prefiller-host localhost \
#        --prefiller-port 8100 \
#        --decoder-host localhost \
#        --decoder-port 8200


python3 disagg_proxy_server.py \
        --host 127.0.0.1 \
        --port 9000 \
        --prefiller-host localhost \
        --prefiller-port 8100 \
        --decoder-host localhost \
        --decoder-port 8200
