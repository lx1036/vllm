



# metadata service
mooncake_http_metadata_server --host 127.0.0.1 --port 8081

# master service
mooncake_master -http_metadata_server_port 8081 -http_metadata_server_host 127.0.0.1 -v=1
