





pip3 install lmcache==0.3.6 nixl pandas datasets numpy==1.26.4 modelscope==1.18.1 transformers==4.51.3 flash_attn==2.5.9.post1 vllm==v0.9.1


###############  安装 vllm==0.8.4 ###############
pip3 install lmcache==0.2.1 vllm==0.8.4 mooncake-transfer-engine==0.3.6.post1

# 需要卸载deepspeed，否则报错 找不到 /usr/local/cuda/bin/nvcc
pip3 uninstall deepspeed

