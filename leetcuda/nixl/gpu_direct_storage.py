import os
import sys

# https://docs.nvidia.com/gpudirect-storage/overview-guide/index.html

from nixl._api import nixl_agent, nixl_agent_config
from nixl.logging import get_logger
import nixl._utils as nixl_utils

logger = get_logger(__name__)

# 首先开发机里安装 nvidia-gds 包
# apt-get update -y && apt install -y nvidia-gds

# python3 gpu_direct_storage.py /work/cache/lmcache-demo/nixl/gds.txt

if __name__ == "__main__":
    if len(sys.argv) < 2:
        logger.error("Please specify file path in argv")
        exit(0)

    logger.info("Using NIXL Plugins from:\n%s", os.environ["NIXL_PLUGIN_DIR"])

    agent_config = nixl_agent_config(backends=[]) # not UCX
    nixl_agent1 = nixl_agent("GDSTester", agent_config)
    nixl_agent1.create_backend("GDS")

    plugin_list = nixl_agent1.get_plugin_list()
    assert "GDS" in plugin_list
    logger.info(
        "Plugin parameters:\n%s\n%s\n",
        nixl_agent1.get_plugin_mem_types("GDS"), # ['DRAM_SEG', 'VRAM_SEG', 'FILE_SEG']
        nixl_agent1.get_plugin_params("GDS"),
    )
    logger.info(
        "Backend parameters:\n%s\n%s\n",
        nixl_agent1.get_backend_mem_types("GDS"), # ['DRAM_SEG', 'VRAM_SEG', 'FILE_SEG']
        nixl_agent1.get_backend_params("GDS"),
    )

    buf_size = 16 * 4096 # 64K 个元素

    # get DRAM buf and initialize it to 0xba for verification
    addr1 = nixl_utils.malloc_passthru(buf_size)
    addr2 = nixl_utils.malloc_passthru(buf_size)
    nixl_utils.ba_buf(addr1, buf_size) # 64K 每个赋值0xba


    agent1_strings = [(addr1, buf_size, 0, "a"), (addr2, buf_size, 0, "b")]
    # Get nixlRegDList from different input types:: (address, len, device ID, meta_info)
    agent1_reg_descs = nixl_agent1.get_reg_descs(agent1_strings, "DRAM")
    # Get nixlXferDList from different input types: (address, len, device ID)
    agent1_xfer1_descs = nixl_agent1.get_xfer_descs([(addr1, buf_size, 0)], "DRAM")
    agent1_xfer2_descs = nixl_agent1.get_xfer_descs([(addr2, buf_size, 0)], "DRAM")

    assert nixl_agent1.register_memory(agent1_reg_descs) is not None

    # user must pass full file path for test
    agent1_fd = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT)
    assert agent1_fd >= 0

    agent1_file_list = [(0, buf_size, agent1_fd, "b")]
    agent1_file_descs = nixl_agent1.register_memory(agent1_file_list, "FILE")
    assert agent1_file_descs is not None
    agent1_xfer_files = agent1_file_descs.trim()

    ###### address1 WRITE -> agent1_fd File -> address2 #####

    # local(agent1_xfer1_descs) -> WRITE -> remote(agent1_xfer_files)
    xfer_handle_1 = nixl_agent1.initialize_xfer("WRITE", agent1_xfer1_descs, agent1_xfer_files, "GDSTester")
    if not xfer_handle_1:
        logger.error("Creating transfer failed.")
        exit()
    state = nixl_agent1.transfer(xfer_handle_1)
    assert state != "ERR"
    while True:
        # Check the state of a transfer operation
        state = nixl_agent1.check_xfer_state(xfer_handle_1)
        if state == "ERR":
            logger.error("Transfer got to Error state.")
            exit()
        elif state == "DONE":
            break

    # read file data back into second buffer
    # agent1_xfer2_descs <- READ <- agent1_xfer_files
    xfer_handle_2 = nixl_agent1.initialize_xfer("READ", agent1_xfer2_descs, agent1_xfer_files, "GDSTester")
    if not xfer_handle_2:
        logger.error("Creating transfer failed.")
        exit()
    state = nixl_agent1.transfer(xfer_handle_2)
    assert state != "ERR"
    while True:
        # Check the state of a transfer operation
        state = nixl_agent1.check_xfer_state(xfer_handle_2)
        if state == "ERR":
            logger.error("Transfer got to Error state.")
            exit()
        elif state == "DONE":
            break

    # transfer verification
    # 验证 addr1 64K 个元素 等于 addr2
    nixl_utils.verify_transfer(addr1, addr2, buf_size)

    # cleanup
    nixl_agent1.release_xfer_handle(xfer_handle_1)
    nixl_agent1.release_xfer_handle(xfer_handle_2)
    nixl_agent1.deregister_memory(agent1_reg_descs)
    nixl_agent1.deregister_memory(agent1_file_descs)
    nixl_utils.free_passthru(addr1)
    nixl_utils.free_passthru(addr2)

    os.close(agent1_fd)

    logger.info("Test Complete.")
