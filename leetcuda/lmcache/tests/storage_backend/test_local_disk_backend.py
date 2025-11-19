import os
import unittest

from leetcuda.lmcache.storage_backend.local_disk_backend import LocalDiskBackend


class TestLocalDiskBackend2(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here

if __name__ == '__main__':
    unittest.main()




class TestLocalDiskBackend:

    def test_init(self, temp_disk_path, async_loop, local_cpu_backend):
        config = create_test_config(temp_disk_path)
        backend = LocalDiskBackend(
            config=config,
            loop=async_loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cuda",
        )

        assert backend.dst_device == "cuda"
        assert backend.local_cpu_backend == local_cpu_backend
        assert backend.path == temp_disk_path
        assert os.path.exists(temp_disk_path)
        assert backend.lmcache_worker is None
        assert backend.instance_id == "test_instance"
        assert backend.usage == 0
        assert len(backend.dict) == 0

        local_cpu_backend.memory_allocator.close()


