import unittest

import msgspec

from vllm.distributed.kv_events import EventBatch


class Event(msgspec.Struct, tag=True, array_like=True):
    id: int
    value: str

class EventBatchSample(EventBatch):



class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, True)  # add assertion here

    def create_test_events(self, count: int):






    def test_basic_publish(self):





if __name__ == '__main__':
    unittest.main()
