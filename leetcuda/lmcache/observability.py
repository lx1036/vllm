from leetcuda.lmcache.utils import thread_safe


class LMCStatsMonitor:
    _instance = None

    @staticmethod
    def GetOrCreate() -> "LMCStatsMonitor":
        if LMCStatsMonitor._instance is None:
            LMCStatsMonitor._instance = LMCStatsMonitor()
        return LMCStatsMonitor._instance


    @thread_safe
    def update_local_storage_usage(self, usage: int):
        self.local_storage_usage_bytes = usage


