from leetcuda.lmcache.cache_controller.message import HealthMsg, HealthRetMsg


class RegistrationController:

    def __init__(self):
        # Mapping from `instance_id` -> `worker_ids`
        self.worker_mapping: dict[str, list[int]] = {}


    def post_init(self, kv_controller, cluster_executor):
        """
        Post initialization of the Registration Controller.
        """
        self.kv_controller = kv_controller
        self.cluster_executor = cluster_executor


    def get_workers(self, instance_id: str) -> list[int]:
        """
        Get worker ids given an instance id.
        """
        return self.worker_mapping.get(instance_id, [])

    async def health(self, msg: HealthMsg) -> HealthRetMsg:
        """
        Check the health of the lmcache worker.
        """
        return await self.cluster_executor.execute("health", msg)
