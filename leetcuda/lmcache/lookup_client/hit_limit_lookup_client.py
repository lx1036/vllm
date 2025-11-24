from leetcuda.lmcache.log import init_logger
from leetcuda.lmcache.lookup_client.abstract_client import LookupClientInterface

logger = init_logger(__name__)


"""
HitLimitLookupClient now is used for test, when lookup is called, cal the cache hit,
- if the cache hit <= (1 - hit_miss_ratio), direct return the result
- if the cache hit > (1 - hit_miss_ratio), re-compute the result by hit_miss_ratio
"""


class HitLimitLookupClient(LookupClientInterface):



