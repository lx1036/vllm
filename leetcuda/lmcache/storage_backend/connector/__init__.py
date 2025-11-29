
import abc
import asyncio
import re
from dataclasses import dataclass
from typing import List, Dict, Optional

from leetcuda.lmcache.log import init_logger
from leetcuda.lmcache.memory_management import MemoryObj, MemoryAllocatorInterface
from leetcuda.lmcache.storage_backend.connector.base_connector import RemoteConnector
from leetcuda.lmcache.storage_backend.connector.blackhole_connector import BlackholeConnector
from leetcuda.lmcache.storage_backend.connector.lmcache_server_connector import LMCServerConnector
from leetcuda.lmcache.storage_backend.connector.redis_connector import RedisConnector
from leetcuda.lmcache.utils import CacheEngineKey

logger = init_logger(__name__)


@dataclass
class ParsedRemoteURL:
    """
    The parsed URL of the format:
    <connector_type>://<host>:<port>[/path][?query],<host2>:<port2>[/path2][?query2],...
    """

    connector_type: str
    hosts: List[str]
    ports: List[int]
    paths: List[str]
    query_params: List[Dict[str, str]]

def parse_remote_url(url: str) -> ParsedRemoteURL:
    """
    Parses the remote URL into its constituent parts with support for:
    - Multiple hosts (comma-separated)
    - Path and query parameters in each host definition
    - Forward compatibility with legacy format

    Raises:
        ValueError: If the URL is invalid.
    """
    pattern = r"(.+)://(.*)"
    m = re.match(pattern, url)
    if m is None:
        logger.error(f"Cannot parse remote_url {url} in the config")
        raise ValueError(f"Invalid remote url {url}")

    connector_type, hosts_section = m.groups()
    hosts = []
    ports = []
    paths = []
    query_params = []

    for host_def in hosts_section.split(","):
        host_pattern = r"""
                ^
                ([^:]+)        # hostname
                :              # :
                (\d+)          # port
                (/?[^?]*)      # path（optional, start with /）
                (?:\?(.*))?    # query（optional，? content after ?）
                $
            """
        match = re.match(host_pattern, host_def, re.VERBOSE)

        if not match:
            raise ValueError(
                f"Invalid host definition: {host_def} in URL: {url}")

        host = match.group(1)
        port = int(match.group(2))
        path = match.group(3).lstrip('/')
        path = path.lstrip('/')
        query_str = match.group(4) or ""

        params_dict = {}
        if query_str:
            for param in query_str.split('&'):
                if '=' in param:
                    key, value = param.split('=', 1)
                    params_dict[key] = value
                elif param:
                    params_dict[param] = ""

        hosts.append(host)
        ports.append(port)
        paths.append(path)
        query_params.append(params_dict)

    return ParsedRemoteURL(connector_type=connector_type,
                           hosts=hosts,
                           ports=ports,
                           paths=paths,
                           query_params=query_params)



def CreateConnector(url: str, loop: asyncio.AbstractEventLoop, memory_allocator: MemoryAllocatorInterface) -> RemoteConnector:


    parsed_url = parse_remote_url(url)
    num_hosts = len(parsed_url.hosts)

    connector_type = parsed_url.connector_type
    match connector_type:
        case "redis":
            connector = RedisConnector(host, port, loop, memory_allocator)

        case "lm":

            if num_hosts == 1:
                host, port = parsed_url.hosts[0], parsed_url.ports[0]
                connector = LMCServerConnector(host, port, loop, memory_allocator)
            else:
                raise ValueError(f"LM connector only supports a single host, but got url:{url}")


        # case "infinistore":


        # case "mooncakestore":

        case "blackhole":
            connector = BlackholeConnector(memory_allocator)

        case _:
            raise ValueError(f"Unknown connector type {connector_type} (url is: {url})")


    logger.info(f"Created connector {connector} for {connector_type}")
    return connector
