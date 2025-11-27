import socket

import zmq
import zmq.asyncio

from leetcuda.lmcache.log import init_logger

logger = init_logger(__name__)

def get_zmq_context(use_asyncio: bool = True):
    if use_asyncio:
        return zmq.asyncio.Context.instance()
    else:
        return zmq.Context.instance()


def get_zmq_socket(context, socket_path: str, protocol: str, role: zmq.SocketType, bind_or_connect: str):
    """
    Create a ZeroMQ socket with the specified protocol and role.
    """
    socket_addr = f"{protocol}://{socket_path}"
    socket = context.socket(role)
    if bind_or_connect == "bind":
        socket.bind(socket_addr)
    elif bind_or_connect == "connect":
        socket.connect(socket_addr)
    # if role in [zmq.PUB, zmq.PUSH, zmq.REP]:  # type: ignore[attr-defined]
    #     socket.bind(socket_addr)
    # elif role in [zmq.SUB, zmq.PULL, zmq.REQ]:  # type: ignore[attr-defined]
    #     socket.connect(socket_addr)
    else:
        raise ValueError(f"Invalid role: {role}")

    return socket

def close_zmq_socket(socket: zmq.asyncio.Socket, linger: int = 0) -> None:
    """
    Close a ZeroMQ socket cleanly.

    :param socket: The zmq.Socket to be closed.
    :param linger: LINGER period (in milliseconds).
    Default is 0 (drop immediately).
    """
    try:
        socket.setsockopt(zmq.LINGER, linger)  # type: ignore[attr-defined]
        socket.close()
    except Exception as e:
        logger.error(f"Warning: Failed to close socket cleanly: {e}")



def get_ip():
    """
    Get the local IP address of the machine.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # "Connect" to a public IP — just to determine local IP
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        logger.warning("Failed to get local IP address. Falling back to loopback address.")
        return "127.0.0.1"  # Fallback to loopback
    finally:
        s.close()

def test_get_ip():
    print(get_ip()) # 192.168.31.223

