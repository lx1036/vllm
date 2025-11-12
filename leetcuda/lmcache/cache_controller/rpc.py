
import zmq
import zmq.asyncio

def get_zmq_context():
    return zmq.asyncio.Context.instance()


def get_zmq_socket(context, socket_path: str, protocol: str, role):
    """
    Create a ZeroMQ socket with the specified protocol and role.
    """
    socket_addr = f"{protocol}://{socket_path}"
    socket = context.socket(role)
    if role in [zmq.PUB, zmq.PUSH, zmq.REP]:  # type: ignore[attr-defined]
        socket.bind(socket_addr)
    elif role in [zmq.SUB, zmq.PULL, zmq.REQ]:  # type: ignore[attr-defined]
        socket.connect(socket_addr)
    else:
        raise ValueError(f"Invalid role: {role}")

    return socket




