import pickle
import threading
import time
from dataclasses import dataclass

import zmq


@dataclass
class NixlRequest:
    """
    A dataclass to represent a request received from the remote peer.
    This can be used to encapsulate the request information.
    """
    keys: list[CacheEngineKey]
    metadatas: list[MemoryObjMetadata]

    def serialize(self) -> bytes:
        return pickle.dumps(self)

    @staticmethod
    def deserialize(s: bytes) -> "NixlRequest":
        return pickle.loads(s)


class NixlPipe:
    """
    An one-directional pipe to send the data from the sender to the receiver.
    """
    TRANSFER_BUFFER_SIZE = 128 * 1024 * 1024





class NixlChannel:
    """
    Provides the primitives to send the data and process the received data.
    It will have some internal threads to handle the data receiving.
    """

    def __init__(self):

        # Initialize the ZeroMQ context
        self._context = zmq.Context()
        self._side_channel = self._context.socket(zmq.PAIR)

        if nixl_config.role == NixlRole.SENDER:
            self._side_channel.connect("tcp://{}:{}".format(nixl_config.peer_host_name, nixl_config.peer_port))
            self._side_channel.setsockopt(zmq.LINGER, 0)
        else:
            self._side_channel.bind("tcp://{}:{}".format(nixl_config.peer_host_name, nixl_config.peer_port))
            self._side_channel.setsockopt(zmq.LINGER, 0)

        # Create NIXL Pipe
        self._pipe = NixlPipe(nixl_config, self._side_channel)

        # Add a timeout for the side channel
        if nixl_config.role == NixlRole.RECEIVER:
            self._side_channel.setsockopt(zmq.RCVTIMEO,5000) # Set a timeout for receiving to avoid blocking


        if nixl_config.role == NixlRole.RECEIVER:
            self._receiver_thread = threading.Thread(target=self._receiver_loop, daemon=True)
            self._receiver_thread.start()





    def prepare_send(self, keys: list[CacheEngineKey], metadatas: list[MemoryObjMetadata]):


        # Initialize connection using side channel
        request = NixlRequest(keys=keys, metadatas=metadatas)

        self._side_channel.send(request.serialize())
        logger.info("Sent the request with %d keys", len(request.keys))


    def _receiver_loop(self):
        poller = zmq.Poller()
        poller.register(self._side_channel, zmq.POLLIN)

        while self._running:
            try:
                # Wait for a request from the side channel with shorter timeout
                events = poller.poll(timeout=POLL_TIMEOUT_MS)
                if not events:
                    continue

                # 服务端server接收到数据
                msg = self._side_channel.recv()
                if not msg:
                    logger.warn("Received empty message on the side channel")
                    time.sleep(0.1)  # Avoid busy waiting
                    continue

                request = NixlRequest.deserialize(msg)
                logger.info("Received request with %d keys", len(request.keys))
                self._process_receive_transaction(keys=request.keys, metadatas=request.metadatas)

            except zmq.Again as e:
                # Handle the timeout when waiting for a message
                logger.debug("Timeout waiting for a message on the side channel: %s", str(e))
                continue

            except Exception as e:
                logger.error("Failed to process receiver loop: %s", str(e))
                if self._running:
                    time.sleep(0.01)
