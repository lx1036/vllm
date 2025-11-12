import socket
import time
import subprocess
import random
import shlex
from dataclasses import dataclass

import pytest


@dataclass
class LMCacheServerProcess:
    server_url: str
    server_process: object


@pytest.fixture(scope="module")
def lmserver_experimental_process(request):

    def ensure_connection(host, port):
        retries = 10
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        successful = False
        while retries > 0:
            retries -= 1
            try:
                print("Probing connection, remaining retries: ", retries)
                client_socket.connect((host, port))
                successful = True
                break
            except ConnectionRefusedError:
                time.sleep(1)
                print("Connection refused!")
                continue
            except Exception as e:
                print(f"other Exception: {e}")
                continue

        client_socket.close()
        return successful

    # Specify remote device
    device = request.param

    # Start the process
    max_retries = 5
    while max_retries > 0:
        max_retries -= 1
        port_number = random.randint(10000, 65500)
        print("Starting the lmcache experimental server process on port")
        proc = subprocess.Popen(
            shlex.split("python3 -m lmcache.experimental.server localhost "
                        f"{port_number} {device}"))

        # Wait for lmcache process to start
        time.sleep(5)

        successful = False
        if proc.poll() is not None:
            successful = True
        else:
            successful = ensure_connection("localhost", port_number)

        if not successful:
            proc.terminate()
            proc.wait()
        else:
            break

    # Yield control back to the test until it finishes
    server_url = f"lm://localhost:{port_number}"
    yield LMCacheServerProcess(server_url, proc)

    # Terminate the process
    proc.terminate()
    proc.wait()

    # Destroy remote disk path
    if device not in ["cpu"]:
        subprocess.run(shlex.split(f"rm -rf {device}"))
