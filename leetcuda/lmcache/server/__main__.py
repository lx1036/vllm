import socket
import threading

from leetcuda.lmcache.server.server_storage_backend import CreateStorageBackend


class LMCacheServer:

    def __init__(self, host, port, device):
        self.host = host
        self.port = port
        # self.data_store = {}
        self.data_store = CreateStorageBackend(device)
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((host, port))
        self.server_socket.listen()


    def run(self):
        print(f"Server started at {self.host}:{self.port}")
        try:
            while True:
                client_socket, addr = self.server_socket.accept()
                print(f"Connected by {addr}")
                threading.Thread(target=self.handle_client, args=(client_socket, )).start()
        finally:
            self.server_socket.close()


    def handle_client(self, client_socket):






def main():
    import sys

    if len(sys.argv) not in [3, 4]:
        print(f"Usage: {sys.argv[0]} <host> <port> <storage>(default:cpu)")
        exit(1)

    host = sys.argv[1]
    port = int(sys.argv[2])
    if len(sys.argv) == 4:
        device = sys.argv[3]
    else:
        device = "cpu"

    server = LMCacheServer(host, port, device)
    server.run()


if __name__ == "__main__":
    main()
