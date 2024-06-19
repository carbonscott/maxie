import traceback
import argparse
import signal
import json
import socket
from multiprocessing import shared_memory, Process
import numpy as np
from maxie.datasets.psana_utils import PsanaImg
from maxie.perf import Timer

import logging

logger = logging.getLogger(__name__)

# Initialize buffer for each process
psana_img_buffer = {}

def get_psana_img(exp, run, access_mode, detector_name):
    """
    Fetches a PsanaImg object for the given parameters, caching the object to avoid redundant initializations.
    """
    key = (exp, run)
    if key not in psana_img_buffer:
        psana_img_buffer[key] = PsanaImg(exp, run, access_mode, detector_name)
    return psana_img_buffer[key]

def worker_process(server_socket, timeout):
    # Ignore CTRL+C in the worker process
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    while True:
        try:
            # Accept a new connection
            connection, client_address = server_socket.accept()

            # Receive request data
            request_data = connection.recv(4096).decode('utf-8')

            if request_data == "DONE":
                logger.debug("Received shutdown signal. Shutting down server.")
                connection.close()
                break

            request_data = json.loads(request_data)
            exp           = request_data.get('exp')
            run           = request_data.get('run')
            access_mode   = request_data.get('access_mode')
            detector_name = request_data.get('detector_name')
            event         = request_data.get('event')
            mode          = request_data.get('mode')

            # Send data or error message back to clients
            shm = None
            try:
                # Fetch psana image data
                psana_img = get_psana_img(exp, run, access_mode, detector_name)
                with Timer(tag=None, is_on=True) as t:
                    ## data = psana_img.get(event, None, mode)
                    data = psana_img.get_masked(event, None, mode, returns_assemble = True, edge_width = 1)

                if data is None:
                    raise ValueError(f"Received None from exp={exp}, run={run}, event={event}!!!")

                # Keep numpy array in a shared memory
                shm = shared_memory.SharedMemory(create=True, size=data.nbytes)
                shared_array = np.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf)
                shared_array[:] = data

                response_data = json.dumps({
                    'name': shm.name,
                    'shape': data.shape,
                    'dtype': str(data.dtype)
                })

                # Send response with shared memory details
                connection.sendall(response_data.encode('utf-8'))

            except Exception as e:
                error_message = {
                    "error": str(e),
                    "traceback": None if isinstance(e, ValueError) else traceback.format_exc()
                }
                connection.sendall(json.dumps(error_message).encode('utf-8'))

            # Wait for the client's acknowledgment
            connection.settimeout(timeout)
            try:
                ack = connection.recv(1024).decode('utf-8')
                if ack == "ACK":
                    if shm is not None:
                        logger.debug(f"Shared memory {shm.name} ready to unlink. Creation took {t.duration * 1e3} ms.")
                else:
                    logger.debug("Did not receive proper acknowledgment from client.")
            except:
                logger.debug("Did not receive ACK for error message within timeout.")

            if shm is not None: unlink_shared_memory(shm.name)

        except Exception as e:
            logger.debug(f"Unexpected error: {e}")
            continue

def unlink_shared_memory(shm_name):
    try:
        shm = shared_memory.SharedMemory(name=shm_name)
        shm.close()
        shm.unlink()
    except FileNotFoundError:
        pass

def start_server(address, num_workers, timeout):
    # Init TCP socket, set reuse, bind, and listen for connections
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(address)
    server_socket.listen()

    # Create and start worker processes
    processes = []
    for _ in range(num_workers):
        p = Process(target=worker_process, args=(server_socket, timeout))
        p.start()
        processes.append(p)

    return processes, server_socket

if __name__ == "__main__":
    logger.info(f"Launching IPC server script...")
    parser = argparse.ArgumentParser(description="Configure the server.")
    parser.add_argument("--hostname"   , type = str, default='localhost', help = "Server hostname (Default: 'localhost').")
    parser.add_argument("--port"       , type = int, default=5000       , help = "Server port (Default: 5000).")
    parser.add_argument("--num_workers", type = int, default=5          , help = "Number of workers supporting data serving (Default: 5).")
    parser.add_argument("--timeout",     type = int, default=5          , help = "Connection timeout (Default: 5 seconds).")
    args = parser.parse_args()

    hostname    = args.hostname
    port        = args.port
    num_workers = args.num_workers
    timeout     = args.timeout

    server_address = (hostname, port)
    processes, server_socket = start_server(server_address, num_workers, timeout)

    logger.info(f"Server is running at {hostname}:{port} with {num_workers} workers.")

    try:
        # Wait to complete, join is wait
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        logger.info("Shutdown signal received")

        for p in processes:
            # Trigger connection to unblock accept() in workers
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as trigger_socket:
                trigger_socket.connect(server_address)
                trigger_socket.sendall("DONE".encode('utf-8'))

        # Wait to complete, join is wait
        for p in processes:
            p.join()
        server_socket.close()
        logger.info("Server shutdown gracefully.")
