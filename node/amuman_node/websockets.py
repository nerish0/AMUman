import asyncio
import json
import logging
import socket
from typing import Optional, Union

import websockets
from pydantic import BaseModel

from amuman_node.api import API
from amuman_node.gpu_monitor import GPUMonitor, GPU
from amuman_node.job_manager import JobRunner

log = logging.getLogger("rich")


class WebsocketMessage(BaseModel):
    command: str
    node_id: int
    job_id: Optional[int] = None
    gpu_device_id: Optional[int] = None
    result: Optional[dict] = None


def parse_message(message: str) -> Union[WebsocketMessage, None]:
    try:
        data = json.loads(message)
        return WebsocketMessage(**data)
    except ValueError as e:
        log.error(f"Error parsing message: {e}")
        return None


class Websockets:
    def __init__(self, api: API, node_id: int, node_name: str, gpm: GPUMonitor) -> None:
        self.api: API = api
        self.node_name: str = node_name
        self.node_id: int = node_id
        self.ping_timeout = 10
        self.sleep_time = 5
        self.first_time_connection = 1
        self.gpm: GPUMonitor = gpm
        self.url = f"{self.api.url.replace('http','ws').replace('/api','')}/ws/node/?node_id={self.node_id}"
        log.debug(f"Websocket URL: {self.url}")
        self.ws: websockets.WebSocketClientProtocol
        self.current_job_runner: Optional[JobRunner] = None
        self.ftc = 0

    async def websocket_loop(self):
        # asyncio.create_task(self.monitor_gpus())
        while True:
            log.debug("WEBSOCKET: starting connection loop...")
            try:
                async with websockets.connect(
                    self.url, extra_headers=self.api.headers
                ) as ws:
                    self.ws = ws
                    if self.ftc == 0:
                        await self.register(ws)
                        self.ftc = 1
                    else:
                        await self.reconnect(ws)
                    while True:
                        if not await self.handle_connection_errors(ws):
                            break
                        await self.handle_connection(ws)
            except (socket.gaierror, ConnectionRefusedError) as e:
                error_msg = (
                    "Socket error - retrying connection"
                    if isinstance(e, socket.gaierror)
                    else "Nobody seems to listen to this endpoint. Please check the URL."
                )
                log.debug(
                    f"{error_msg} in {self.sleep_time} sec (Ctrl-C to quit)")
                await asyncio.sleep(self.sleep_time)
            except websockets.exceptions.InvalidStatusCode as e:
                log.error(f"Invalid status code: {e}")
                await asyncio.sleep(self.sleep_time)

    async def reconnect(self, ws: websockets.WebSocketClientProtocol) -> None:
        await ws.send(
            WebsocketMessage(command="register", node_id=self.node_id).json()
        )
        log.info("WEBSOCKET: Reconnecting.")
        await self.send_current_job_status(ws)

    async def register(self, ws: websockets.WebSocketClientProtocol) -> None:
        log.info("WEBSOCKET: Registering with the manager...")
        await ws.send(
            WebsocketMessage(command="register", node_id=self.node_id).json()
        )
        log.info("WEBSOCKET: Connection started.")

    async def send_ping(self, ws):
        pong = await ws.ping()
        await asyncio.wait_for(pong, timeout=self.ping_timeout)
        log.debug("Ping OK, keeping connection alive...")

    async def handle_connection(
        self, websocket: websockets.WebSocketClientProtocol
    ) -> None:
        while True:
            try:
                message: Union[str, bytes] = await websocket.recv()
                if isinstance(message, bytes):
                    log.error(
                        "Received bytes instead of plain text from websocket")
                else:
                    log.debug(f"Received message: {message}")
                    await self.process_message(message)

            except websockets.ConnectionClosed:
                log.warning("Connection to the WebSocket server closed.")
                break

    async def process_message(self, message: str | bytearray | memoryview) -> None:
        if isinstance(message, str):
            msg = parse_message(message)
            print("MSG:", msg)
            if msg is None:
                return
            if msg.node_id != self.node_id:
                log.debug(
                    f"Command not for this node. {msg.node_id=} {self.node_id=}")
                return
            if msg.command == "update_gpus":
                log.info("Updating GPUs")
                await self.execute_update_gpus()

            elif msg.command == "bench_gpus":
                await self.execute_bench_gpus()

            elif msg.command == "run_job":
                if msg.job_id is None:
                    log.error("No job_id in message")
                    return
                if msg.gpu_device_id is None:
                    log.error("No gpu_device_id in message")
                    return
                log.info("Running job")
                self.current_job_runner = JobRunner(
                    self.node_id, self.api, msg.job_id, msg.gpu_device_id, self.gpm
                )
                await self.current_job_runner.run_job()
            else:
                log.error(f"Unknown command: {msg.command}")
        else:
            log.error("Received message is not a string")

    async def execute_bench_gpus(self) -> None:
        if self.gpm:
            for gpu in self.gpm.gpus:
                log.info(f"Running bench gpu for: {gpu.device_id}")
                benchmark_res = gpu.amumax_run()
                log.info(f"Result: {benchmark_res}. Time: {gpu.speed}")

    async def execute_update_gpus(self) -> None:
        if self.gpm and len(self.gpm.gpus) > 0:
            for gpu in self.gpm.gpus:
                log.debug(f"Updating GPU: {gpu.device_id}")
                gpu.update_status()

            self.gpm.api_post("update")

    async def handle_connection_errors(self, ws):
        try:
            await self.send_ping(ws)
        except Exception:
            log.debug(
                f"WEBSOCKET: Lost connection, retrying in {self.sleep_time}s")
            await asyncio.sleep(self.sleep_time)
            return False
        return True

    async def send_current_job_status(self, ws: websockets.WebSocketClientProtocol) -> None:
        if self.current_job_runner and self.current_job_runner.is_running():
            job_status = self.current_job_runner.get_status()

            await ws.send(
                WebsocketMessage(
                    command="job_status",
                    node_id=self.node_id,
                    job_id=self.current_job_runner.job.id,
                    result=job_status,
                ).json()
            )

    # async def monitor_gpus(self):
    #     while True:
    #         await self.fetch_and_check_gpus()
    #         await asyncio.sleep(10)

    # async def fetch_and_check_gpus(self):
    #     try:
    #         print("))))))))", self.current_job_runner.gpu_id)
    #         all_gpus = self.api.fetch_all_gpus()
    #         print("))))))))", self.current_job_runner.gpu_id)
    #         for gpu in all_gpus:
    #             print("))))))))", gpu, self.current_job_runner.gpu_id)
    #             if gpu.status != 'UNAVAILABLE' and not await self.check_gpu_connection(gpu):
    #                 log.warning(f"GPU {gpu.device_id} is disconnected.")
    #                 gpu.status = 'UNAVAILABLE'
    #                 print("))))))))", gpu, self.current_job_runner.gpu_id)
    #                 self.api.update_gpu_status(gpu, self.current_job_runner.gpu_id)
    #                 #await self.update_node_status()
    #     except Exception as e:
    #         log.error(f"Error fetching or checking GPUs: {e}")

    # async def check_gpu_connection(self, gpu: GPU) -> bool:
    #     try:

    #         return gpu.ping()
    #     except Exception as e:
    #         log.error(f"Error checking GPU {gpu.device_id} status: {e}")
    #         return False

    # async def update_node_status(self):
    #     try:
    #         all_gpus = self.api.fetch_all_gpus()
    #         node_status = all(gpu.status == 'connected' for gpu in all_gpus)
    #         if not node_status:
    #             log.warning(f"Node {self.node_id} is disconnected.")
    #     except Exception as e:
    #         log.error(f"Error updating node status: {e}")

    async def close(self):
        log.info("Closing connection...")
        await self.ws.close()
