import argparse
import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from typing import List, Dict, Tuple, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from leetcuda.lmcache.cache_controller.controller_manager import LMCacheControllerManager
from leetcuda.lmcache.cache_controller.message import (
    QueryInstMsg,
    QueryInstRetMsg,
    ErrorMsg,
    LookupMsg,
    LookupRetMsg,
    ClearMsg,
    ClearRetMsg,
    PinMsg,
    PinRetMsg,
    CompressMsg,
    CompressRetMsg,
    DecompressMsg,
    DecompressRetMsg,
    MoveMsg,
    MoveRetMsg,
    HealthMsg,
    HealthRetMsg,
    CheckFinishMsg,
    CheckFinishRetMsg
)

from leetcuda.lmcache.log import init_logger

logger = init_logger(__name__)

def create_app(controller_urls: dict[str, str]) -> FastAPI:
    """
    Create a FastAPI application with endpoints for LMCache operations.
    """
    lmcache_controller_manager = LMCacheControllerManager(controller_urls)

    # 这段代码的主要作用是在 FastAPI 应用启动时启动一个后台任务，并在应用关闭时取消该任务。
    # 这样可以确保后台任务在应用的生命周期内正确运行，并在应用关闭时进行适当的清理。
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Start background task here, goroutine
        lmcache_cluster_monitor_task = asyncio.create_task(lmcache_controller_manager.start_all())
        # yield 是异步上下文管理器的关键部分。当 yield 被执行时，控制权返回到调用者，应用开始正常运行。
        # 在应用运行期间，lmcache_cluster_monitor_task 会在后台继续执行。
        yield
        # 当应用关闭时，yield 之后的代码会被执行。首先调用 lmcache_cluster_monitor_task.cancel() 来取消后台任务。
        # cancel 方法会尝试取消任务，但不会立即停止任务的执行，而是将任务的状态设置为取消状态。
        lmcache_cluster_monitor_task.cancel()
        try:
            # 使用 await 等待任务完成。如果任务已经被取消，await 会引发 asyncio.CancelledError 异常。
            await lmcache_cluster_monitor_task
        except asyncio.CancelledError:
            pass

    app = FastAPI(lifespan=lifespan)

    class QueryInstRequest(BaseModel):
        event_id: str
        ip: str

    class QueryInstResponse(BaseModel):
        event_id: str
        res: str  # the instance id
    @app.post("/query_instance")
    async def query_instance(req: QueryInstRequest):
        try:
            event_id = "QueryInst" + str(uuid.uuid4())
            msg = QueryInstMsg(
                event_id=event_id,
                ip=req.ip,
            )
            ret_msg = await lmcache_controller_manager.handle_orchestration_message(msg)
            assert not isinstance(ret_msg, ErrorMsg), ret_msg.error
            assert isinstance(ret_msg, QueryInstRetMsg)
            return QueryInstResponse(
                event_id=ret_msg.event_id,
                res=ret_msg.instance_id,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    class LookupRequest(BaseModel):
        tokens: List[int]

    class LookupResponse(BaseModel):
        event_id: str
        # a list of (instance_id, location, token_count)
        layout_info: Dict[str, Tuple[str, int]]
    @app.post("/lookup", response_model=LookupResponse)
    async def lookup(req: LookupRequest):
        try:
            event_id = "Lookup" + str(uuid.uuid4())
            msg = LookupMsg(
                event_id=event_id,
                tokens=req.tokens,
            )
            ret_msg = await lmcache_controller_manager.handle_orchestration_message(msg)
            assert not isinstance(ret_msg, ErrorMsg), ret_msg.error
            assert isinstance(ret_msg, LookupRetMsg)
            return LookupResponse(
                event_id=ret_msg.event_id,
                layout_info=ret_msg.layout_info,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    class ClearRequest(BaseModel):
        instance_id: str
        location: str

    class ClearResponse(BaseModel):
        event_id: str
        num_tokens: int
    @app.post("/clear", response_model=ClearResponse)
    async def clear(req: ClearRequest):
        try:
            event_id = "Clear" + str(uuid.uuid4())
            msg = ClearMsg(
                event_id=event_id,
                instance_id=req.instance_id,
                location=req.location,
            )
            ret_msg = await lmcache_controller_manager.handle_orchestration_message(msg)
            assert not isinstance(ret_msg, ErrorMsg), ret_msg.error
            assert isinstance(ret_msg, ClearRetMsg)
            return ClearResponse(
                event_id=ret_msg.event_id,
                num_tokens=ret_msg.num_tokens,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e


    class PinRequest(BaseModel):
        instance_id: str
        location: str
        tokens: list[int]

    class PinResponse(BaseModel):
        event_id: str
        num_tokens: int
    @app.post("/pin", response_model=PinResponse)
    async def pin(req: PinRequest):
        try:
            event_id = "Pin" + str(uuid.uuid4())
            msg = PinMsg(
                event_id=event_id,
                instance_id=req.instance_id,
                location=req.location,
                tokens=req.tokens,
            )
            ret_msg = await lmcache_controller_manager.handle_orchestration_message(msg)
            assert not isinstance(ret_msg, ErrorMsg), ret_msg.error
            assert isinstance(ret_msg, PinRetMsg)
            return PinResponse(event_id=ret_msg.event_id, num_tokens=ret_msg.num_tokens)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    class CompressRequest(BaseModel):
        instance_id: str
        method: str
        location: str
        tokens: Optional[List[int]] = []

    class CompressResponse(BaseModel):
        event_id: str
        num_tokens: int
    @app.post("/compress", response_model=CompressResponse)
    async def compress(req: CompressRequest):
        try:
            event_id = "Compress" + str(uuid.uuid4())
            msg = CompressMsg(
                event_id=event_id,
                instance_id=req.instance_id,
                method=req.method,
                location=req.location,
                tokens=req.tokens,
            )
            ret_msg = await lmcache_controller_manager.handle_orchestration_message(msg)
            assert not isinstance(ret_msg, ErrorMsg), ret_msg.error
            assert isinstance(ret_msg, CompressRetMsg)
            return CompressResponse(
                event_id=ret_msg.event_id,
                num_tokens=ret_msg.num_tokens,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    class DecompressRequest(BaseModel):
        instance_id: str
        method: str
        location: str
        tokens: Optional[List[int]] = []

    class DecompressResponse(BaseModel):
        event_id: str
        num_tokens: int
    @app.post("/decompress", response_model=DecompressResponse)
    async def decompress(req: DecompressRequest):
        try:
            event_id = "Decompress" + str(uuid.uuid4())
            msg = DecompressMsg(
                event_id=event_id,
                instance_id=req.instance_id,
                method=req.method,
                location=req.location,
                tokens=req.tokens,
            )
            ret_msg = await lmcache_controller_manager.handle_orchestration_message(msg)
            assert isinstance(ret_msg, DecompressRetMsg)
            return DecompressResponse(
                event_id=ret_msg.event_id,
                num_tokens=ret_msg.num_tokens,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    class MoveRequest(BaseModel):
        # (instance_id, location)
        old_position: Tuple[str, str]
        new_position: Tuple[str, str]
        tokens: Optional[List[int]] = []
        # copy: Optional[bool] = False

    class MoveResponse(BaseModel):
        event_id: str
        num_tokens: int
    @app.post("/move", response_model=MoveResponse)
    async def move(req: MoveRequest):
        try:
            event_id = "Move" + str(uuid.uuid4())
            msg = MoveMsg(
                event_id=event_id,
                old_position=req.old_position,
                new_position=req.new_position,
                tokens=req.tokens,
                copy=req.copy,
            )
            ret_msg = await lmcache_controller_manager.handle_orchestration_message(msg)
            assert not isinstance(ret_msg, ErrorMsg), ret_msg.error
            assert isinstance(ret_msg, MoveRetMsg)
            return MoveResponse(
                event_id=ret_msg.event_id,
                num_tokens=ret_msg.num_tokens,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    class HealthRequest(BaseModel):
        instance_id: str

    class HealthResponse(BaseModel):
        event_id: str
        # worker_id -> error_code
        error_codes: dict[int, int]
    @app.post("/health", response_model=HealthResponse)
    async def health(req: HealthRequest):
        try:
            event_id = "health" + str(uuid.uuid4())
            msg = HealthMsg(
                event_id=event_id,
                instance_id=req.instance_id,
            )
            ret_msg = await lmcache_controller_manager.handle_orchestration_message(msg)
            assert not isinstance(ret_msg, ErrorMsg), ret_msg.error
            assert isinstance(ret_msg, HealthRetMsg)
            return HealthResponse(
                event_id=ret_msg.event_id,
                error_codes=ret_msg.error_codes,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    class CheckFinishRequest(BaseModel):
        event_id: str

    class CheckFinishResponse(BaseModel):
        status: str
    @app.post("/check_finish", response_model=CheckFinishResponse)
    async def check_finish(req: CheckFinishRequest):
        try:
            msg = CheckFinishMsg(
                event_id=req.event_id,
            )
            ret_msg = await lmcache_controller_manager.handle_orchestration_message(msg)
            assert not isinstance(ret_msg, ErrorMsg), ret_msg.error
            assert isinstance(ret_msg, CheckFinishRetMsg)
            return CheckFinishResponse(status=ret_msg.status)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    return app

# python3 . --port=9000 --monitor-ports='{"pull": 8300, "reply": 8400}'
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument(
        "--monitor-ports",
        type=json.loads,
        default=None,
        help='JSON string of monitor ports, e.g. \'{"pull": 8300, "reply": 8400}\'',
    )
    parser.add_argument(
        "--monitor-pull-port",
        type=int,
        default=8300,
        help="The controller pull port to maintain backward compatibility.",
    )
    parser.add_argument(
        "--monitor-reply-port",
        type=int,
        default=8400,
        help="The controller pull port to maintain backward compatibility.",
    )

    args = parser.parse_args()

    try:
        if args.monitor_ports is not None:
            controller_urls = {
                "pull": f"{args.host}:{args.monitor_ports['pull']}",
                "reply": f"{args.host}:{args.monitor_ports['reply']}",
            }
        else:
            logger.warning("Argument --monitor-port will be deprecated soon. Please use --monitor-ports instead.")
            controller_urls = {
                "pull": f"{args.host}:{args.monitor_pull_port}",
                "reply": f"{args.host}:{args.monitor_reply_port}",
            }

        logger.info(f"Starting LMCache controller at {args.host}:{args.port}")
        logger.info(f"controller_urls: {controller_urls}")

        app = create_app(controller_urls)

        uvicorn.run(app, host=args.host, port=args.port)
    except TimeoutError as e:
        logger.error(e)

# python3
if __name__ == "__main__":
    main()
