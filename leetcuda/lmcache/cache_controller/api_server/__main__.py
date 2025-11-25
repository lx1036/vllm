import argparse
import json
import uuid
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from leetcuda.lmcache.cache_controller.message import QueryInstMsg, QueryInstRetMsg, ErrorMsg
from leetcuda.lmcache.log import init_logger

logger = init_logger(__name__)


def create_app(controller_urls: dict[str, str]) -> FastAPI:
    """
    Create a FastAPI application with endpoints for LMCache operations.
    """
    lmcache_controller_manager = LMCacheControllerManager(controller_urls)

    @asynccontextmanager
    async def lifespan(app: FastAPI):


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




    @app.post("/lookup", response_model=LookupResponse)
    async def lookup(req: LookupRequest):


    @app.post("/clear", response_model=ClearResponse)
    async def clear(req: ClearRequest):



    @app.post("/pin", response_model=PinResponse)
    async def pin(req: PinRequest):


    @app.post("/compress", response_model=CompressResponse)
    async def compress(req: CompressRequest):


    @app.post("/decompress", response_model=DecompressResponse)
    async def decompress(req: DecompressRequest):


    @app.post("/move", response_model=MoveResponse)
    async def move(req: MoveRequest):


    @app.post("/health", response_model=HealthResponse)
    async def health(req: HealthRequest):



    @app.post("/check_finish", response_model=CheckFinishResponse)
    async def check_finish(req: CheckFinishRequest):






    return app

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
        "--monitor-port",
        type=int,
        default=9001,
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
            logger.warning(
                "Argument --monitor-port will be deprecated soon. "
                "Please use --monitor-ports instead."
            )
            controller_urls = {
                "pull": f"{args.host}:{args.monitor_port}",
                "reply": None,
            }

        app = create_app(controller_urls)

        logger.info(f"Starting LMCache controller at {args.host}:{args.port}")
        logger.info(f"Monitoring lmcache workers at ports {args.monitor_ports}")

        uvicorn.run(app, host=args.host, port=args.port)
    except TimeoutError as e:
        logger.error(e)


if __name__ == "__main__":
    main()

