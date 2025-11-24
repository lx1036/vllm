import argparse
import json
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

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

