import asyncio
import uvicorn

from server.app import app


async def start_server():
    config = uvicorn.Config(
        app=app,
        host='0.0.0.0',
        port=8000,
        log_level="info"
    )
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == '__main__':
    try:
        asyncio.run(start_server())
    except KeyboardInterrupt:
        print("Server stopped.")
