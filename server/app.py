import os
import logging

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from contextlib import asynccontextmanager
from qdrant_client import AsyncQdrantClient


load_dotenv()
logging.basicConfig(
    filename=os.getcwd() + "/logs/app.log",
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',
    format='%(asctime)s %(levelname)s %(message)s',
)


@asynccontextmanager
async def lifespan(fast_api: FastAPI):
    try:
        logging.info("Connecting to Qdrant Cloud...\n")

        fast_api.qdrant_client = AsyncQdrantClient(
            url=os.getenv("QDRANT_ENDPOINT"),
            api_key=os.getenv("QDRANT_API"),
        )

        logging.info("Connected to Qdrant Cloud!\n")
    except Exception as error:
        logging.error(f"Error connecting to Qdrant Cloud: {error}")
        raise error

    try:
        logging.info("Loading and initializing the model...\n")
        image_path = os.getcwd() + "/images/sample-face.jpg"

        # check if the image exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")

        import face_recognition

        face_recognition.embedding(
            image_path,
            expand_percentage=3,
            model_name="Facenet512",
            align=True,
            normalization="base",
            anti_spoofing=True
        )

        logging.info("Model is loaded and initialized successfully!\n")
    except Exception as e:
        logging.error(f"Error initializing the model: {e}")
        raise e

    try:
        logging.info("Registering routes...")

        from server import routes as routes_module
        fast_api.include_router(routes_module.router)

        logging.info("Routes included successfully.")
    except Exception as e:
        logging.error(f"Failed to include routes during lifespan startup: {e}")
        raise e

    try:
        yield
    finally:
        client = fast_api.qdrant_client
        if client:
            try:
                await client.close()
            except Exception as e:
                logging.error(f"Error closing Qdrant client: {e}")
                raise e


app = FastAPI(lifespan=lifespan)
allowed_origins = os.getenv("ALLOWED_ORIGINS", '').split(",") \
    if os.getenv("APP_ENV") == "production" \
    else os.getenv("ALLOWED_ORIGINS_DEV", '').split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)
