import logging
import cv2
import numpy
from io import BytesIO
from typing import List, Dict, Any

from fastapi import UploadFile, File, Request, HTTPException
from PIL import Image
from qdrant_client import AsyncQdrantClient

import face_recognition


def get_qdrant_client(request: Request) -> AsyncQdrantClient:
    # Return the qdrant client stored on the FastAPI app instance
    return request.app.state.qdrant_client


async def get_embeddings(image: UploadFile = File(...)) -> List[Dict[str, Any]]:
    if not image:
        raise HTTPException(status_code=400, detail="No image provided")

    try:
        image_data = await image.read()
        pil_image = Image.open(BytesIO(image_data))

        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')

        image_np = numpy.array(pil_image)
        image_enhanced = cv2.detailEnhance(image_np, sigma_s=4, sigma_r=0.09)

        embeddings = face_recognition.embedding(
            image_enhanced,
            expand_percentage=3,
            model_name="Facenet512",
            align=True,
            normalization="base",
            anti_spoofing=True
        )

        return embeddings
    except ValueError as e:
        logging.error(msg=str(e))
        raise e
    except Exception as e:
        logging.error(msg=f"Error in processing image: {e}")
        raise e
