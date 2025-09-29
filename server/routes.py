import os
import datetime
import logging

import httpx

from fastapi import APIRouter, UploadFile, File, Depends, Form, HTTPException
from qdrant_client import QdrantClient

from .deps import get_qdrant_client, get_embeddings
from https import Response

router = APIRouter(
    prefix="/api"
)

@router.get("/health")
def health_check():
    return {"status": "ok"}


@router.post("/verify-face")
async def verify_face(
    image: UploadFile = File(...),
    qdrant: QdrantClient = Depends(get_qdrant_client)
):
    try:
        embeddings = await get_embeddings(image)
        if len(embeddings) > 1:
            raise HTTPException(status_code=400, detail="Multiple faces detected")

        is_real = all(i['is_real'] for i in embeddings)

        if not is_real:
            raise HTTPException(status_code=400, detail="Face is not real, please try again.")

        results = qdrant.query_points(
            collection_name="faces",
            query=embeddings[0]["embedding"],
            limit=1,
            with_payload=["user_id"],
            score_threshold=0.8,
        )

        if not results.points:
            raise HTTPException(status_code=404, detail="No match found. Please register your face.")

        user_id = results.points[0].payload["user_id"]

        # If the results have a match, send the user_id to ASP .NET API then send a response back to the frontend
        async with httpx.AsyncClient() as client:
            api_url = os.getenv("API_URL")
            auth_key = os.getenv("AUTH_KEY")

            response = await client.post(
                url=api_url + "/api/Auth/login/face",
                headers={"X-Api-Key": auth_key},
                json={"user_id": user_id}
            )

            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=response.text)

        # Contains {user, token, refresh_token}
        return response.json()
    except ValueError as e:
        logging.error(msg=str(e))
        return Response(message="Unable to detect face, ensure the face and camera is clear", success=False)
    except HTTPException as e:
        logging.error(msg=str(e))
        return Response(message=e.detail, success=False)
    except Exception as e:
        logging.error(msg=str(e))
        return Response(message="Something went wrong. Please try again.", success=False)


@router.post("/register_face")
async def register_face(
    image: UploadFile = File(...),
    user_id: str = Form(...),
    name: str = Form(...),
    qdrant: QdrantClient = Depends(get_qdrant_client)
):
    try:
        embeddings = await get_embeddings(image)

        if len(embeddings) > 1:
            raise HTTPException(status_code=400, detail="Multiple faces detected")

        is_real = all(i['is_real'] for i in embeddings)

        if not is_real:
            raise HTTPException(status_code=400, detail="Face is not real")

        # Store the embedding in Qdrant with user_id as the id
        qdrant.upsert(
            collection_name="faces",
            points=[
                {
                    "id": user_id,
                    "vector": embeddings,
                    "payload": {
                        "user_id": user_id,
                        "name": name,
                        "registered_at": datetime.datetime.now().isoformat()
                    }
                }
            ]
        )

        return Response(message="Face registered successfully", success=True)
    except ValueError as e:
        logging.error(msg=str(e))
        return Response(message="Unable to detect face, ensure the face and camera is clear", success=False)
    except HTTPException as e:
        logging.error(msg=str(e))
        return Response(message=e.detail, success=False)
    except Exception as e:
        logging.error(msg=str(e))
        return Response(message="Something went wrong. Please try again.", success=False)
