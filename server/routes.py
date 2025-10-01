import logging
import os
import datetime
from typing import Any
import uuid

import httpx

from fastapi import APIRouter, UploadFile, File, Depends, Form, HTTPException, Response
from qdrant_client import AsyncQdrantClient, models

from .dtos import AuthResponseDto, FaceRegisterRequest
from .deps import get_qdrant_client, get_embeddings

router = APIRouter()
logging.basicConfig(
    filename=os.getcwd() + "/logs/app.log",
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',
    format='%(asctime)s %(levelname)s %(message)s',
)

@router.get("/health")
def health_check():
    return {"status": "ok"}


@router.post("/api/verify-face")
async def verify_face(
    response: Response,
    image: UploadFile = File(...),
    qdrant: AsyncQdrantClient = Depends(get_qdrant_client)
):
    try:
        embeddings = await get_embeddings(image)
        if len(embeddings) > 1:
            raise HTTPException(status_code=400, detail="Multiple faces detected")

        is_real = all(i['is_real'] for i in embeddings)

        if not is_real:
            raise HTTPException(status_code=400, detail="Face is not real, please try again.")

        qdrantResult = await qdrant.query_points(
            collection_name="faces",
            query=embeddings[0]["embedding"],
            limit=1,
            with_payload=["user_id"],
            score_threshold=0.8,
        )

        if not qdrantResult.points or len(qdrantResult.points) == 0 or qdrantResult.points[0].payload is None:
            raise HTTPException(status_code=404, detail="No match found. Please register your face.")

        user_id = qdrantResult.points[0].payload['user_id']

        # If the results have a match, send the user_id to ASP .NET API then send a response back to the frontend
        async with httpx.AsyncClient() as client:
            api_url = os.getenv("API_URL")
            auth_key = os.getenv("AUTH_KEY") # TODO: For dev and testing purposes only, will replace when deploying

            if auth_key is None:
                raise HTTPException(status_code=500, detail="Auth key not found")

            auth_response = await client.post(
                url=f"{api_url}/api/Auth/face-login",
                headers={
                    "X-Api-Key": auth_key,
                    "Content-Type": "application/json"
                },
                json={"user_id": user_id}
            )

            if auth_response.status_code not in [200, 201]:
                raise HTTPException(status_code=auth_response.status_code, detail=auth_response.text)
            else:
                # Same data type as AuthResponseDto
                data: dict[str, Any] = auth_response.json()

        if data['token']:
            response.set_cookie(
                key="jwt_token",
                value=data['token'],
                expires=datetime.datetime.fromisoformat(data['expiration']),
                httponly=True,
                secure=True,
                samesite="lax",
                path="/"
            )

        if data['refreshToken']:
            response.set_cookie(
                key="refresh_token",
                value=data['refreshToken'],
                expires=datetime.datetime.fromisoformat(data['refreshTokenExpiration']),
                httponly=True,
                secure=True,
                samesite="strict",
                path="/api/Auth/refresh-token"
            )

        # Contains {user, token, refresh_token}
        return AuthResponseDto(**data)
    except ValueError as e:
        logging.error(msg=str(e))
        return Response(
			status_code=400,
			content={
				"message": "Unable to detect face, ensure the face and camera is clear",
				"success": False
			}
		)
    except HTTPException as e:
        logging.error(msg=str(e))
        return Response(
            status_code=e.status_code,
            content={"message": e.detail}
		)
    except Exception as e:
        logging.error(msg=str(e))
        return Response(status_code=500, content={"message": "Something went wrong. Please try again."})


@router.post("/api/register-face")
async def register_face(
    request: FaceRegisterRequest,
    image: UploadFile = File(...),
    qdrant: AsyncQdrantClient = Depends(get_qdrant_client),
    response: Response = Response()
):
    try:
        embeddings = await get_embeddings(image)

        if len(embeddings) > 1:
            raise HTTPException(status_code=400, detail="Multiple faces detected")

        is_real = all(i['is_real'] for i in embeddings)

        if not is_real:
            raise HTTPException(status_code=400, detail="Face is not real")

        async with httpx.AsyncClient() as client:
            api_url = os.getenv("API_URL")
            auth_key = os.getenv("AUTH_KEY")

            if auth_key is None:
                raise HTTPException(status_code=500, detail="Auth key not found")

            auth_response = await client.post(
                url=f"{api_url}/api/Auth/face-register",
                headers={
                    "X-Api-Key": auth_key,
                    "Content-Type": "application/json"
				},
                json={
                    "firstName": request.firstName,
                    "lastname": request.lastName,
                    "email": request.email,
                    "position": request.position,
                    "departmentId": request.departmentId,
                    "storeId": request.storeId
				}
            )

            if auth_response.status_code not in [200, 201]:
                raise HTTPException(status_code=auth_response.status_code, detail=auth_response.text)
            else:
                # Same data type as AuthResponseDto
                data: dict[str, Any] = auth_response.json()


		# Store the embedding in Qdrant with user_id as the id after successful registration
        await qdrant.upsert(
            collection_name="faces",
            points=[
				models.PointStruct(
					id=str(uuid.uuid5(uuid.NAMESPACE_DNS, data['user']['id'])),
					vector=embeddings[0]['embedding'],
					payload={
						"user_id": data['user']['id'],
						"name": data['user']['firstName'] + " " + data['user']['lastName'],
						"register_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
					}
				)
			]
        )

		# Set the cookies if the tokens are present
        if data['token']:
            response.set_cookie(
				key="jwt_token",
				value=data['token'],
				expires=datetime.datetime.fromisoformat(data['expiration']),
				httponly=True,
				secure=True,
				samesite="lax",
				path="/"
			)

        if data['refreshToken']:
            response.set_cookie(
                key="refresh_token",
                value=data['refreshToken'],
                expires=datetime.datetime.fromisoformat(data['refreshTokenExpiration']),
                httponly=True,
                secure=True,
                samesite="strict",
                path="/api/Auth/refresh-token"
            )

        return AuthResponseDto(**data)
    except ValueError as e:
        logging.error(msg=str(e))
        return Response(
			status_code=400,
			content={
				"message": "Unable to detect face, ensure the face and camera is clear",
			}
		)
    except HTTPException as e:
        logging.error(msg=str(e))
        return Response(
			status_code=e.status_code,
			content={"message": e.detail}
		)
    except Exception as e:
        logging.error(msg=str(e))
        return Response(
			status_code=500,
			content={"message": "Something went wrong. Please try again."}
		)
