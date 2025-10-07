import logging
import os
from typing import Any
import uuid
import httpx

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Response
from qdrant_client import AsyncQdrantClient, models

from server.deps import INTERNAL_SERVICE_KEY, get_embeddings, get_qdrant_client, verify_internal_request
from server.dtos import ApiResponseDto, FaceRegisterRequestDto
from server.utils.jwt_helper import create_signed_jwt
from server.utils.rsa_keys import rsa_manager

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


@router.get("/internal/jwks", dependencies=[Depends(verify_internal_request)])
async def get_jwks():
    """
    Internal endpoint for ASP.NET Core to fetch public keys in JWK format.
    This endpoint should be protected and only accessible from your backend service.
    """
    try:
        jwks = rsa_manager.get_public_jwk()
        return jwks
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve keys: {str(e)}")


@router.post("/api/verify-face")
async def verify_face(
    response: Response,
    image: UploadFile = File(...),
    qdrant: AsyncQdrantClient = Depends(get_qdrant_client),
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
            score_threshold=0.85,
        )

        if not qdrantResult.points or len(qdrantResult.points) == 0 or qdrantResult.points[0].payload is None:
            raise HTTPException(status_code=404, detail="No match found. Please register your face.")

        user_id = qdrantResult.points[0].payload["user_id"]
        jwt_token = create_signed_jwt(payload={"user_id": user_id})

        return ApiResponseDto(
            message="Face verified successfully",
            success=True,
            statusCode=200,
            data={
                "jwt_token": jwt_token,
                "user_id": user_id
            }
        )
    except ValueError as e:
        logging.error(msg=str(e))
        return ApiResponseDto(
            message="Unable to detect face, ensure the face and camera is clear",
            success=False,
            statusCode=400
		)
    except HTTPException as e:
        logging.error(msg=str(e))
        return ApiResponseDto(message=e.detail, success=False, statusCode=e.status_code)
    except Exception as e:
        logging.error(msg=str(e))
        return ApiResponseDto(message="Something went wrong. Please try again.", success=False, statusCode=500)


@router.post("/api/register-face")
async def register_face(
    request: FaceRegisterRequestDto,
    image: UploadFile = File(...),
    qdrant: AsyncQdrantClient = Depends(get_qdrant_client)
):
    try:
        embeddings = await get_embeddings(image)

        if len(embeddings) == 0:
            logging.warning("No face detected in image")
            raise HTTPException(
                status_code=400,
                detail="No face detected. Please ensure your face is clearly visible."
            )

        if len(embeddings) > 1:
            raise HTTPException(
                status_code=400,
                detail="Multiple faces detected. Please ensure only one face is in the image."
            )

        is_real = all(i['is_real'] for i in embeddings)

        if not is_real:
            raise HTTPException(
                status_code=400,
                detail="Liveness check failed. Please use a real face, not a photo or video."
            )


        existing_face = await qdrant.search(
			collection_name="faces",
			query_vector=embeddings[0]['embedding'],
			limit=1,
			with_payload=["email"],
			score_threshold=0.85
		)

        if existing_face and len(existing_face) > 0 and existing_face[0].payload and existing_face[0].payload["email"] == request.email:
            raise HTTPException(
                status_code=400,
                detail="Face already registered. Please login instead or use a different email."
            )

        payload = {
            "firstName": request.firstName,
            "lastName": request.lastName,
            "email": request.email,
            "position": request.position,
            "departmentId": request.departmentId,
            "storeId": request.storeId,
            "timeLogged": request.timeLogged.isoformat()
        }
        signed_jwt = create_signed_jwt(payload=payload)

        try:
            async with httpx.AsyncClient() as client:
                api_url = os.getenv("API_URL")

                auth_response = await client.post(
                    url=f"{api_url}/api/Auth/face-register",
                    headers={
                        "Authorization": f"Bearer {signed_jwt}",
                        "X-Internal-Key": INTERNAL_SERVICE_KEY,
                        "Content-Type": "application/json"
                    },
                    json=payload
                )

                if auth_response.status_code not in [200, 201]:
                    raise HTTPException(status_code=auth_response.status_code, detail=auth_response.text)
                else:
                    # Same data type as AuthResponseDto from core service
                    data: dict[str, Any] = auth_response.json()

        except httpx.TimeoutException as e:
            logging.error(f"Failed to register face: {e}")
            raise HTTPException(status_code=504, detail="Registration service is unavailable. Please try again later.")
        except httpx.RequestError as e:
            logging.error(f"Failed to connect to auth service: {e}")
            raise HTTPException(status_code=500, detail="Failed to connect to auth service")

		# Store the embedding in Qdrant with user_id as the id after successful registration
        await qdrant.upsert(
            collection_name="faces",
            points=[
				models.PointStruct(
					id=str(uuid.uuid5(uuid.NAMESPACE_DNS, data['user']['id'])),
					vector=embeddings[0]['embedding'],
					payload={
						"user_id": data['user']['id'],
						"email": data['user']['email'],
						"registered_at": request.timeLogged.isoformat()
					}
				)
			]
        )

        return ApiResponseDto(
			message="Face registered successfully",
			success=True,
			statusCode=200,
			data=data
		)
    except ValueError as e:
        logging.error(msg=str(e))
        return ApiResponseDto(
            message="Unable to detect face, ensure the face and camera is clear",
            success=False,
            statusCode=400
		)
    except HTTPException as e:
        logging.error(msg=str(e))
        return ApiResponseDto(message=e.detail, success=False, statusCode=e.status_code)
    except Exception as e:
        logging.error(msg=str(e))
        return ApiResponseDto(message="Something went wrong. Please try again.", success=False, statusCode=500)
