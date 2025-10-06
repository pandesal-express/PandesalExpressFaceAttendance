import jwt
import datetime

from typing import Dict, Optional
from jwt import InvalidAudienceError, InvalidIssuerError, InvalidSignatureError, ExpiredSignatureError, InvalidTokenError
from .rsa_keys import rsa_manager

def create_signed_jwt(payload: Dict, expires_in_minutes: int = 5) -> str:
    """
    Create a short-lived JWT signed by the current RSA private key.
    This JWT is used for secure communication with Main Backend.
    """
    private_pem = rsa_manager.get_private_pem()

    now = datetime.datetime.now(datetime.timezone.utc)
    payload = {
        **payload,
        "iat": now,
        "exp": now + datetime.timedelta(minutes=expires_in_minutes),
        "iss": "face-service",
		"aud": "core-service"
    }
    return jwt.encode(
        payload,
        private_pem,
        algorithm="RS256",
        headers={"kid": rsa_manager.get_current_kid()}
	)


def verify_signed_jwt(token: str) -> Optional[Dict]:
    """
    Verify a JWT using the current or previous RSA public key.
    Tries current key first, then falls back to previous key.

    Args:
        token: JWT token string to verify

    Returns:
        Decoded payload if valid, None if invalid
    """
    # Try to decode header to get kid
    try:
        unverified_header = jwt.get_unverified_header(token)
        token_kid = unverified_header.get("kid")
    except InvalidTokenError:
        return None

    # Build list of keys to try (current first, then previous)
    keys_to_try = [(rsa_manager.get_public_pem(), rsa_manager.get_current_kid())]

    if (prev_pem := rsa_manager.get_previous_public_pem()) and (prev_kid := rsa_manager.get_previous_kid()):
        keys_to_try.append((prev_pem, prev_kid))

    # Try to verify with matching kid first
    if token_kid:
        keys_to_try.sort(key=lambda x: x[1] != token_kid)

    for public_pem, key_kid in keys_to_try:
        try:
            decoded = jwt.decode(
                token,
                public_pem,
                algorithms=["RS256"],
                audience="core-service",
                issuer="face-service",
                options={
                    "verify_signature": True,
                    "verify_exp": True,
                    "verify_iat": True,
                    "verify_aud": True,
                    "verify_iss": True,
                }
            )
            return decoded

        except ExpiredSignatureError:
            # Token expired, don't try other keys
            print("[JWT] Token expired")
            return None

        except (InvalidSignatureError, InvalidAudienceError, InvalidIssuerError):
            # Try next key
            continue

        except InvalidTokenError as e:
            # Malformed token, don't try other keys
            print(f"[JWT] Invalid token: {str(e)}")
            return None

    print("[JWT] Token verification failed with all available keys")
    return None
