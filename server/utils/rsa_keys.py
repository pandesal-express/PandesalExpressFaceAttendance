# rsa_keys.py
import asyncio
import base64
import datetime
import secrets
from typing import Optional, Dict
from threading import Lock
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization


class RSAKeyManager:
    def __init__(self, rotation_interval_minutes: int = 60, is_prod: bool = False):
        self.rotation_interval = datetime.timedelta(minutes=rotation_interval_minutes)
        self.is_prod = is_prod
        self._lock = Lock()  # Thread safety

        # Key storage
        self._current_private_key = None
        self._current_public_key = None
        self._current_kid = None

        self._previous_public_key = None
        self._previous_kid = None

        self._last_rotation = None

        # Generate initial keys
        self.generate_keys()

    async def start_rotation(self):
        """Background task for continuous key rotation."""
        while True:
            await asyncio.sleep(self.rotation_interval.total_seconds())
            self.generate_keys()

    def generate_keys(self):
        """Generate or load RSA keypair based on environment."""
        with self._lock:
            if not self.is_prod:
                print("[RSA] Generating new local development RSA keypair...")
                private_key = rsa.generate_private_key(
                    public_exponent=65537,
                    key_size=2048
                )
            else:
                # TODO: Load from secure key management service (AWS KMS, Azure Key Vault, etc.)
                print("[RSA] Loading RSA keys from secure storage...")
                raise NotImplementedError("Production RSA key loading not yet implemented.")

            # Store previous key for grace period
            if self._current_public_key:
                self._previous_public_key = self._current_public_key
                self._previous_kid = self._current_kid

            # Update current keys
            self._current_private_key = private_key
            self._current_public_key = private_key.public_key()
            self._current_kid = self._generate_kid()
            self._last_rotation = datetime.datetime.now(datetime.timezone.utc)

            print(f"[RSA] Key rotated at {self._last_rotation.isoformat()}, kid: {self._current_kid}")

    def _generate_kid(self) -> str:
        """Generate a unique key identifier."""
        timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d%H%M%S")
        random_suffix = secrets.token_hex(4)
        return f"face-service-{timestamp}-{random_suffix}"

    def get_private_key(self):
        """Get the current private key object."""
        with self._lock:
            if not self._current_private_key:
                raise RuntimeError("RSA private key not initialized.")
            return self._current_private_key

    def get_private_pem(self) -> bytes:
        """Get current private key in PEM format."""
        if not self._current_private_key:
            raise RuntimeError("RSA private key not initialized.")
        return self._current_private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )

    def get_public_pem(self) -> bytes:
        """Get current public key in PEM format."""
        with self._lock:
            if not self._current_public_key:
                raise RuntimeError("RSA public key not initialized.")
            return self._current_public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )

    def get_previous_public_pem(self) -> Optional[bytes]:
        """Get previous public key in PEM format (for grace period)."""
        with self._lock:
            if not self._previous_public_key:
                return None
            return self._previous_public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )

    def get_current_kid(self) -> str:
        """Get current key ID."""
        with self._lock:
            if not self._current_kid:
                raise RuntimeError("Key ID not initialized.")
            return self._current_kid

    def get_previous_kid(self) -> Optional[str]:
        """Get previous key ID."""
        with self._lock:
            return self._previous_kid

    def get_public_jwk(self) -> Dict:
        """
        Get public keys in JWK format.
        Returns current and optionally previous key.
        """
        with self._lock:
            if not self._current_public_key:
                raise RuntimeError("RSA public key not initialized.")

            if not self._current_kid:
                raise RuntimeError("Key ID not initialized.")

            jwks = {"keys": [self._key_to_jwk(self._current_public_key, self._current_kid)]}

            if self._previous_public_key and self._previous_kid:
                jwks["keys"].append(
                    self._key_to_jwk(self._previous_public_key, self._previous_kid)
                )

            return jwks

    def _key_to_jwk(self, public_key, kid: str) -> dict:
        """Convert RSA public key to JWK format."""
        numbers = public_key.public_numbers()

        # Convert to bytes with proper length
        e_bytes = numbers.e.to_bytes((numbers.e.bit_length() + 7) // 8, "big")
        n_bytes = numbers.n.to_bytes((numbers.n.bit_length() + 7) // 8, "big")

        # Base64url encode
        e = base64.urlsafe_b64encode(e_bytes).decode().rstrip("=")
        n = base64.urlsafe_b64encode(n_bytes).decode().rstrip("=")

        return {
            "kty": "RSA",
            "alg": "RS256",
            "use": "sig",
            "kid": kid,
            "n": n,
            "e": e,
        }

    def get_rotation_info(self) -> Dict:
        """Get information about key rotation status."""
        with self._lock:
            next_rotation = None
            if self._last_rotation:
                next_rotation = self._last_rotation + self.rotation_interval

            return {
                "last_rotation": self._last_rotation.isoformat() if self._last_rotation else None,
                "next_rotation": next_rotation.isoformat() if next_rotation else None,
                "current_kid": self._current_kid,
                "previous_kid": self._previous_kid,
                "rotation_interval_minutes": self.rotation_interval.total_seconds() / 60
            }


# Initialize the manager
rsa_manager = RSAKeyManager(rotation_interval_minutes=15, is_prod=False)