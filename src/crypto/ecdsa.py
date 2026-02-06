"""
ECDSA Signature Module

Handles signing and verification operations using ECDSA.
"""

import hashlib
import base64
from ecdsa import SigningKey


class SignatureManager:
    """
    Manages ECDSA signing operations for federated learning clients.

    Provides methods to sign model updates and related data.
    """

    def __init__(self, private_key: SigningKey):
        """
        Initialize the signature manager.

        Args:
            private_key: ECDSA private signing key
        """
        self.private_key = private_key

    def sign(self, message: bytes) -> str:
        """
        Sign a message using ECDSA.

        Args:
            message: Message bytes to sign

        Returns:
            Base64-encoded signature string
        """
        signature = self.private_key.sign(
            message,
            hashfunc=hashlib.sha256,
        )
        return base64.b64encode(signature).decode()

    def sign_hash(self, message_hash: bytes) -> str:
        """
        Sign a pre-computed hash.

        Args:
            message_hash: SHA-256 hash to sign

        Returns:
            Base64-encoded signature string
        """
        signature = self.private_key.sign_digest(message_hash)
        return base64.b64encode(signature).decode()

    def sign_dict(self, data: dict) -> tuple[str, str]:
        """
        Sign a dictionary after canonical serialization.

        Args:
            data: Dictionary to sign

        Returns:
            Tuple of (signature, message_hash)
        """
        import json

        # Serialize deterministically
        serialized = json.dumps(data, sort_keys=True).encode()
        message_hash = hashlib.sha256(serialized).digest()

        signature = self.sign_hash(message_hash)

        return signature, message_hash.hex()
