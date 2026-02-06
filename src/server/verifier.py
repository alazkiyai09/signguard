"""
Signature Verifier Module

Handles verification of ECDSA signatures for federated learning updates.
"""

import hashlib
import json
from typing import Optional
from ecdsa import VerifyingKey, BadSignatureError


class SignatureVerifier:
    """
    Verifies ECDSA signatures for federated learning updates.

    This class provides methods to verify that model updates
    originate from authenticated clients.
    """

    def __init__(self):
        """Initialize the signature verifier."""
        pass

    def verify(
        self,
        signature: str,
        message_hash: bytes,
        public_key_pem: str,
    ) -> bool:
        """
        Verify an ECDSA signature.

        Args:
            signature: Base64-encoded signature string
            message_hash: SHA-256 hash of the message
            public_key_pem: PEM-encoded public key

        Returns:
            True if signature is valid, False otherwise
        """
        try:
            # Load public key
            public_key = VerifyingKey.from_pem(public_key_pem.encode())

            # Decode signature from base64
            import base64
            signature_bytes = base64.b64decode(signature)

            # Verify signature
            public_key.verify(signature_bytes, message_hash, hashfunc=hashlib.sha256)
            return True

        except BadSignatureError:
            return False
        except Exception as e:
            print(f"Verification error: {e}")
            return False

    def hash_update(self, update: dict) -> bytes:
        """
        Compute SHA-256 hash of model update.

        Args:
            update: Dictionary containing model updates

        Returns:
            SHA-256 hash as bytes
        """
        # Serialize update deterministically
        serialized = json.dumps(update, sort_keys=True).encode()
        return hashlib.sha256(serialized).digest()
