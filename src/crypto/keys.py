"""
Key Management Module

Handles generation and storage of cryptographic keys for ECDSA signatures.
"""

from ecdsa import SigningKey, VerifyingKey, NIST256p
import base64


class KeyManager:
    """
    Manages ECDSA key pairs for federated learning clients.

    Uses the NIST256p curve for cryptographic operations.
    """

    def __init__(self, curve: type = NIST256p):
        """
        Initialize the key manager.

        Args:
            curve: Elliptic curve to use (default: NIST256p)
        """
        self.curve = curve
        self._signing_key = None
        self._verifying_key = None
        self._generate_keypair()

    def _generate_keypair(self) -> None:
        """Generate a new ECDSA key pair."""
        self._signing_key = SigningKey.generate(curve=self.curve)
        self._verifying_key = self._signing_key.get_verifying_key()

    @property
    def private_key(self) -> SigningKey:
        """Get the private signing key."""
        return self._signing_key

    @property
    def public_key(self) -> VerifyingKey:
        """Get the public verifying key."""
        return self._verifying_key

    @property
    def public_key_pem(self) -> str:
        """Get the public key in PEM format."""
        return self._verifying_key.to_pem().decode()

    @property
    def private_key_pem(self) -> str:
        """Get the private key in PEM format."""
        return self._signing_key.to_pem().decode()

    def save_keys(self, private_key_path: str, public_key_path: str) -> None:
        """
        Save keys to files.

        Args:
            private_key_path: Path to save private key
            public_key_path: Path to save public key
        """
        with open(private_key_path, "wb") as f:
            f.write(self._signing_key.to_pem())

        with open(public_key_path, "wb") as f:
            f.write(self._verifying_key.to_pem())

    @classmethod
    def load_keys(cls, private_key_path: str) -> "KeyManager":
        """
        Load keys from a private key file.

        Args:
            private_key_path: Path to private key file

        Returns:
            KeyManager instance with loaded keys
        """
        with open(private_key_path, "rb") as f:
            signing_key = SigningKey.from_pem(f.read())

        manager = cls.__new__(cls)
        manager.curve = NIST256p
        manager._signing_key = signing_key
        manager._verifying_key = signing_key.get_verifying_key()

        return manager
