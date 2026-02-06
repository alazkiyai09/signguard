"""
Tests for cryptographic functions.
"""

import pytest
import hashlib
from signguard.crypto.keys import KeyManager
from signguard.crypto.ecdsa import SignatureManager


def test_key_generation():
    """Test that keys are generated correctly."""
    manager = KeyManager()
    assert manager.private_key is not None
    assert manager.public_key is not None


def test_public_key_pem():
    """Test public key PEM format."""
    manager = KeyManager()
    pem = manager.public_key_pem
    assert "BEGIN PUBLIC KEY" in pem
    assert "END PUBLIC KEY" in pem


def test_sign_and_verify():
    """Test signing and verification."""
    manager = KeyManager()
    signer = SignatureManager(manager.private_key)

    message = b"Test message for signing"
    signature = signer.sign(message)

    assert signature is not None
    assert len(signature) > 0

    # Verify the signature
    from ecdsa import VerifyingKey, BadSignatureError
    try:
        manager.public_key.verify(
            signature.encode(),
            message,
            hashfunc=hashlib.sha256
        )
        verified = True
    except BadSignatureError:
        verified = False

    assert verified
