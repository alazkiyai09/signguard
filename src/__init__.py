"""
SignGuard: ECDSA-based cryptographic verification system for detecting poisoning attacks in federated learning.

Copyright (c) 2024 Ahmad Whafa Azka Al Azkiyai
Licensed under MIT License - see LICENSE file for details.
"""

__version__ = "0.1.0"
__author__ = "Ahmad Whafa Azka Al Azkiyai"
__email__ = "azka.alazkiyai@outlook.com"

from signguard.client import SignGuardClient
from signguard.server import SignGuardServer

__all__ = ["SignGuardClient", "SignGuardServer", "__version__"]
