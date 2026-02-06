"""
SignGuard Client Implementation

This module implements the federated learning client with cryptographic signature capabilities.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Callable
import numpy as np
from ecdsa import SigningKey, NIST256p
import hashlib
import json

from signguard.crypto.keys import KeyManager
from signguard.crypto.ecdsa import SignatureManager
from signguard.client.trainer import LocalTrainer


class SignGuardClient:
    """
    A federated learning client with ECDSA signature capabilities.

    This client trains a local model and signs the updates before sending
    them to the aggregation server.

    Attributes:
        client_id: Unique identifier for this client
        model: The local PyTorch model
        key_manager: Manages cryptographic keys
        signature_manager: Handles signing operations
        trainer: Handles local training
        is_malicious: Whether this client simulates malicious behavior
    """

    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        device: torch.device,
        is_malicious: bool = False,
        attack_type: Optional[str] = None,
    ):
        """
        Initialize a SignGuard client.

        Args:
            client_id: Unique identifier for this client
            model: PyTorch model to train
            device: Device to train on (cpu or cuda)
            is_malicious: Whether this client acts maliciously
            attack_type: Type of attack ('label_flip', 'backdoor', 'poison')
        """
        self.client_id = client_id
        self.model = model.to(device)
        self.device = device
        self.is_malicious = is_malicious
        self.attack_type = attack_type

        # Initialize cryptographic components
        self.key_manager = KeyManager()
        self.signature_manager = SignatureManager(self.key_manager.private_key)

        # Initialize trainer
        self.trainer = LocalTrainer(
            model=self.model,
            device=device,
            is_malicious=is_malicious,
            attack_type=attack_type,
        )

        # Server connection
        self.server_url: Optional[str] = None
        self.reputation_score = 1.0

    def connect(self, server_url: str) -> bool:
        """Connect to the aggregation server."""
        self.server_url = server_url
        # In real implementation, establish gRPC connection
        return True

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        epochs: int = 5,
        learning_rate: float = 0.01,
    ) -> Dict[str, torch.Tensor]:
        """
        Train the local model on the provided data.

        Args:
            train_loader: DataLoader containing training data
            epochs: Number of local training epochs
            learning_rate: Learning rate for optimization

        Returns:
            Dictionary of model updates (parameter differences)
        """
        # Store initial parameters
        initial_params = {
            name: param.clone().detach()
            for name, param in self.model.named_parameters()
        }

        # Perform local training
        self.trainer.train(train_loader, epochs, learning_rate)

        # Compute updates
        updates = {
            name: param.clone().detach() - initial_params[name]
            for name, param in self.model.named_parameters()
        }

        return updates

    def sign_updates(
        self, updates: Dict[str, torch.Tensor]
    ) -> Dict[str, str]:
        """
        Sign the model updates using ECDSA.

        Args:
            updates: Dictionary of parameter updates

        Returns:
            Dictionary containing signature data
        """
        # Serialize updates
        update_bytes = self._serialize_updates(updates)
        update_hash = hashlib.sha256(update_bytes).digest()

        # Sign the hash
        signature = self.signature_manager.sign(update_hash)

        return {
            "signature": signature,
            "public_key": self.key_manager.public_key_pem,
            "client_id": str(self.client_id),
        }

    def _serialize_updates(
        self, updates: Dict[str, torch.Tensor]
    ) -> bytes:
        """Serialize model updates to bytes."""
        serialized = {}
        for name, tensor in updates.items():
            serialized[name] = tensor.cpu().numpy().tolist()
        return json.dumps(serialized, sort_keys=True).encode()

    def get_signed_update(
        self,
        updates: Dict[str, torch.Tensor],
    ) -> Dict:
        """
        Get signed model update ready for transmission.

        Args:
            updates: Dictionary of parameter updates

        Returns:
            Dictionary containing updates and signature
        """
        signature_data = self.sign_updates(updates)

        return {
            "client_id": self.client_id,
            "updates": {
                name: tensor.cpu().numpy().tolist()
                for name, tensor in updates.items()
            },
            "signature": signature_data["signature"],
            "public_key": signature_data["public_key"],
        }

    def receive_global_model(
        self,
        global_params: Dict[str, np.ndarray],
    ) -> None:
        """
        Update local model with global parameters from server.

        Args:
            global_params: Dictionary of global model parameters
        """
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in global_params:
                    param.data = torch.tensor(
                        global_params[name],
                        dtype=param.dtype,
                        device=self.device,
                    )


def main():
    """Main entry point for the client CLI."""
    import argparse

    parser = argparse.ArgumentParser(description="SignGuard Client")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()

    # Load configuration and start client
    with open(args.config) as f:
        config = json.load(f)

    # Initialize and run client
    # Implementation depends on specific use case
    print(f"Starting SignGuard client with config: {args.config}")


if __name__ == "__main__":
    main()
