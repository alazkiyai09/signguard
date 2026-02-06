"""
SignGuard Server Implementation

This module implements the federated learning aggregation server with
signature verification and attack detection capabilities.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
import numpy as np
import json
from ecdsa import VerifyingKey, NIST256p
import hashlib

from signguard.server.aggregator import Aggregator
from signguard.server.verifier import SignatureVerifier


class SignGuardServer:
    """
    Federated learning aggregation server with security features.

    This server receives signed model updates from clients, verifies
    signatures, detects attacks, and performs robust aggregation.

    Attributes:
        global_model: The global PyTorch model
        aggregator: Handles Byzantine-robust aggregation
        verifier: Verifies ECDSA signatures
        client_reputations: Reputation scores for each client
        detection_history: History of attack detection events
    """

    def __init__(
        self,
        model: nn.Module,
        num_clients: int,
        num_malicious: int = 0,
        aggregation_method: str = "multi_krum",
        signature_verification: bool = True,
    ):
        """
        Initialize the SignGuard server.

        Args:
            model: The global PyTorch model
            num_clients: Total number of clients
            num_malicious: Estimated number of malicious clients
            aggregation_method: Aggregation algorithm to use
            signature_verification: Whether to verify signatures
        """
        self.global_model = model
        self.num_clients = num_clients
        self.num_malicious = num_malicious
        self.signature_verification = signature_verification

        # Initialize components
        self.aggregator = Aggregator(
            method=aggregation_method,
            num_malicious=num_malicious,
        )
        self.verifier = SignatureVerifier()

        # Reputation system
        self.client_reputations: Dict[int, float] = {
            i: 1.0 for i in range(num_clients)
        }

        # Detection history
        self.detection_history: List[Dict] = []

        # Client public keys
        self.client_keys: Dict[int, VerifyingKey] = {}

    def register_client(self, client_id: int, public_key_pem: str) -> bool:
        """
        Register a client's public key.

        Args:
            client_id: Client identifier
            public_key_pem: PEM-encoded public key

        Returns:
            True if registration successful
        """
        try:
            public_key = VerifyingKey.from_pem(public_key_pem.encode())
            self.client_keys[client_id] = public_key
            return True
        except Exception as e:
            print(f"Failed to register client {client_id}: {e}")
            return False

    def verify_update(
        self,
        update: Dict[str, Any],
    ) -> tuple[bool, Optional[str]]:
        """
        Verify a client's update signature.

        Args:
            update: Dictionary containing updates and signature

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.signature_verification:
            return True, None

        client_id = update.get("client_id")
        signature = update.get("signature")
        public_key_pem = update.get("public_key")
        updates = update.get("updates", {})

        # Check all required fields
        if not all([client_id is not None, signature, public_key_pem, updates]):
            return False, "Missing required fields"

        # Serialize and hash updates
        update_bytes = json.dumps(updates, sort_keys=True).encode()
        update_hash = hashlib.sha256(update_bytes).digest()

        # Verify signature
        try:
            is_valid = self.verifier.verify(
                signature=signature,
                message_hash=update_hash,
                public_key_pem=public_key_pem,
            )
            return is_valid, None if is_valid else "Invalid signature"
        except Exception as e:
            return False, f"Verification error: {str(e)}"

    def detect_anomalies(
        self,
        updates: List[Dict[str, Any]],
    ) -> List[int]:
        """
        Detect potentially malicious updates using statistical analysis.

        Args:
            updates: List of client updates

        Returns:
            List of suspicious client IDs
        """
        if len(updates) < 3:
            return []

        # Compute update norms
        norms = []
        for update in updates:
            updates_dict = update.get("updates", {})
            norm = np.sqrt(sum(
                np.sum(np.array(v) ** 2)
                for v in updates_dict.values()
            ))
            norms.append(norm)

        # Detect outliers using z-score
        norms = np.array(norms)
        mean_norm = np.mean(norms)
        std_norm = np.std(norms)

        threshold = 2.5
        suspicious = []

        for i, norm in enumerate(norms):
            z_score = abs((norm - mean_norm) / (std_norm + 1e-8))
            if z_score > threshold:
                client_id = updates[i].get("client_id")
                if client_id is not None:
                    suspicious.append(client_id)

        return suspicious

    def aggregate_updates(
        self,
        verified_updates: List[Dict[str, Any]],
    ) -> Dict[str, np.ndarray]:
        """
        Aggregate verified client updates using Byzantine-robust method.

        Args:
            verified_updates: List of verified client updates

        Returns:
            Dictionary of aggregated parameters
        """
        # Extract updates
        update_list = []
        for update in verified_updates:
            updates_dict = update.get("updates", {})
            update_list.append(updates_dict)

        # Perform aggregation
        aggregated = self.aggregator.aggregate(update_list)

        return aggregated

    def update_global_model(
        self,
        aggregated_params: Dict[str, np.ndarray],
    ) -> None:
        """
        Update the global model with aggregated parameters.

        Args:
            aggregated_params: Aggregated parameters from clients
        """
        with torch.no_grad():
            for name, param in self.global_model.named_parameters():
                if name in aggregated_params:
                    param.data = torch.tensor(
                        aggregated_params[name],
                        dtype=param.dtype,
                        device=param.device,
                    )

    def federated_round(
        self,
        client_updates: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Execute one round of federated learning.

        Args:
            client_updates: List of updates from clients

        Returns:
            Dictionary with round results
        """
        results = {
            "total_received": len(client_updates),
            "verified": 0,
            "rejected": 0,
            "suspicious": [],
        }

        verified_updates = []

        for update in client_updates:
            # Verify signature
            is_valid, error = self.verify_update(update)

            if is_valid:
                verified_updates.append(update)
                results["verified"] += 1
            else:
                results["rejected"] += 1
                # Update reputation
                client_id = update.get("client_id")
                if client_id is not None:
                    self.client_reputations[client_id] *= 0.8

        # Detect anomalies
        suspicious = self.detect_anomalies(verified_updates)
        results["suspicious"] = suspicious

        # Filter out suspicious updates
        clean_updates = [
            u for u in verified_updates
            if u.get("client_id") not in suspicious
        ]

        # Aggregate
        if clean_updates:
            aggregated = self.aggregate_updates(clean_updates)
            self.update_global_model(aggregated)

        # Record detection event
        self.detection_history.append(results)

        return results

    def get_detection_report(self) -> Dict:
        """Generate a report of attack detection activity."""
        return {
            "total_rounds": len(self.detection_history),
            "total_rejected": sum(r["rejected"] for r in self.detection_history),
            "total_suspicious": sum(len(r["suspicious"]) for r in self.detection_history),
            "client_reputations": self.client_reputations,
        }


def main():
    """Main entry point for the server CLI."""
    import argparse

    parser = argparse.ArgumentParser(description="SignGuard Server")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--port", type=int, default=5000, help="Server port")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    print(f"Starting SignGuard server on port {args.port}")
    # Server implementation would continue here


if __name__ == "__main__":
    main()
