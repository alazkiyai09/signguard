"""
Local Trainer Module

Handles the local training process for federated learning clients.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional
from tqdm import tqdm


class LocalTrainer:
    """
    Handles local model training for federated learning.

    Supports both benign and malicious training behaviors for
    security testing and evaluation.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        is_malicious: bool = False,
        attack_type: Optional[str] = None,
    ):
        """
        Initialize the local trainer.

        Args:
            model: PyTorch model to train
            device: Device to train on
            is_malicious: Whether to simulate malicious behavior
            attack_type: Type of attack to simulate
        """
        self.model = model
        self.device = device
        self.is_malicious = is_malicious
        self.attack_type = attack_type

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        epochs: int = 5,
        learning_rate: float = 0.01,
    ) -> dict:
        """
        Train the model for the specified number of epochs.

        Args:
            train_loader: DataLoader containing training data
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer

        Returns:
            Dictionary with training metrics
        """
        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        self.model.train()

        metrics = {"losses": [], "accuracies": []}

        for epoch in range(epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
            for batch_idx, (data, target) in enumerate(pbar):
                data, target = data.to(self.device), target.to(self.device)

                # Apply attack if malicious
                if self.is_malicious:
                    data, target = self._apply_attack(data, target)

                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()

                # Apply gradient attack if malicious
                if self.is_malicious:
                    self._apply_gradient_attack()

                optimizer.step()

                epoch_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

                pbar.set_postfix({"loss": loss.item(), "acc": 100.0 * correct / total})

            avg_loss = epoch_loss / len(train_loader)
            accuracy = 100.0 * correct / total

            metrics["losses"].append(avg_loss)
            metrics["accuracies"].append(accuracy)

        return metrics

    def _apply_attack(
        self,
        data: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple:
        """
        Apply data poisoning attack.

        Args:
            data: Input data batch
            target: Target labels

        Returns:
            Tuple of (modified_data, modified_target)
        """
        if self.attack_type == "label_flip":
            # Flip labels to a different class
            num_classes = target.max().item() + 1
            target = (target + 1) % num_classes
        elif self.attack_type == "backdoor":
            # Add backdoor trigger (simple pattern)
            data[:, :, 0:2, 0:2] += 0.5
            # Set backdoor target
            target[:] = 0
        elif self.attack_type == "poison":
            # Add noise to data
            noise = torch.randn_like(data) * 0.3
            data = data + noise
            data = torch.clamp(data, 0, 1)

        return data, target

    def _apply_gradient_attack(self) -> None:
        """Apply gradient poisoning attack."""
        if self.attack_type == "gradient":
            with torch.no_grad():
                for param in self.model.parameters():
                    if param.grad is not None:
                        # Add random noise to gradients
                        noise = torch.randn_like(param.grad) * 0.5
                        param.grad += noise
        elif self.attack_type == "sign_flip":
            with torch.no_grad():
                for param in self.model.parameters():
                    if param.grad is not None:
                        # Flip gradient signs
                        param.grad = -param.grad
