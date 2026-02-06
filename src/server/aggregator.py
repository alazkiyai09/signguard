"""
Aggregator Module

Implements various Byzantine-robust aggregation algorithms for federated learning.
"""

import numpy as np
from typing import List, Dict, Any
import torch


class Aggregator:
    """
    Federated learning aggregator with multiple Byzantine-robust methods.

    Supported methods:
    - fedavg: Standard Federated Averaging
    - krum: Krum algorithm for Byzantine robustness
    - multi_krum: Multi-Krum (aggregates multiple updates)
    - trimmed_mean: Trimmed mean aggregation
    - median: Coordinate-wise median
    """

    def __init__(
        self,
        method: str = "multi_krum",
        num_malicious: int = 0,
    ):
        """
        Initialize the aggregator.

        Args:
            method: Aggregation method to use
            num_malicious: Estimated number of malicious clients
        """
        self.method = method.lower()
        self.num_malicious = num_malicious

        self.valid_methods = [
            "fedavg",
            "krum",
            "multi_krum",
            "trimmed_mean",
            "median",
        ]

        if self.method not in self.valid_methods:
            raise ValueError(f"Unknown aggregation method: {method}")

    def aggregate(
        self,
        updates: List[Dict[str, Any]],
    ) -> Dict[str, np.ndarray]:
        """
        Aggregate client updates using the configured method.

        Args:
            updates: List of update dictionaries from clients

        Returns:
            Dictionary of aggregated parameters
        """
        if not updates:
            return {}

        if self.method == "fedavg":
            return self._fedavg(updates)
        elif self.method == "krum":
            return self._krum(updates)
        elif self.method == "multi_krum":
            return self._multi_krum(updates)
        elif self.method == "trimmed_mean":
            return self._trimmed_mean(updates)
        elif self.method == "median":
            return self._median(updates)

    def _fedavg(
        self,
        updates: List[Dict[str, Any]],
    ) -> Dict[str, np.ndarray]:
        """
        Standard Federated Averaging.

        Args:
            updates: List of update dictionaries

        Returns:
            Averaged parameters
        """
        # Flatten all updates
        flattened = [self._flatten(u) for u in updates]

        # Average
        avg = np.mean(flattened, axis=0)

        # Unflatten to original shape
        return self._unflatten(avg, updates[0])

    def _krum(
        self,
        updates: List[Dict[str, Any]],
    ) -> Dict[str, np.ndarray]:
        """
        Krum aggregation algorithm.

        Selects the update closest to others, designed to be
        robust against Byzantine failures.

        Args:
            updates: List of update dictionaries

        Returns:
            Selected parameters
        """
        flattened = [self._flatten(u) for u in updates]

        # Compute distances
        distances = np.zeros((len(flattened), len(flattened)))
        for i in range(len(flattened)):
            for j in range(i + 1, len(flattened)):
                dist = np.sum((flattened[i] - flattened[j]) ** 2)
                distances[i, j] = dist
                distances[j, i] = dist

        # Krum score: sum of closest n-f-2 distances
        n = len(flattened)
        f = self.num_malicious
        num_closest = n - f - 2

        scores = []
        for i in range(n):
            sorted_dist = np.sort(distances[i])
            scores.append(np.sum(sorted_dist[:num_closest]))

        # Select update with minimum score
        selected_idx = np.argmin(scores)
        return self._unflatten(flattened[selected_idx], updates[0])

    def _multi_krum(
        self,
        updates: List[Dict[str, Any]],
    ) -> Dict[str, np.ndarray]:
        """
        Multi-Krum aggregation.

        Selects multiple (n-2f) closest updates and averages them.

        Args:
            updates: List of update dictionaries

        Returns:
            Averaged parameters from selected updates
        """
        flattened = [self._flatten(u) for u in updates]

        # Compute distances
        distances = np.zeros((len(flattened), len(flattened)))
        for i in range(len(flattened)):
            for j in range(i + 1, len(flattened)):
                dist = np.sum((flattened[i] - flattened[j]) ** 2)
                distances[i, j] = dist
                distances[j, i] = dist

        # Krum scores
        n = len(flattened)
        f = self.num_malicious
        num_closest = n - f - 2

        scores = []
        for i in range(n):
            sorted_dist = np.sort(distances[i])
            scores.append(np.sum(sorted_dist[:num_closest]))

        # Select n-2f updates with best scores
        num_selected = max(1, n - 2 * f)
        selected_indices = np.argsort(scores)[:num_selected]

        # Average selected updates
        selected = [flattened[i] for i in selected_indices]
        avg = np.mean(selected, axis=0)

        return self._unflatten(avg, updates[0])

    def _trimmed_mean(
        self,
        updates: List[Dict[str, Any]],
        trim_ratio: float = 0.2,
    ) -> Dict[str, np.ndarray]:
        """
        Trimmed mean aggregation.

        Removes extreme values and averages the rest.

        Args:
            updates: List of update dictionaries
            trim_ratio: Fraction to trim from each end

        Returns:
            Trimmed mean parameters
        """
        flattened = [self._flatten(u) for u in updates]
        flattened = np.array(flattened)

        # Number to trim
        n = len(flattened)
        num_trim = int(n * trim_ratio)

        # Trim and average
        sorted_arr = np.sort(flattened, axis=0)
        if num_trim > 0:
            trimmed = sorted_arr[num_trim:-num_trim]
        else:
            trimmed = sorted_arr

        avg = np.mean(trimmed, axis=0)
        return self._unflatten(avg, updates[0])

    def _median(
        self,
        updates: List[Dict[str, Any]],
    ) -> Dict[str, np.ndarray]:
        """
        Coordinate-wise median aggregation.

        Args:
            updates: List of update dictionaries

        Returns:
            Median parameters
        """
        flattened = [self._flatten(u) for u in updates]
        flattened = np.array(flattened)

        median = np.median(flattened, axis=0)
        return self._unflatten(median, updates[0])

    def _flatten(self, update: Dict[str, Any]) -> np.ndarray:
        """Flatten update dictionary to 1D array."""
        arrays = []
        for v in update.values():
            if isinstance(v, list):
                arrays.append(np.array(v).flatten())
            elif isinstance(v, np.ndarray):
                arrays.append(v.flatten())
        return np.concatenate(arrays) if arrays else np.array([])

    def _unflatten(
        self,
        flat: np.ndarray,
        template: Dict[str, Any],
    ) -> Dict[str, np.ndarray]:
        """Unflatten 1D array back to update dictionary format."""
        result = {}
        idx = 0

        for key, value in template.items():
            if isinstance(value, list):
                arr = np.array(value)
                size = arr.size
                result[key] = flat[idx:idx + size].reshape(arr.shape)
                idx += size
            elif isinstance(value, np.ndarray):
                size = value.size
                result[key] = flat[idx:idx + size].reshape(value.shape)
                idx += size

        return result
