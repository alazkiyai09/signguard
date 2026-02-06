"""
Multi-Krum Aggregation Algorithm

Implementation of the Multi-Krum algorithm for Byzantine-robust federated learning.
"""

import numpy as np
from typing import List


def multi_krum(
    updates: List[np.ndarray],
    num_malicious: int = 0,
) -> np.ndarray:
    """
    Multi-Krum aggregation algorithm.

    Selects multiple (n-2f) closest updates and averages them.

    Args:
        updates: List of parameter updates as numpy arrays
        num_malicious: Estimated number of malicious clients

    Returns:
        Averaged parameters from selected updates
    """
    n = len(updates)
    if n == 0:
        raise ValueError("No updates provided")

    if n == 1:
        return updates[0]

    # Compute pairwise distances
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sum((updates[i] - updates[j]) ** 2)
            distances[i, j] = dist
            distances[j, i] = dist

    # Krum scores
    f = num_malicious
    num_closest = max(0, n - f - 2)

    scores = np.zeros(n)
    for i in range(n):
        sorted_dist = np.sort(distances[i])
        scores[i] = np.sum(sorted_dist[:num_closest + 1])

    # Select n-2f updates with best scores
    num_selected = max(1, n - 2 * f)
    selected_indices = np.argsort(scores)[:num_selected]

    # Average selected updates
    selected = [updates[i] for i in selected_indices]
    return np.mean(selected, axis=0)
