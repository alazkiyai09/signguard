"""
Krum Aggregation Algorithm

Implementation of the Krum algorithm for Byzantine-robust federated learning.
"""

import numpy as np
from typing import List, Dict


def krum(
    updates: List[np.ndarray],
    num_malicious: int = 0,
) -> np.ndarray:
    """
    Krum aggregation algorithm.

    Selects the update closest to other updates, robust to
    a limited number of Byzantine failures.

    Args:
        updates: List of parameter updates as numpy arrays
        num_malicious: Estimated number of malicious clients

    Returns:
        Selected update
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

    # Krum score: sum of closest n-f-2 distances
    f = num_malicious
    num_closest = max(0, n - f - 2)

    scores = np.zeros(n)
    for i in range(n):
        sorted_dist = np.sort(distances[i])
        scores[i] = np.sum(sorted_dist[:num_closest + 1])

    # Return update with minimum score
    selected_idx = np.argmin(scores)
    return updates[selected_idx]
