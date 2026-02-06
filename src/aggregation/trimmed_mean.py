"""
Trimmed Mean Aggregation Algorithm

Implementation of the trimmed mean algorithm for Byzantine-robust federated learning.
"""

import numpy as np
from typing import List


def trimmed_mean(
    updates: List[np.ndarray],
    trim_ratio: float = 0.2,
) -> np.ndarray:
    """
    Trimmed mean aggregation algorithm.

    Removes extreme values from both ends and averages the rest.

    Args:
        updates: List of parameter updates as numpy arrays
        trim_ratio: Fraction to trim from each end

    Returns:
        Trimmed mean parameters
    """
    n = len(updates)
    if n == 0:
        raise ValueError("No updates provided")

    if n == 1:
        return updates[0]

    # Stack updates
    stacked = np.stack(updates)

    # Number to trim
    num_trim = max(1, int(n * trim_ratio))

    # Sort and trim along axis 0
    sorted_updates = np.sort(stacked, axis=0)
    trimmed = sorted_updates[num_trim:-num_trim]

    # Average
    return np.mean(trimmed, axis=0)
