"""
Tests for aggregation algorithms.
"""

import pytest
import numpy as np
from signguard.aggregation.krum import krum
from signguard.aggregation.multi_krum import multi_krum
from signguard.aggregation.trimmed_mean import trimmed_mean


def test_krum_basic():
    """Test basic Krum aggregation."""
    updates = [
        np.array([1.0, 2.0, 3.0]),
        np.array([1.1, 2.1, 3.1]),
        np.array([1.0, 2.0, 3.0]),
    ]

    result = krum(updates, num_malicious=0)
    assert result is not None
    assert result.shape == (3,)


def test_multi_krum_basic():
    """Test basic Multi-Krum aggregation."""
    updates = [
        np.array([1.0, 2.0, 3.0]),
        np.array([1.1, 2.1, 3.1]),
        np.array([1.0, 2.0, 3.0]),
    ]

    result = multi_krum(updates, num_malicious=0)
    assert result is not None
    assert result.shape == (3,)


def test_trimmed_mean_basic():
    """Test basic trimmed mean aggregation."""
    updates = [
        np.array([1.0, 2.0, 3.0]),
        np.array([1.1, 2.1, 3.1]),
        np.array([1.0, 2.0, 3.0]),
        np.array([1.05, 2.05, 3.05]),
    ]

    result = trimmed_mean(updates, trim_ratio=0.25)
    assert result is not None
    assert result.shape == (3,)


def test_krum_byzantine():
    """Test Krum with Byzantine updates."""
    # Benign updates
    benign = [np.array([1.0, 1.0, 1.0]) for _ in range(7)]

    # Malicious update (far from benign)
    malicious = [np.array([10.0, 10.0, 10.0])]

    updates = benign + malicious

    result = krum(updates, num_malicious=1)

    # Result should be closer to benign updates
    distance_to_benign = np.linalg.norm(result - benign[0])
    distance_to_malicious = np.linalg.norm(result - malicious[0])

    assert distance_to_benign < distance_to_malicious
