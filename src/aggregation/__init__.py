"""SignGuard Aggregation Module"""

from signguard.aggregation.krum import krum
from signguard.aggregation.multi_krum import multi_krum
from signguard.aggregation.trimmed_mean import trimmed_mean

__all__ = ["krum", "multi_krum", "trimmed_mean"]
