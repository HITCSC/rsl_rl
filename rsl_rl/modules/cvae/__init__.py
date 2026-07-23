"""`rsl_rl.modules.cvae` package initializer.

Expose the primary CVAE classes so callers can do:

    from rsl_rl.modules.cvae import VaeConfig, VaeModel

Adding this file also avoids import issues on some Python setups and makes
the package explicit.
"""

from .config import VaeConfig  # noqa: F401
from .model import VaeModel, HeightMapCNN  # noqa: F401

__all__ = ["VaeConfig", "VaeModel", "HeightMapCNN"]
