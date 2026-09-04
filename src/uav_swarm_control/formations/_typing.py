"""Shared array types for formation geometry."""

import numpy as np
from numpy.typing import NDArray

type FloatArray = NDArray[np.float64]

__all__ = ["FloatArray"]
