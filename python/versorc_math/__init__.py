# ============================================================================
# VERSORC-MATH: CLIFFORD GEOMETRIC ALGEBRA & VERSOR QUANTIZATION FOR PYTHON
# License: Apache-2.0
# ============================================================================

from ._versorc_math import (
    Cl3,
    Cl2,
    PgaMotor,
    CgaPoint,
    CgaSphere,
    CgaCircle,
    CgaConformalMotor,
    StaVector,
    StaSpinor,
    DyadicWalshVersor,
    Clifford3BitTensor,
    PoincarePoint,
    get_e8_roots,
    get_lloyd_max_3bit_centroids,
    fwht,
)

from . import plotting
from . import ganja

__all__ = [
    "Cl3",
    "Cl2",
    "PgaMotor",
    "CgaPoint",
    "CgaSphere",
    "CgaCircle",
    "CgaConformalMotor",
    "StaVector",
    "StaSpinor",
    "DyadicWalshVersor",
    "Clifford3BitTensor",
    "PoincarePoint",
    "get_e8_roots",
    "get_lloyd_max_3bit_centroids",
    "fwht",
    "plotting",
    "ganja",
]

__version__ = "0.1.0"
