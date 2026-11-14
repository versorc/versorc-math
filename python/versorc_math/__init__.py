# ============================================================================
# VERSORC-MATH: CLIFFORD GEOMETRIC ALGEBRA & VERSOR QUANTIZATION FOR PYTHON
# License: Apache-2.0
# ============================================================================

from ._versorc_math import (
    VersorBufferF32,
    VersorBufferI32,
    Cl3,
    Cl2,
    PgaMotor,
    PgaPoint,
    PgaPlane,
    PgaLine,
    Quaternion,
    DualQuaternion,
    CgaPoint,
    CgaSphere,
    CgaCircle,
    CgaConformalMotor,
    StaVector,
    StaSpinor,
    DyadicWalshVersor,
    Clifford3BitTensor,
    PoincarePoint,
    CayleyKleinSpace,
    get_e8_roots,
    project_to_e8,
    project_to_d8,
    get_lloyd_max_3bit_centroids,
    fwht,
)

try:
    from ._versorc_math import GpuComputeContext
except ImportError:
    GpuComputeContext = None

from . import ganja

__all__ = [
    "Cl3",
    "Cl2",
    "PgaMotor",
    "PgaPoint",
    "PgaPlane",
    "PgaLine",
    "Quaternion",
    "DualQuaternion",
    "CgaPoint",
    "CgaSphere",
    "CgaCircle",
    "CgaConformalMotor",
    "StaVector",
    "StaSpinor",
    "DyadicWalshVersor",
    "Clifford3BitTensor",
    "PoincarePoint",
    "CayleyKleinSpace",
    "VersorBufferF32",
    "VersorBufferI32",
    "get_e8_roots",
    "project_to_e8",
    "project_to_d8",
    "get_lloyd_max_3bit_centroids",
    "fwht",
    "GpuComputeContext",
    "ganja",
]

__version__ = "0.1.2"
