# ============================================================================
# VERSORC-MATH: COMPREHENSIVE PYTHON & NUMPY INTEGRATION TEST SUITE
# License: Apache-2.0
# ============================================================================

import math
import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import versorc_math as vm
from versorc_math import (
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
    get_e8_roots,
    get_lloyd_max_3bit_centroids,
    fwht,
)

def test_cl3_algebra_axioms():
    e1 = Cl3.e1()
    e2 = Cl3.e2()
    e3 = Cl3.e3()
    I = Cl3.I()

    # Square of orthonormal vectors in Cl(3, 0) is +1
    assert math.isclose((e1 * e1).scalar, 1.0, abs_tol=1e-5)
    assert math.isclose((e2 * e2).scalar, 1.0, abs_tol=1e-5)
    assert math.isclose((e3 * e3).scalar, 1.0, abs_tol=1e-5)

    # Anticommutation of orthogonal basis vectors: e1 * e2 = -e2 * e1 = e1 ^ e2
    e12 = e1 * e2
    assert math.isclose((e12 + (e2 * e1)).norm(), 0.0, abs_tol=1e-5)
    assert math.isclose(((e1 ^ e2) - e12).norm(), 0.0, abs_tol=1e-5)

    # Pseudoscalar I = e1 * e2 * e3, I^2 = -1 in Cl(3, 0)
    I_prod = e1 * e2 * e3
    assert math.isclose((I - I_prod).norm(), 0.0, abs_tol=1e-5)
    I_sq = I * I
    assert math.isclose(I_sq.scalar, -1.0, abs_tol=1e-5)

    # Hodge dual of e1: e1 * I^{-1} = e1 * (-e123) = -e23
    e23 = e2 * e3
    dual_e1 = e1.dual()
    assert math.isclose((dual_e1 + e23).norm(), 0.0, abs_tol=1e-5)

def test_cl3_inverse_and_reversion():
    # Construct general multivector A
    u = Cl3.from_vector(1.0, 2.0, 3.0)
    s = Cl3.from_scalar(2.5)
    A = s + u

    # Reverse ~A
    A_rev = ~A
    assert math.isclose(A_rev.scalar, 2.5, abs_tol=1e-5)

    # Exact closed-form inverse
    inv_A = A.inverse()
    prod = A * inv_A
    assert math.isclose(prod.scalar, 1.0, abs_tol=1e-4)
    # Higher grade parts must be zero
    assert math.isclose(prod.grade(1).norm(), 0.0, abs_tol=1e-4)
    assert math.isclose(prod.grade(2).norm(), 0.0, abs_tol=1e-4)

def test_cl3_numpy_roundtrip():
    coords = np.array([1.5, -2.0, 3.0, 0.5, -1.0, 4.0, 0.0, -0.5], dtype=np.float32)
    mv = Cl3.from_numpy(coords)
    out_arr = mv.to_numpy()
    np.testing.assert_allclose(coords, out_arr, atol=1e-6)

def test_cl2_planar_algebra():
    e1 = Cl2.e1()
    e2 = Cl2.e2()
    I = Cl2.I()

    # In Cl(2, 0), I = e12 and I^2 = -1 (isomorphic to complex numbers)
    assert math.isclose((I * I).scalar, -1.0, abs_tol=1e-5)
    z1 = Cl2.from_scalar(3.0) + (e1 * 4.0) # Actually 3 + 4*I
    z = Cl2.from_scalar(3.0) + (I * 4.0)
    inv_z = z.inverse()
    prod = z * inv_z
    assert math.isclose(prod.scalar, 1.0, abs_tol=1e-5)

def test_pga_motor_rigid_transformation():
    # Pure translation along X by +10.0
    motor_trans = PgaMotor(translation=[10.0, 0.0, 0.0])
    pt = [1.0, 2.0, 3.0]
    transformed = motor_trans.transform_point(pt)
    assert math.isclose(transformed[0], 11.0, abs_tol=1e-5)
    assert math.isclose(transformed[1], 2.0, abs_tol=1e-5)
    assert math.isclose(transformed[2], 3.0, abs_tol=1e-5)

    # 90 deg rotation around Z axis
    motor_rot = PgaMotor(rotation_axis=[0.0, 0.0, 1.0], angle_rad=math.pi / 2.0)
    transformed_rot = motor_rot.transform_point([1.0, 0.0, 0.0])
    assert math.isclose(transformed_rot[0], 0.0, abs_tol=1e-5)
    assert math.isclose(transformed_rot[1], 1.0, abs_tol=1e-5)
    assert math.isclose(transformed_rot[2], 0.0, abs_tol=1e-5)

    # Motor composition and inverse
    m_combined = motor_trans * motor_rot
    inv_m = m_combined.inverse()
    pt_orig = inv_m.transform_point(m_combined.transform_point(pt))
    np.testing.assert_allclose(pt, pt_orig, atol=1e-5)

def test_cga_inversion_and_circle():
    p1 = CgaPoint(0.0, 0.0, 0.0)
    p2 = CgaPoint(3.0, 4.0, 0.0)
    assert math.isclose(p1.distance(p2), 5.0, abs_tol=1e-5)

    # Sphere centered at origin with radius 2.0
    sphere = CgaSphere(0.0, 0.0, 0.0, 2.0)
    assert math.isclose(sphere.radius, 2.0, abs_tol=1e-5)

    # Inversion of point (4, 0, 0) across sphere of R=2 should map to (1, 0, 0)
    # since r' = R^2 / r = 4 / 4 = 1.0
    p_ext = CgaPoint(4.0, 0.0, 0.0)
    p_inv = sphere.invert_in_sphere(p_ext)
    coords = p_inv.to_euclidean()
    assert math.isclose(coords[0], 1.0, abs_tol=1e-5)
    assert math.isclose(coords[1], 0.0, abs_tol=1e-5)
    assert math.isclose(coords[2], 0.0, abs_tol=1e-5)

    # Conformal Motor: dilation by 2.0
    cm = CgaConformalMotor(dilation_factor=2.0)
    p_dilated = cm.transform_point(p2)
    coords_dilated = p_dilated.to_euclidean()
    assert math.isclose(coords_dilated[0], 6.0, abs_tol=1e-5)
    assert math.isclose(coords_dilated[1], 8.0, abs_tol=1e-5)

def test_sta_spinor_and_lounesto_classes():
    # Standard Dirac electron spinor (Class 2: sigma != 0, omega == 0)
    psi_electron = StaSpinor(scalar=1.0, ex=0.0, ey=0.0, ez=0.0, bx=0.0, by=0.0, bz=0.0, pseudoscalar=0.0)
    assert psi_electron.lounesto_class() == "Class2_DiracElectron"

    cov = psi_electron.bilinear_covariants()
    assert "sigma" in cov
    assert "omega" in cov
    assert "j" in cov
    assert math.isclose(cov["sigma"], 1.0, abs_tol=1e-5)
    assert math.isclose(cov["omega"], 0.0, abs_tol=1e-5)

    # General Dirac spinor with non-zero scalar and pseudoscalar (Class 1)
    psi_general = StaSpinor(scalar=1.0, pseudoscalar=0.5)
    assert psi_general.lounesto_class() == "Class1_DiracGeneral"

def test_e8_roots_and_lloyd_max():
    roots = get_e8_roots()
    assert roots.shape == (240, 8)
    # Norm squared of each E8 minimal root is exactly 2.0
    norms_sq = np.sum(roots ** 2, axis=1)
    np.testing.assert_allclose(norms_sq, 2.0, atol=1e-5)

    centroids = get_lloyd_max_3bit_centroids()
    assert centroids.shape == (8,)

def test_clifford_3bit_tensor_matvec():
    # 256 x 256 matrix (multiples of 128)
    np.random.seed(42)
    W = np.random.randn(256, 256).astype(np.float32)
    x = np.random.randn(256).astype(np.float32)

    tensor = Clifford3BitTensor.from_numpy(W)
    assert tensor.rows == 256
    assert tensor.cols == 256
    assert tensor.effective_bpw == 4.0

    y_quant = tensor.matvec(x)
    assert y_quant.shape == (256,)
    # Verify directional correlation between quantized matvec and FP32 matvec
    y_fp32 = W @ x
    corr = np.corrcoef(y_quant, y_fp32)[0, 1]
    # 3-bit quantized representation should have high correlation (> 0.85)
    assert corr > 0.85, f"Expected high correlation, got {corr}"

if __name__ == "__main__":
    test_cl3_algebra_axioms()
    test_cl3_inverse_and_reversion()
    test_cl3_numpy_roundtrip()
    test_cl2_planar_algebra()
    test_pga_motor_rigid_transformation()
    test_cga_inversion_and_circle()
    test_sta_spinor_and_lounesto_classes()
    test_e8_roots_and_lloyd_max()
    test_clifford_3bit_tensor_matvec()
    print("ALL 9 PYTHON INTEGRATION TESTS PASSED EMPIRICALLY!")
