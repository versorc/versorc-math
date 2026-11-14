# ============================================================================
# VERSORC-MATH: PYTHON PEP 3118 & GEOMETRIC ALGEBRA INTEGRATION TEST SUITE
# License: Apache-2.0
# ============================================================================

import array
import json
import math
import os
import random
import struct
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import versorc_math as vm
from versorc_math import (
    CgaCircle,
    CgaConformalMotor,
    CgaPoint,
    CgaSphere,
    Cl2,
    Cl3,
    Clifford3BitTensor,
    PgaMotor,
    StaSpinor,
    VersorBufferF32,
    VersorBufferI32,
    ganja,
    get_e8_roots,
    get_lloyd_max_3bit_centroids,
)

try:
    import numpy as np  # Optional interop verification only; never required.
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False


def assert_vec_close(test_case, actual, expected, abs_tol=1e-5):
    test_case.assertEqual(len(actual), len(expected))
    for i, (a, e) in enumerate(zip(actual, expected)):
        test_case.assertTrue(
            math.isclose(a, e, abs_tol=abs_tol),
            f"Index {i}: expected {e}, got {a}",
        )


def pearson_corr(xs, ys):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    dx = math.sqrt(sum((a - mx) ** 2 for a in xs))
    dy = math.sqrt(sum((b - my) ** 2 for b in ys))
    return num / (dx * dy)


class TestGeometricAlgebrasStdlib(unittest.TestCase):
    def test_cl3_algebra_axioms(self):
        e1 = Cl3.e1()
        e2 = Cl3.e2()
        e3 = Cl3.e3()
        ps = Cl3.I()

        self.assertTrue(math.isclose((e1 * e1).scalar, 1.0, abs_tol=1e-5))
        self.assertTrue(math.isclose((e2 * e2).scalar, 1.0, abs_tol=1e-5))
        self.assertTrue(math.isclose((e3 * e3).scalar, 1.0, abs_tol=1e-5))
        self.assertTrue(math.isclose(e1.x, 1.0, abs_tol=1e-6))
        self.assertTrue(math.isclose(e2.y, 1.0, abs_tol=1e-6))
        self.assertTrue(math.isclose(e3.z, 1.0, abs_tol=1e-6))

        e12 = e1 * e2
        self.assertTrue(math.isclose((e12 + (e2 * e1)).norm(), 0.0, abs_tol=1e-5))
        self.assertTrue(math.isclose(((e1 ^ e2) - e12).norm(), 0.0, abs_tol=1e-5))

        ps_prod = e1 * e2 * e3
        self.assertTrue(math.isclose((ps - ps_prod).norm(), 0.0, abs_tol=1e-5))
        self.assertTrue(math.isclose((ps * ps).scalar, -1.0, abs_tol=1e-5))

        e23 = e2 * e3
        dual_e1 = e1.dual()
        self.assertTrue(math.isclose((dual_e1 + e23).norm(), 0.0, abs_tol=1e-5))

    def test_cl3_inverse_and_reversion(self):
        u = Cl3.from_vector(1.0, 2.0, 3.0)
        s = Cl3.from_scalar(2.5)
        mv = s + u

        mv_rev = ~mv
        self.assertTrue(math.isclose(mv_rev.scalar, 2.5, abs_tol=1e-5))

        inv_mv = mv.inverse()
        prod = mv * inv_mv
        self.assertTrue(math.isclose(prod.scalar, 1.0, abs_tol=1e-4))
        self.assertTrue(math.isclose(prod.grade(1).norm(), 0.0, abs_tol=1e-4))
        self.assertTrue(math.isclose(prod.grade(2).norm(), 0.0, abs_tol=1e-4))

    def test_cl3_pep3118_and_buffer_roundtrip(self):
        coords = array.array("f", [1.5, -2.0, 3.0, 0.5, -1.0, 4.0, 0.0, -0.5])
        mv = Cl3.from_buffer(coords)

        buf = mv.to_buffer()
        self.assertIsInstance(buf, VersorBufferF32)
        self.assertEqual(buf.shape, (8,))
        self.assertEqual(buf.ndim, 1)
        self.assertEqual(buf.size, 8)
        self.assertEqual(buf.dtype, "float32")
        self.assertEqual(buf.nbytes, 32)
        assert_vec_close(self, buf.tolist(), list(coords), abs_tol=1e-6)

        with memoryview(mv) as view:
            self.assertEqual(view.shape, (8,))
            self.assertEqual(view.format, "f")
            self.assertEqual(view.itemsize, 4)
            assert_vec_close(self, view.tolist(), list(coords), abs_tol=1e-6)

        with memoryview(buf) as bview:
            self.assertEqual(bview.shape, (8,))
            self.assertEqual(struct.unpack("<8f", buf.tobytes()), tuple(coords))

    def test_cl2_planar_algebra(self):
        e1 = Cl2.e1()
        e2 = Cl2.e2()
        ps = Cl2.I()

        self.assertTrue(math.isclose((ps * ps).scalar, -1.0, abs_tol=1e-5))
        self.assertTrue(math.isclose((e1 * e2).e12, 1.0, abs_tol=1e-5))
        z = Cl2.from_scalar(3.0) + (ps * 4.0)
        prod = z * z.inverse()
        self.assertTrue(math.isclose(prod.scalar, 1.0, abs_tol=1e-5))

        buf_mv = Cl2.from_buffer(array.array("f", [3.0, 0.0, 0.0, 4.0]))
        with memoryview(buf_mv) as view:
            self.assertEqual(view.shape, (4,))
            assert_vec_close(self, view.tolist(), [3.0, 0.0, 0.0, 4.0], abs_tol=1e-6)

    def test_pga_motor_rigid_transformation(self):
        motor_trans = PgaMotor(translation=[10.0, 0.0, 0.0])
        pt = [1.0, 2.0, 3.0]
        assert_vec_close(self, motor_trans.transform_point(pt), [11.0, 2.0, 3.0], abs_tol=1e-5)

        motor_rot = PgaMotor(rotation_axis=[0.0, 0.0, 1.0], angle_rad=math.pi / 2.0)
        assert_vec_close(
            self,
            motor_rot.transform_point([1.0, 0.0, 0.0]),
            [0.0, 1.0, 0.0],
            abs_tol=1e-5,
        )

        m_combined = motor_trans * motor_rot
        inv_m = m_combined.inverse()
        pt_orig = inv_m.transform_point(m_combined.transform_point(pt))
        assert_vec_close(self, pt_orig, pt, abs_tol=1e-5)

    def test_cga_inversion_and_circle(self):
        p1 = CgaPoint(0.0, 0.0, 0.0)
        p2 = CgaPoint(3.0, 4.0, 0.0)
        self.assertTrue(math.isclose(p1.distance(p2), 5.0, abs_tol=1e-5))

        sphere = CgaSphere(0.0, 0.0, 0.0, 2.0)
        self.assertTrue(math.isclose(sphere.radius, 2.0, abs_tol=1e-5))

        p_ext = CgaPoint(4.0, 0.0, 0.0)
        p_inv = sphere.invert_in_sphere(p_ext)
        assert_vec_close(self, p_inv.to_euclidean(), [1.0, 0.0, 0.0], abs_tol=1e-5)

        circle = CgaCircle.from_sphere_plane(sphere, [0.0, 0.0, 1.0], 0.0)
        self.assertTrue(math.isclose(circle.radius, 2.0, abs_tol=1e-5))

        cm = CgaConformalMotor(dilation_factor=2.0)
        p_dilated = cm.transform_point(p2)
        assert_vec_close(self, p_dilated.to_euclidean(), [6.0, 8.0, 0.0], abs_tol=1e-5)

    def test_sta_spinor_and_lounesto_classes(self):
        psi_electron = StaSpinor(
            scalar=1.0,
            ex=0.0,
            ey=0.0,
            ez=0.0,
            bx=0.0,
            by=0.0,
            bz=0.0,
            pseudoscalar=0.0,
        )
        self.assertEqual(psi_electron.lounesto_class(), "Class2_DiracElectron")

        cov = psi_electron.bilinear_covariants()
        self.assertIn("sigma", cov)
        self.assertIn("omega", cov)
        self.assertIn("j", cov)
        self.assertTrue(math.isclose(cov["sigma"], 1.0, abs_tol=1e-5))
        self.assertTrue(math.isclose(cov["omega"], 0.0, abs_tol=1e-5))

        psi_general = StaSpinor(scalar=1.0, pseudoscalar=0.5)
        self.assertEqual(psi_general.lounesto_class(), "Class1_DiracGeneral")

    def test_e8_roots_and_lloyd_max(self):
        roots = get_e8_roots()
        self.assertIsInstance(roots, VersorBufferF32)
        self.assertEqual(roots.shape, (240, 8))
        self.assertEqual(roots.ndim, 2)
        self.assertEqual(roots.nbytes, 240 * 8 * 4)

        rows = roots.tolist()
        self.assertEqual(len(rows), 240)
        for row in rows:
            norm_sq = sum(v * v for v in row)
            self.assertTrue(math.isclose(norm_sq, 2.0, abs_tol=1e-5))

        with memoryview(roots) as view:
            self.assertEqual(view.shape, (240, 8))
            self.assertEqual(view.strides, (32, 4))

        centroids = get_lloyd_max_3bit_centroids()
        self.assertEqual(centroids.shape, (8,))
        c_list = centroids.tolist()
        self.assertLess(c_list[0], c_list[1])
        self.assertLess(c_list[1], c_list[2])
        self.assertLess(c_list[2], c_list[3])
        for i in range(4):
            self.assertTrue(math.isclose(c_list[i], -c_list[i + 4], abs_tol=1e-6))

    def test_clifford_3bit_tensor_matvec_stdlib_2d_buffer(self):
        rng = random.Random(42)
        rows, cols = 128, 128
        w_rows = [[rng.gauss(0.0, 1.0) for _ in range(cols)] for _ in range(rows)]
        x = [rng.gauss(0.0, 1.0) for _ in range(cols)]

        flat_w = array.array("f", (v for row in w_rows for v in row))
        w_view = memoryview(flat_w).cast("B").cast("f", shape=(rows, cols))
        tensor = Clifford3BitTensor.from_buffer(w_view, seed=42)

        self.assertEqual(tensor.rows, rows)
        self.assertEqual(tensor.cols, cols)
        self.assertEqual(tensor.effective_bpw, 4.0)
        self.assertEqual(tensor.bytes_per_row, 64)
        self.assertEqual(len(tensor.to_bytes()), rows * 64)

        y_quant = tensor.matvec(array.array("f", x)).tolist()
        y_fp32 = [sum(row[j] * x[j] for j in range(cols)) for row in w_rows]
        corr = pearson_corr(y_quant, y_fp32)
        self.assertGreater(corr, 0.85, f"Expected correlation > 0.85, got {corr}")

    def test_ganja_schema_payload_zero_deps(self):
        motor = PgaMotor(translation=[1.0, 2.0, 3.0])
        payload = ganja.to_ganja_payload(motor)
        self.assertEqual(payload["algebra"], "Cl*(3,0,1)")
        self.assertEqual(payload["signature"], [3, 0, 1])
        self.assertEqual(len(payload["coefficients"]), 8)

        scene = ganja.GanjaScene(3, 0, 1, title="Test Scene")
        scene.add_element(motor, label="M", color="#ffaa00")
        decoded = json.loads(scene.to_json())
        self.assertEqual(decoded["title"], "Test Scene")
        self.assertEqual(len(decoded["elements"]), 1)


@unittest.skipUnless(HAS_NUMPY, "Optional NumPy interop tests require numpy installed")
class TestOptionalNumpyInterop(unittest.TestCase):
    def test_zero_copy_memoryview_and_array_interface(self):
        roots_buf = get_e8_roots()
        arr = np.asarray(roots_buf)
        self.assertEqual(arr.shape, (240, 8))
        self.assertEqual(arr.dtype, np.float32)
        self.assertTrue(np.shares_memory(arr, memoryview(roots_buf)))
        np.testing.assert_allclose(np.sum(arr ** 2, axis=1), 2.0, atol=1e-5)

        iface = roots_buf.__array_interface__
        self.assertEqual(iface["shape"], (240, 8))
        self.assertEqual(iface["typestr"], "<f4")

        mv = Cl3.from_vector(1.0, 2.0, 3.0)
        mv_arr = np.asarray(mv)
        self.assertEqual(mv_arr.shape, (8,))
        self.assertEqual(mv_arr.dtype, np.float32)
        self.assertTrue(np.shares_memory(mv_arr, memoryview(mv)))
        np.testing.assert_allclose(mv_arr, [0.0, 1.0, 2.0, 0.0, 3.0, 0.0, 0.0, 0.0], atol=1e-6)

        mv2 = Cl3.from_buffer(np.array([1.5, -2.0, 3.0, 0.5, -1.0, 4.0, 0.0, -0.5], dtype=np.float32))
        np.testing.assert_allclose(np.asarray(mv2.to_buffer()), [1.5, -2.0, 3.0, 0.5, -1.0, 4.0, 0.0, -0.5], atol=1e-6)

        d8_buf = vm.project_to_d8(np.array([0.9, 1.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))
        self.assertIsInstance(d8_buf, VersorBufferI32)
        d8_arr = np.asarray(d8_buf)
        self.assertEqual(d8_arr.dtype, np.int32)
        np.testing.assert_array_equal(d8_arr, [1, 1, 0, 0, 0, 0, 0, 0])

    def test_numpy_tensor_matvec_interop(self):
        rng = np.random.default_rng(42)
        w = rng.standard_normal((128, 256), dtype=np.float32)
        x = rng.standard_normal(256, dtype=np.float32)

        tensor = Clifford3BitTensor.from_buffer(w)
        y_quant = np.asarray(tensor.matvec(x))
        self.assertEqual(y_quant.shape, (128,))
        y_fp32 = w @ x
        corr = float(np.corrcoef(y_quant, y_fp32)[0, 1])
        self.assertGreater(corr, 0.85)


if __name__ == "__main__":
    unittest.main()
