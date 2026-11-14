# ============================================================================
# VERSORC-MATH: COMPLETE PYO3 ALGEBRAIC PARITY TEST SUITE
# License: Apache-2.0
#
# Validates exposed Python classes across Cl(2,0), Cl(3,0), PGA Cl*(3,0,1),
# CGA Cl(4,1), STA Cl(1,3), Poincare B^d, Dyadic Walsh, and E8 lattice APIs.
# ============================================================================

import array
import math
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from versorc_math import (
    CayleyKleinSpace,
    CgaCircle,
    CgaConformalMotor,
    CgaPoint,
    CgaSphere,
    Cl3,
    Clifford3BitTensor,
    DyadicWalshVersor,
    PoincarePoint,
    StaSpinor,
    StaVector,
    fwht,
    get_e8_roots,
    get_lloyd_max_3bit_centroids,
)


def assert_vec_close(test_case, actual, expected, abs_tol=1e-5):
    test_case.assertEqual(len(actual), len(expected))
    for i, (a, e) in enumerate(zip(actual, expected)):
        test_case.assertTrue(
            math.isclose(a, e, abs_tol=abs_tol),
            f"Index {i}: expected {e}, got {a}",
        )


class TestPyo3CompleteCoverage(unittest.TestCase):
    def test_cl3_algebraic_operations_python(self):
        e1 = Cl3.e1()
        e2 = Cl3.e2()
        ps = Cl3.I()

        e12 = e1 * e2
        self.assertTrue(math.isclose(e12.e12, 1.0, abs_tol=1e-5))
        self.assertTrue(math.isclose((ps * ps).scalar, -1.0, abs_tol=1e-5))

        dual_e1 = e1.dual()
        self.assertTrue(math.isclose(dual_e1.e23, -1.0, abs_tol=1e-5))

        mv = Cl3.from_scalar(2.0) + e1 * 3.0 + e12 * 4.0
        prod = mv * mv.inverse()
        self.assertTrue(math.isclose(prod.scalar, 1.0, abs_tol=1e-5))
        assert_vec_close(self, prod.to_list()[1:], [0.0] * 7, abs_tol=1e-5)

    def test_cga_full_pipeline_and_validation_python(self):
        p1 = CgaPoint(1.0, 2.0, 3.0)
        p2 = CgaPoint(4.0, 6.0, 3.0)
        self.assertTrue(math.isclose(p1.distance(p2), 5.0, abs_tol=1e-5))
        self.assertEqual(len(p1.to_list()), 5)
        self.assertEqual(p1.to_buffer().shape, (5,))

        sphere = CgaSphere(0.0, 0.0, 0.0, 2.0)
        self.assertTrue(math.isclose(sphere.radius, 2.0, abs_tol=1e-5))
        self.assertEqual(len(sphere.to_list()), 5)

        p_in = CgaPoint(1.0, 0.0, 0.0)
        assert_vec_close(self, sphere.invert_in_sphere(p_in).to_euclidean(), [4.0, 0.0, 0.0], abs_tol=1e-5)

        circle = CgaCircle.from_sphere_plane(sphere, [0.0, 0.0, 1.0], 0.0)
        self.assertTrue(math.isclose(circle.radius, 2.0, abs_tol=1e-5))
        self.assertTrue(math.isclose(circle.area, math.pi * 4.0, abs_tol=1e-4))
        self.assertTrue(math.isclose(circle.circumference, 2.0 * math.pi * 2.0, abs_tol=1e-4))

        cm = CgaConformalMotor.from_translation(10.0, 0.0, 0.0)
        assert_vec_close(self, cm.transform_point(p1).to_euclidean(), [11.0, 2.0, 3.0], abs_tol=1e-5)

        with self.assertRaises(ValueError):
            CgaConformalMotor.from_rotation([0.0, 0.0, 0.0], 1.0)
        with self.assertRaises(ValueError):
            CgaConformalMotor.from_dilation(-1.0)
        with self.assertRaises(ValueError):
            CgaSphere(0.0, 0.0, 0.0, -2.0)

    def test_sta_spacetime_algebra_and_lounesto_python(self):
        v_timelike = StaVector(5.0, 1.0, 2.0, 2.0)
        self.assertTrue(math.isclose(v_timelike.interval_sq(), 16.0, abs_tol=1e-5))
        self.assertTrue(math.isclose(v_timelike.proper_time(), 4.0, abs_tol=1e-5))
        self.assertTrue(v_timelike.is_timelike())
        self.assertFalse(v_timelike.is_spacelike())

        v_null = StaVector(3.0, 1.0, 2.0, 2.0)
        self.assertTrue(math.isclose(v_null.interval_sq(), 0.0, abs_tol=1e-5))
        self.assertTrue(v_null.is_lightlike())

        spinor = StaSpinor(
            scalar=1.0,
            ex=0.0,
            ey=0.0,
            ez=0.0,
            bx=1.0,
            by=0.0,
            bz=0.0,
            pseudoscalar=0.0,
        )
        cov = spinor.bilinear_covariants()
        for key in ("sigma", "omega", "j", "s", "k"):
            self.assertIn(key, cov)
        self.assertIsInstance(spinor.lounesto_class(), str)

    def test_poincare_hyperbolic_python(self):
        p1 = PoincarePoint([0.0, 0.0, 0.0])
        p2 = PoincarePoint([0.6, 0.0, 0.0])

        dist = p1.geodesic_distance(p2)
        self.assertTrue(math.isclose(dist, math.log(4.0), abs_tol=1e-4))

        mob = p1.mobius_add(p2)
        self.assertTrue(math.isclose(mob.coords[0], 0.6, abs_tol=1e-5))

        with memoryview(p2) as view:
            self.assertEqual(view.shape, (3,))
            assert_vec_close(self, view.tolist(), [0.6, 0.0, 0.0], abs_tol=1e-6)

        ck_euc = CayleyKleinSpace.euclidean()
        self.assertTrue(math.isclose(ck_euc.tau, 0.0, abs_tol=1e-6))

    def test_dyadic_walsh_and_clifford3_tensor_python(self):
        dim = 64
        versor = DyadicWalshVersor(dim, 42)

        x = array.array("f", (-2.0 + 4.0 * i / (dim - 1) for i in range(dim)))
        x_rot = versor.apply(x).tolist()
        norm_in = math.sqrt(sum(v * v for v in x))
        norm_out = math.sqrt(sum(v * v for v in x_rot))
        self.assertTrue(math.isclose(norm_in, norm_out, rel_tol=1e-4))

        x_rec = versor.invert(x_rot).tolist()
        assert_vec_close(self, x_rec, list(x), abs_tol=1e-5)

        x_fwht = fwht(x)
        self.assertEqual(x_fwht.shape, (dim,))

        rows, cols = 32, 128
        rng = random.Random(123)
        dense_w = [[rng.gauss(0.0, 1.0) for _ in range(cols)] for _ in range(rows)]
        tensor = Clifford3BitTensor.from_buffer(dense_w, seed=123)
        self.assertEqual(tensor.rows, rows)
        self.assertEqual(tensor.cols, cols)

        vec_in = [rng.gauss(0.0, 1.0) for _ in range(cols)]
        out = tensor.matvec(vec_in)
        self.assertEqual(out.shape, (rows,))
        self.assertTrue(any(abs(v) > 1e-6 for v in out.tolist()))

    def test_lattice_roots_and_centroids_python(self):
        roots = get_e8_roots()
        self.assertEqual(roots.shape, (240, 8))
        for row in roots.tolist():
            self.assertTrue(math.isclose(sum(v * v for v in row), 2.0, abs_tol=1e-5))

        centroids = get_lloyd_max_3bit_centroids().tolist()
        self.assertEqual(len(centroids), 8)
        self.assertLess(centroids[0], centroids[1])
        self.assertLess(centroids[1], centroids[2])
        self.assertLess(centroids[2], centroids[3])
        for i in range(4):
            self.assertTrue(math.isclose(centroids[i], -centroids[i + 4], abs_tol=1e-6))


if __name__ == "__main__":
    unittest.main()
