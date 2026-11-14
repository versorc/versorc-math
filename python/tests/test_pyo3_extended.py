# ============================================================================
# VERSORC-MATH: EXTENDED PYO3 & VULKAN DISPATCH VALIDATION SUITE
# License: Apache-2.0
# ============================================================================

import array
import math
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import versorc_math as vm
from versorc_math import (
    CayleyKleinSpace,
    Clifford3BitTensor,
    DualQuaternion,
    PgaLine,
    PgaMotor,
    PgaPlane,
    PgaPoint,
    Quaternion,
    VersorBufferF32,
    VersorBufferI32,
    project_to_d8,
    project_to_e8,
)

try:
    import numpy as np  # Optional interop test support only.
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


class TestExtendedPyo3Bindings(unittest.TestCase):
    def test_pga_entities_python(self):
        p1 = PgaPoint(0.0, 0.0, 3.0)
        self.assertEqual(p1.to_tuple(), (0.0, 0.0, 3.0))
        assert_vec_close(self, p1.to_list(), [0.0, 0.0, 3.0], abs_tol=1e-6)
        buf = p1.to_buffer()
        self.assertIsInstance(buf, VersorBufferF32)
        self.assertEqual(buf.shape, (3,))
        assert_vec_close(self, buf.tolist(), [0.0, 0.0, 3.0], abs_tol=1e-6)
        with memoryview(p1) as view:
            self.assertEqual(view.shape, (3,))
            assert_vec_close(self, view.tolist(), [0.0, 0.0, 3.0], abs_tol=1e-6)

        plane = PgaPlane(0.0, 0.0, 1.0, 0.0)
        self.assertTrue(math.isclose(plane.distance_to_point(p1), 3.0, abs_tol=1e-5))

        p_refl = plane.reflect_point(p1)
        self.assertTrue(math.isclose(p_refl.z, -3.0, abs_tol=1e-5))

        p0 = PgaPoint.origin()
        line = PgaLine.from_points(p0, p1)
        p_proj = line.project_point(PgaPoint(2.0, 0.0, 1.5))
        self.assertTrue(math.isclose(p_proj.x, 0.0, abs_tol=1e-5))
        self.assertTrue(math.isclose(p_proj.z, 1.5, abs_tol=1e-5))

    def test_quaternion_and_dual_quaternion_python(self):
        q = Quaternion.from_axis_angle([0.0, 0.0, 1.0], math.pi / 2.0)
        v_rot = q.rotate_vector([1.0, 0.0, 0.0])
        assert_vec_close(self, v_rot, [0.0, 1.0, 0.0], abs_tol=1e-5)

        dq = DualQuaternion.from_rotation_translation(q, [0.0, 5.0, 0.0])
        pt_trans = dq.transform_point([1.0, 0.0, 0.0])
        assert_vec_close(self, pt_trans, [0.0, 6.0, 0.0], abs_tol=1e-5)

        dq_id = DualQuaternion.from_rotation_translation(Quaternion(), [0.0, 0.0, 0.0])
        dq_mid = dq_id.sclerp(dq, 0.5)
        pt_mid = dq_mid.transform_point([1.0, 0.0, 0.0])
        self.assertTrue(
            math.isclose(pt_mid[1], 2.5 + math.sin(math.pi / 4.0), abs_tol=1e-4)
        )

    def test_cayley_klein_space_python(self):
        ck = CayleyKleinSpace.hyperbolic()
        self.assertTrue(math.isclose(ck.tau, -1.0, abs_tol=1e-5))

        p0 = array.array("f", [0.0, 0.0, 0.0])
        p1 = array.array("f", [0.5, 0.0, 0.0])

        dist = ck.distance(p0, p1)
        self.assertTrue(math.isclose(dist, math.log(3.0), rel_tol=1e-4))
        self.assertTrue(math.isclose(ck.conformal_factor(p0), 2.0, abs_tol=1e-5))

    def test_e8_and_d8_lattice_projection_python(self):
        x = array.array("f", [0.9, 1.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        proj_e8 = project_to_e8(x)
        proj_d8 = project_to_d8(x)

        self.assertIsInstance(proj_e8, VersorBufferF32)
        self.assertIsInstance(proj_d8, VersorBufferI32)
        self.assertEqual(len(proj_d8), 8)
        self.assertEqual(proj_d8[0], 1)
        self.assertEqual(proj_d8[1], 1)
        assert_vec_close(self, proj_e8.tolist(), [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], abs_tol=1e-5)
        self.assertEqual(proj_d8.tolist(), [1, 1, 0, 0, 0, 0, 0, 0])

    def test_clifford_3bit_effective_bpw_and_validation(self):
        rows, cols = 32, 256
        rng = random.Random(7)
        weights = [[rng.gauss(0.0, 1.0) for _ in range(cols)] for _ in range(rows)]
        tensor = Clifford3BitTensor.from_buffer(weights)
        self.assertTrue(math.isclose(tensor.effective_bpw, 4.0, abs_tol=1e-5))
        self.assertEqual(tensor.bytes_per_row, 128)
        self.assertEqual(len(tensor.to_bytes()), rows * 128)

        with self.assertRaises(ValueError):
            Clifford3BitTensor.from_buffer([[1.0] * 64])


class TestVulkanGpuComputeContextPython(unittest.TestCase):
    def setUp(self):
        if not hasattr(vm, "GpuComputeContext"):
            self.skipTest("Built without vulkan feature")
        try:
            self.devices = vm.GpuComputeContext.probe_devices()
            self.ctx = vm.GpuComputeContext.init()
        except Exception as exc:
            self.skipTest(f"Vulkan physical GPU unavailable: {exc}")

    def test_probe_devices_and_core_shaders(self):
        self.assertIsInstance(self.devices, list)
        self.assertGreater(len(self.devices), 0)
        self.assertIn("name", self.devices[0])
        self.assertIn("is_discrete", self.devices[0])

        a = array.array("f", [1.0, 2.0, 3.0, 4.0])
        b = array.array("f", [10.0, 20.0, 30.0, 40.0])
        c = self.ctx.dispatch_vec_add(a, b)
        assert_vec_close(self, c.tolist(), [11.0, 22.0, 33.0, 44.0], abs_tol=1e-5)

        gate = [0.0, 2.0, -2.0, 10.0]
        up = [5.0, 1.0, 1.0, 2.0]
        swiglu = self.ctx.dispatch_silu_mul(gate, up).tolist()
        expected_swiglu = [(g / (1.0 + math.exp(-g))) * u for g, u in zip(gate, up)]
        assert_vec_close(self, swiglu, expected_swiglu, abs_tol=1e-5)

        rng = random.Random(11)
        x = [rng.gauss(0.0, 1.0) for _ in range(512)]
        gamma = [1.0] * 512
        normed = self.ctx.dispatch_rmsnorm(x, gamma, 1e-5).tolist()
        rms = math.sqrt(sum(v * v for v in x) / len(x) + 1e-5)
        expected_normed = [v / rms for v in x]
        assert_vec_close(self, normed, expected_normed, abs_tol=1e-4)

        q = [rng.gauss(0.0, 1.0) for _ in range(4 * 64)]
        rope_out = self.ctx.dispatch_rope(q, 10, 4, 64, 10000.0, True)
        self.assertEqual(rope_out.shape, (4 * 64,))
        self.assertTrue(any(abs(u - v) > 1e-4 for u, v in zip(q, rope_out.tolist())))

    def test_ga_and_hyperbolic_vulkan_dispatchers(self):
        cl = self.ctx.dispatch_cl3_batch(
            [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            op_code=0,
        )
        self.assertEqual(cl.shape, (1, 8))
        assert_vec_close(self, cl.tolist()[0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], abs_tol=1e-5)

        pb = self.ctx.dispatch_poincare_batch(
            [[0.0, 0.0, 0.0]],
            [[0.5, 0.0, 0.0]],
            op_code=0,
        )
        self.assertEqual(pb.shape, (1, 3))
        assert_vec_close(self, pb.tolist()[0], [0.5, 0.0, 0.0], abs_tol=1e-5)

        pd = self.ctx.dispatch_poincare_distance([[0.0, 0.0, 0.0]], [[0.6, 0.0, 0.0]])
        self.assertEqual(pd.shape, (1,))
        self.assertTrue(math.isclose(pd[0], math.log(4.0), abs_tol=1e-4))

        motor = PgaMotor(translation=[1.0, 2.0, 3.0])
        pg = self.ctx.dispatch_pga_transform([[1.0, 0.0, 0.0]], motor)
        self.assertEqual(pg.shape, (1, 3))
        assert_vec_close(self, pg.tolist()[0], [2.0, 2.0, 3.0], abs_tol=1e-5)

        cg = self.ctx.dispatch_cga_transform([[1.0, 2.0, 3.0]], op_type=1, params=[2.0, 0.0, 0.0])
        self.assertEqual(cg.shape, (1, 3))
        assert_vec_close(self, cg.tolist()[0], [2.0, 4.0, 6.0], abs_tol=1e-5)

    @unittest.skipUnless(HAS_NUMPY, "Optional NumPy GPU buffer test requires numpy installed")
    def test_vulkan_numpy_buffer_interop(self):
        a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        b = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float32)
        out = self.ctx.dispatch_vec_add(a, b)
        out_arr = np.asarray(out)
        self.assertTrue(np.shares_memory(out_arr, memoryview(out)))
        np.testing.assert_allclose(out_arr, [1.5, 3.5, 5.5, 7.5], atol=1e-5)


if __name__ == "__main__":
    unittest.main()
