#!/usr/bin/env python3
# ============================================================================
# VERSORC-MATH: GANJA FORMAT INTERCHANGE EXAMPLE (DATA-ONLY, NO HTML/CDN)
# License: Apache-2.0
# ============================================================================

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python")))

import versorc_math as vm


def main() -> None:
    pga_scene = vm.ganja.GanjaScene(p=3, q=0, r=1, title="3D PGA Motor & Point Schema")
    motor = vm.PgaMotor(translation=[2.0, 1.2, 1.5], rotation_axis=[0.0, 0.0, 1.0], angle_rad=3.14159)
    origin = vm.PgaPoint.origin()
    moved = vm.PgaPoint(*motor.transform_point([0.0, 0.0, 0.0]))

    pga_scene.add_element(motor, label="RigidMotor", color="#ffaa00")
    pga_scene.add_element(origin, label="Origin", color="#888888")
    pga_scene.add_element(moved, label="Transformed", color="#44ff44")
    print(pga_scene.to_json())

    cl3_bivector = vm.Cl3.e1() ^ vm.Cl3.e2()
    print(vm.ganja.to_ganja_json(cl3_bivector))


if __name__ == "__main__":
    main()
