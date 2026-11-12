#!/usr/bin/env python3
# ============================================================================
# VERSORC-MATH: INTERACTIVE GANJA.JS VISUALIZATION EXAMPLES
# License: Apache-2.0
# ============================================================================

import os
import sys

# Ensure local python package is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python")))

import versorc_math as vm

def main():
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "visualizations"))
    os.makedirs(out_dir, exist_ok=True)
    print(f"Generating interactive GanJa.js HTML visualizations in: {out_dir}")

    # 1. 3D Projective Geometric Algebra (PGA): Screw Kinematics & Sclerp
    pga_path = os.path.join(out_dir, "01_pga_screw_motion.html")
    print(" -> Rendering 3D PGA Screw Kinematics...")
    vm.ganja.plot_pga_screw_motion(
        translation=[2.0, 1.2, 1.5],
        rotation_axis=[0.0, 0.0, 1.0],
        angle_rad=3.14159,
        save_path=pga_path,
    )
    print(f"    Saved: {pga_path}")

    # 2. 3D Conformal Geometric Algebra (CGA): Spheres, Circles & Inversions
    cga_path = os.path.join(out_dir, "02_cga_conformal_spheres.html")
    print(" -> Rendering 3D CGA Conformal Inversions & Circles...")
    vm.ganja.plot_cga_spheres_and_circles(
        sphere_center=[0.0, 0.0, 0.0],
        sphere_radius=2.0,
        plane_normal=[0.0, 1.0, 0.0],
        plane_dist=0.5,
        save_path=cga_path,
    )
    print(f"    Saved: {cga_path}")

    # 3. 3D Euclidean Clifford Algebra (Cl3): 2-Blade Bivector & Rotor Precession
    cl3_path = os.path.join(out_dir, "03_cl3_bivector_orbit.html")
    print(" -> Rendering 3D Cl(3,0) Bivector 2-Blade & Rotor Orbit...")
    vm.ganja.plot_cl3_bivector_orbit(
        vector_u=[1.8, 0.0, 0.0],
        vector_v=[0.0, 1.8, 0.0],
        save_path=cl3_path,
    )
    print(f"    Saved: {cl3_path}")

    print("\nSUCCESS: All 3 interactive GanJa.js WebGL scenes generated!")
    print("Open any of the generated .html files in any web browser to interact in 3D.")

if __name__ == "__main__":
    main()
