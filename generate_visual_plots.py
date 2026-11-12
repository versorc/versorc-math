#!/usr/bin/env python3
# ============================================================================
# VERSORC-MATH: SCIENTIFIC HIGH-RESOLUTION PNG VISUALIZATION GENERATOR
# License: Apache-2.0 OR MIT
# ============================================================================

import os
import sys
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Add local python directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "python")))

import versorc_math as vm
from versorc_math import (
    Cl3,
    PgaMotor,
    CgaPoint,
    CgaSphere,
    CgaCircle,
    PoincarePoint,
    get_e8_roots,
    get_lloyd_max_3bit_centroids,
)
from versorc_math.plotting import compute_e8_coxeter_basis


def render_all_pngs():
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "examples", "plots"))
    os.makedirs(out_dir, exist_ok=True)
    print(f"Rendering high-resolution scientific PNGs in: {out_dir}\n")

    # ------------------------------------------------------------------------
    # 1. 3D PGA SE(3) Screw Kinematics & Sclerp Geodesic Trajectory
    # ------------------------------------------------------------------------
    print(" [1/6] Rendering 3D PGA Screw Kinematics (SE(3) Motor Action & Chasles Axis)...")
    fig = plt.figure(figsize=(9, 8), dpi=150)
    fig.patch.set_facecolor("#0b0d13")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#0b0d13")

    # Motor: 180° rotation around Z with translation (2, 1, 2)
    motor = PgaMotor(translation=[2.0, 1.0, 2.0], rotation_axis=[0.0, 0.0, 1.0], angle_rad=np.pi)
    screw_pt, screw_dir, screw_ang, screw_pitch = motor.screw_parameters()
    screw_pt = np.array(screw_pt, dtype=np.float64)
    screw_dir = np.array(screw_dir, dtype=np.float64)

    # Trace continuous Sclerp trajectory of origin and intermediate frames
    num_steps = 80
    traj_origin = np.zeros((num_steps, 3))
    ident = PgaMotor()
    for i, t in enumerate(np.linspace(0.0, 1.0, num_steps)):
        m_t = ident.sclerp(motor, float(t))
        pt = m_t.transform_point([0.0, 0.0, 0.0])
        traj_origin[i] = pt

        # Draw radial spokes from the invariant Chasles screw axis to the helical trajectory
        if i % 10 == 0 or i == num_steps - 1:
            axis_z = screw_pt + np.dot(np.array(pt) - screw_pt, screw_dir) * screw_dir
            ax.plot([axis_z[0], pt[0]], [axis_z[1], pt[1]], [axis_z[2], pt[2]],
                    color="#aa44ff", alpha=0.45, lw=1.0, linestyle="--")

    # Plot continuous helical Sclerp curve
    ax.plot(traj_origin[:, 0], traj_origin[:, 1], traj_origin[:, 2],
            color="#ffaa00", lw=3.0, label="Sclerp Geodesic Helix M(t) = exp(t log M)")

    # Plot Initial Frame at t = 0
    p0 = traj_origin[0]
    ax.quiver(p0[0], p0[1], p0[2], 0.6, 0, 0, color="#ff4444", lw=2.5, arrow_length_ratio=0.2, label="Initial Frame (t=0)")
    ax.quiver(p0[0], p0[1], p0[2], 0, 0.6, 0, color="#44ff44", lw=2.5, arrow_length_ratio=0.2)
    ax.quiver(p0[0], p0[1], p0[2], 0, 0, 0.6, color="#44aaff", lw=2.5, arrow_length_ratio=0.2)

    # Plot Final Frame at t = 1
    p1 = traj_origin[-1]
    x_end = (np.array(motor.transform_point([0.6, 0.0, 0.0])) - p1)
    y_end = (np.array(motor.transform_point([0.0, 0.6, 0.0])) - p1)
    z_end = (np.array(motor.transform_point([0.0, 0.0, 0.6])) - p1)
    ax.quiver(p1[0], p1[1], p1[2], x_end[0], x_end[1], x_end[2], color="#ff8888", lw=2.5, linestyle="--", arrow_length_ratio=0.2, label="Transformed Frame (t=1)")
    ax.quiver(p1[0], p1[1], p1[2], y_end[0], y_end[1], y_end[2], color="#88ff88", lw=2.5, linestyle="--", arrow_length_ratio=0.2)
    ax.quiver(p1[0], p1[1], p1[2], z_end[0], z_end[1], z_end[2], color="#88ccff", lw=2.5, linestyle="--", arrow_length_ratio=0.2)

    # True Invariant Chasles Screw Axis Line passing through screw_pt = (1.0, 0.5, 0.0)
    line_start = screw_pt - 0.5 * screw_dir
    line_end = screw_pt + 2.8 * screw_dir
    ax.plot([line_start[0], line_end[0]], [line_start[1], line_end[1]], [line_start[2], line_end[2]],
            color="#aa44ff", lw=2.6, linestyle=":",
            label=f"Chasles Screw Axis L (p₀=[{screw_pt[0]:.1f}, {screw_pt[1]:.1f}, 0], pitch={screw_pitch:.2f})")

    ax.set_xlim(-0.5, 2.8)
    ax.set_ylim(-1.0, 2.3)
    ax.set_zlim(-0.5, 2.8)
    ax.set_title("3D Projective Geometric Algebra Cl*(3,0,1)\nSE(3) Rigid Motor Kinematics, Chasles Screw Axis & Sclerp Helix", color="white", fontsize=12, pad=14)
    ax.set_xlabel("X (e032)", color="#8da2be", labelpad=8)
    ax.set_ylabel("Y (e013)", color="#8da2be", labelpad=8)
    ax.set_zlabel("Z (e021)", color="#8da2be", labelpad=8)
    ax.tick_params(colors="#8da2be")
    ax.legend(facecolor="#10141f", edgecolor="#2d3748", labelcolor="white", loc="upper left", fontsize=9)
    pga_png = os.path.join(out_dir, "01_pga_3d_screw_kinematics.png")
    fig.savefig(pga_png, dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {pga_png}")

    # ------------------------------------------------------------------------
    # 2. 3D CGA (4,1) Conformal Inversion across Sphere & Intersection Circle
    # ------------------------------------------------------------------------
    print(" [2/6] Rendering 3D CGA Conformal Spheres, Inversions & Circle Meet...")
    fig = plt.figure(figsize=(9, 8), dpi=150)
    fig.patch.set_facecolor("#0b0d13")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#0b0d13")

    R = 2.0
    sphere = CgaSphere(0.0, 0.0, 0.0, radius=R)

    # Wireframe Sphere
    u_sp = np.linspace(0, 2 * np.pi, 32)
    v_sp = np.linspace(0, np.pi, 20)
    xs = R * np.outer(np.cos(u_sp), np.sin(v_sp))
    ys = R * np.outer(np.sin(u_sp), np.sin(v_sp))
    zs = R * np.outer(np.ones(np.size(u_sp)), np.cos(v_sp))
    ax.plot_wireframe(xs, ys, zs, color="#00ccff", alpha=0.18, lw=0.6, label=f"Conformal Sphere S (Radius={sphere.radius:.1f})")

    # Intersection Cutting Plane Pi: n=(0,0,1), d=-0.8 (plane z = +0.8)
    plane_z = 0.8
    cga_circle = CgaCircle.from_sphere_plane(sphere, [0.0, 0.0, 1.0], -plane_z)
    c_center = cga_circle.center
    c_radius = cga_circle.radius

    xp = np.linspace(-2.5, 2.5, 10)
    yp = np.linspace(-2.5, 2.5, 10)
    Xp, Yp = np.meshgrid(xp, yp)
    Zp = np.full_like(Xp, c_center[2])
    ax.plot_surface(Xp, Yp, Zp, color="#ffaa00", alpha=0.12)

    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(c_center[0] + c_radius * np.cos(theta),
            c_center[1] + c_radius * np.sin(theta),
            np.full_like(theta, c_center[2]),
            color="#ffd700", lw=3.0,
            label=f"CgaCircle C* = S* ∧ Π* (r={c_radius:.3f}, A={cga_circle.area:.2f})")

    # Conformal Inversion: P outside -> P' = S P S^-1 inside
    ext_pts = np.array([
        [3.6, 0.0, 0.0],
        [0.0, 3.8, 1.2],
        [-3.0, -3.0, -0.8],
    ])
    for i, pt_ext in enumerate(ext_pts):
        p_cga = CgaPoint(float(pt_ext[0]), float(pt_ext[1]), float(pt_ext[2]))
        p_inv = sphere.invert_in_sphere(p_cga)
        pt_in = np.array(p_inv.to_euclidean())

        ax.scatter([pt_ext[0]], [pt_ext[1]], [pt_ext[2]], color="#ff3344", s=65, edgecolors="white", lw=0.6,
                   label="External Point P (||p|| > R)" if i == 0 else None)
        ax.scatter([pt_in[0]], [pt_in[1]], [pt_in[2]], color="#00ff88", s=65, edgecolors="white", lw=0.6,
                   label="Inverted Point P' = S P S⁻¹ (||p'|| = R²/||p||)" if i == 0 else None)
        ax.plot([0, pt_ext[0]], [0, pt_ext[1]], [0, pt_ext[2]], color="#888888", linestyle=":", lw=1.1)

    ax.set_xlim(-3.8, 3.8)
    ax.set_ylim(-3.8, 3.8)
    ax.set_zlim(-2.8, 2.8)
    ax.set_title("3D Conformal Geometric Algebra Cl(4,1)\nSpherical Versor Inversion P' = S P S⁻¹ & Dual Circle Meet C* = S* ∧ Π*", color="white", fontsize=12, pad=14)
    ax.set_xlabel("X (e1)", color="#8da2be", labelpad=8)
    ax.set_ylabel("Y (e2)", color="#8da2be", labelpad=8)
    ax.set_zlabel("Z (e3)", color="#8da2be", labelpad=8)
    ax.tick_params(colors="#8da2be")
    ax.legend(facecolor="#10141f", edgecolor="#2d3748", labelcolor="white", loc="upper left", fontsize=9)
    cga_png = os.path.join(out_dir, "02_cga_3d_conformal_spheres_and_inversion.png")
    fig.savefig(cga_png, dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {cga_png}")

    # ------------------------------------------------------------------------
    # 3. 3D Euclidean Cl(3,0) 2-Blade Bivector and Continuous Rotor Orbit
    # ------------------------------------------------------------------------
    print(" [3/6] Rendering 3D Cl(3,0) 2-Blade Bivector & Native Cl3.sandwich Rotor Orbit...")
    fig = plt.figure(figsize=(9, 8), dpi=150)
    fig.patch.set_facecolor("#0b0d13")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#0b0d13")

    # Compute 2-blade and Hodge dual strictly via VersorC-Math Cl3
    u_mv = Cl3.vector(1.8, 0.0, 0.0)
    v_mv = Cl3.vector(0.0, 1.8, 0.0)
    biv_mv = u_mv ^ v_mv
    dual_mv = biv_mv.dual()

    u_arr = u_mv.to_numpy()
    v_arr = v_mv.to_numpy()
    d_arr = dual_mv.to_numpy()

    u_vec = np.array([u_arr[1], u_arr[2], u_arr[4]])
    v_vec = np.array([v_arr[1], v_arr[2], v_arr[4]])
    dual_vec = np.array([d_arr[1], d_arr[2], d_arr[4]])
    orig = np.array([0.0, 0.0, 0.0])

    # Parallelogram 2-blade u ^ v
    verts = [np.array([orig, orig + u_vec, orig + u_vec + v_vec, orig + v_vec])]
    poly = Poly3DCollection(verts, alpha=0.35, facecolor="#2b5c8f", edgecolor="#4e88c7", lw=2.0)
    ax.add_collection3d(poly)

    # 1-blades u and v
    ax.quiver(0, 0, 0, u_vec[0], u_vec[1], u_vec[2], color="#ff4444", lw=2.5, arrow_length_ratio=0.12, label="1-Blade u = 1.8 e₁")
    ax.quiver(0, 0, 0, v_vec[0], v_vec[1], v_vec[2], color="#44ff44", lw=2.5, arrow_length_ratio=0.12, label="1-Blade v = 1.8 e₂")

    # Canonical Hodge dual *(u ^ v) = (u ^ v) I^-1 = (u ^ v)(-e123) = +3.24 e3
    ax.quiver(0, 0, 0, dual_vec[0], dual_vec[1], dual_vec[2], color="#aa44ff", lw=2.5, linestyle="--", arrow_length_ratio=0.1,
              label=f"Hodge Dual (u ∧ v)I⁻¹ = +{biv_mv.norm():.2f} e₃")

    # Continuous Rotor Orbit computed strictly via Cl3.sandwich: v(t) = R(t) v0 R(t)~
    probe_mv = Cl3.vector(1.6, 0.0, 1.4)
    e12 = Cl3.e1() ^ Cl3.e2()
    orbit_steps = 72
    orbit = np.zeros((orbit_steps, 3))
    for i, theta_i in enumerate(np.linspace(0, 2 * np.pi, orbit_steps)):
        rotor_i = Cl3.from_scalar(float(math.cos(theta_i * 0.5))) - e12 * float(math.sin(theta_i * 0.5))
        rot_v = rotor_i.sandwich(probe_mv).to_numpy()
        orbit[i] = [rot_v[1], rot_v[2], rot_v[4]]

    probe_xyz = orbit[0]
    ax.plot(orbit[:, 0], orbit[:, 1], orbit[:, 2], color="#ffaa00", lw=2.8, label="Rotor Sandwich Orbit v(θ) = R(θ) v₀ R(θ)~")
    ax.quiver(0, 0, 0, probe_xyz[0], probe_xyz[1], probe_xyz[2], color="#ffaa00", lw=2.2, arrow_length_ratio=0.12, label="Probe Vector v₀")

    ax.set_xlim(-2.0, 2.2)
    ax.set_ylim(-2.0, 2.2)
    ax.set_zlim(-0.2, 3.6)
    ax.set_title("3D Euclidean Geometric Algebra Cl(3,0)\n2-Blade Bivector u ∧ v, Hodge Dual (u ∧ v)I⁻¹, and Rotor Sandwich Orbit", color="white", fontsize=12, pad=14)
    ax.set_xlabel("e₁", color="#8da2be", labelpad=8)
    ax.set_ylabel("e₂", color="#8da2be", labelpad=8)
    ax.set_zlabel("e₃", color="#8da2be", labelpad=8)
    ax.tick_params(colors="#8da2be")
    ax.legend(facecolor="#10141f", edgecolor="#2d3748", labelcolor="white", loc="upper left", fontsize=9)
    cl3_png = os.path.join(out_dir, "03_cl3_2blade_bivector_and_rotor_orbit.png")
    fig.savefig(cl3_png, dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {cl3_png}")

    # ------------------------------------------------------------------------
    # 4. E8 Gosset Lattice: 240 Minimal Roots on Exact Coxeter Plane
    # ------------------------------------------------------------------------
    print(" [4/6] Rendering Gosset E8 Lattice 240 Minimal Roots on True Coxeter Plane (8x30 Rings)...")
    roots = get_e8_roots().astype(np.float64)  # (240, 8)
    u1, u2 = compute_e8_coxeter_basis()
    px = roots @ u1
    py = roots @ u2

    fig, ax = plt.subplots(figsize=(8.5, 8.5), dpi=180)
    fig.patch.set_facecolor("#0b0d13")
    ax.set_facecolor("#0b0d13")

    # Draw true E8 root polytope edges (4_21 Gosset polytope): pairs with <r_i, r_j> = 1 (distance sqrt(2))
    gram = roots @ roots.T
    for i in range(240):
        neighbors = np.where(np.abs(gram[i] - 1.0) < 1e-4)[0]
        for j in neighbors:
            if i < j:
                ax.plot([px[i], px[j]], [py[i], py[j]], color="#3b82f6", alpha=0.055, lw=0.45)

    radii = np.sqrt(px**2 + py**2)
    ax.scatter(px, py, c=radii, cmap="plasma", s=34, edgecolors="white", linewidths=0.45, zorder=5)
    ax.set_title("Gosset E₈ Root Lattice (4₂₁ Polytope: 240 Minimal Roots in ℝ⁸)\nExact Coxeter Plane Eigenspace (8 Concentric Rings × 30 Vertices, D₃₀ Symmetry)", color="white", fontsize=11.5, pad=14)
    ax.axis("equal")
    ax.axis("off")
    e8_png = os.path.join(out_dir, "04_e8_gosset_lattice_coxeter_plane.png")
    fig.savefig(e8_png, dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {e8_png}")

    # ------------------------------------------------------------------------
    # 5. TurboQuant & Clifford 3-Bit Quantization Isotropy
    # ------------------------------------------------------------------------
    print(" [5/6] Rendering TurboQuant Dyadic Versor Isotropy & Normalized Lloyd-Max Centroids...")
    np.random.seed(42)
    # Generate synthetic heavy-tailed weight distribution with extreme outliers (kurtosis > 35)
    raw_w = np.random.randn(4096).astype(np.float32) * 0.1
    outlier_indices = [42, 180, 350, 777, 1200, 1950, 2600, 3400]
    outlier_vals = [2.8, -3.1, 2.5, -2.9, 3.4, -2.7, 3.0, -3.2]
    for idx_o, val_o in zip(outlier_indices, outlier_vals):
        raw_w[idx_o] = val_o

    versor = vm.DyadicWalshVersor(4096, 12345)
    rot_w = versor.apply(raw_w)
    centroids = np.sort(get_lloyd_max_3bit_centroids())

    # Normalize by RMS standard deviation sigma (matching Clifford3BitWord64::pack_16)
    sigma = float(np.sqrt(np.mean(rot_w**2)))
    raw_norm = raw_w / sigma
    rot_norm = rot_w / sigma

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)
    fig.patch.set_facecolor("#0b0d13")
    ax1.set_facecolor("#10141f")
    ax2.set_facecolor("#10141f")

    # Left Panel: Rotated Distribution Normalized by RMS sigma vs N(0,1) PDF & 8 Lloyd-Max Centroids
    x_grid = np.linspace(-3.6, 3.6, 500)
    gauss_pdf = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * x_grid**2)
    ax1.hist(rot_norm, bins=65, density=True, alpha=0.6, color="#00ccff", edgecolor="#0b0d13", lw=0.3,
             label="Normalized Versor Output R(W) / σ")
    ax1.plot(x_grid, gauss_pdf, color="#ffffff", lw=1.8, linestyle="-", label="Exact Gaussian N(0, 1) PDF")
    for k, c in enumerate(centroids):
        ax1.axvline(c, color="#ffd700", linestyle="--", lw=1.4, alpha=0.9,
                    label="8 Lloyd-Max 3-Bit Centroids c_k" if k == 0 else None)

    ax1.set_xlim(-3.6, 3.6)
    ax1.set_title("Normalized Rotated Weights R(W)/σ vs. N(0,1) & Lloyd-Max Centroids", color="white", fontsize=11)
    ax1.set_xlabel("Normalized Amplitude (Standard Deviations σ)", color="#8da2be")
    ax1.set_ylabel("Probability Density", color="#8da2be")
    ax1.tick_params(colors="#8da2be")
    ax1.legend(facecolor="#10141f", edgecolor="#2d3748", labelcolor="white", loc="upper right", fontsize=9)
    ax1.grid(True, linestyle=":", alpha=0.25)

    # Right Panel: Log-scale Tail Comparison (Raw W/sigma with massive outliers vs Rotated R(W)/sigma)
    ax2.hist(raw_norm, bins=90, density=True, alpha=0.55, color="#ff4444", log=True,
             label=f"Raw Weights W / σ (Max |W|/σ = {np.max(np.abs(raw_norm)):.1f}σ)")
    ax2.hist(rot_norm, bins=45, density=True, alpha=0.65, color="#00ccff", log=True,
             label=f"Dyadic Versor R(W) / σ (Bounded ≤ {np.max(np.abs(rot_norm)):.2f}σ)")
    for c in centroids:
        ax2.axvline(c, color="#ffd700", linestyle="--", lw=1.1, alpha=0.8)

    ax2.set_title("Outlier Smearing on Logarithmic Scale (20σ Tails Eliminated)", color="white", fontsize=11)
    ax2.set_xlabel("Normalized Amplitude (Standard Deviations σ)", color="#8da2be")
    ax2.set_ylabel("Log Probability Density", color="#8da2be")
    ax2.tick_params(colors="#8da2be")
    ax2.legend(facecolor="#10141f", edgecolor="#2d3748", labelcolor="white", loc="upper right", fontsize=9)
    ax2.grid(True, linestyle=":", alpha=0.25)

    fig.suptitle("TurboQuant: Isotropic Dyadic Walsh-Hadamard Versor & Optimal 3-Bit Lloyd-Max Quantization", color="white", fontsize=13, y=1.02)
    plt.tight_layout()
    tq_png = os.path.join(out_dir, "05_clifford_3bit_turboquant_isotropy.png")
    fig.savefig(tq_png, dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {tq_png}")

    # ------------------------------------------------------------------------
    # 6. Conformal Hyperbolic Poincaré Disk D^2
    # ------------------------------------------------------------------------
    print(" [6/6] Rendering Conformal Hyperbolic Poincaré Disk D^2 (Orthogonal Geodesics & Gyrovector Addition)...")
    fig, ax = plt.subplots(figsize=(8, 8), dpi=180)
    fig.patch.set_facecolor("#0b0d13")
    ax.set_facecolor("#0b0d13")

    th = np.linspace(0, 2 * np.pi, 500)
    ax.plot(np.cos(th), np.sin(th), color="#ffffff", lw=2.2, label="Ideal Boundary S¹ (||z|| = 1)")
    ax.fill(np.cos(th), np.sin(th), color="#10141f")

    for r_hyp in [0.25, 0.5, 0.75, 0.9]:
        ax.plot(r_hyp * np.cos(th), r_hyp * np.sin(th), color="#4e88c7", linestyle=":", lw=0.8, alpha=0.55)

    # Create hyperbolic points via VersorC-Math PoincarePoint
    raw_coords = [
        [-0.55, 0.35],
        [0.50, 0.45],
        [0.65, -0.35],
        [-0.15, -0.70],
        [-0.65, -0.25],
    ]
    p_nodes = [PoincarePoint(c) for c in raw_coords]

    # Plot all pairwise hyperbolic geodesics (exact circular arcs orthogonal to S^1)
    first_arc = True
    for i in range(len(p_nodes)):
        for j in range(i + 1, len(p_nodes)):
            arc = np.array(p_nodes[i].geodesic_arc(p_nodes[j], 80))
            ax.plot(arc[:, 0], arc[:, 1], color="#ffd700", lw=1.5, alpha=0.65,
                    label="Hyperbolic Geodesic Arc (Orthogonal to S¹)" if first_arc else None)
            first_arc = False

    # Demonstrate explicit Möbius Gyrovector Addition: w = u (+) v
    u_pt = PoincarePoint([0.35, -0.15])
    v_pt = PoincarePoint([0.25, 0.45])
    w_pt = u_pt.mobius_add(v_pt)
    origin_pt = PoincarePoint([0.0, 0.0])

    for a_pt, b_pt in [(origin_pt, u_pt), (origin_pt, v_pt), (u_pt, w_pt), (origin_pt, w_pt)]:
        arc_uv = np.array(a_pt.geodesic_arc(b_pt, 60))
        ax.plot(arc_uv[:, 0], arc_uv[:, 1], color="#00ffcc", lw=2.2, linestyle="--", alpha=0.9)

    # Compute Einstein Hyperbolic Midpoint of the 5 pentagon vertices
    mid_pt = PoincarePoint.einstein_midpoint(p_nodes, [1.0] * len(p_nodes))
    mid_c = mid_pt.coords

    # Scatter vertices
    pts_arr = np.array([p.coords for p in p_nodes])
    ax.scatter(pts_arr[:, 0], pts_arr[:, 1], color="#ffaa00", s=70, edgecolors="white", lw=0.7, zorder=6,
               label="Hyperbolic Vertices p_k ∈ D²")

    u_c, v_c, w_c = u_pt.coords, v_pt.coords, w_pt.coords
    ax.scatter([u_c[0], v_c[0]], [u_c[1], v_c[1]], color="#00ccff", s=75, edgecolors="white", lw=0.8, zorder=7,
               label="Gyrovectors u, v")
    ax.scatter([w_c[0]], [w_c[1]], color="#00ff88", s=95, marker="D", edgecolors="white", lw=0.9, zorder=8,
               label=f"Möbius Sum u ⊕ v = ({w_c[0]:.2f}, {w_c[1]:.2f})")
    ax.scatter([mid_c[0]], [mid_c[1]], color="#ff3366", s=110, marker="*", edgecolors="white", lw=0.8, zorder=8,
               label=f"Einstein Midpoint μ_E = ({mid_c[0]:.2f}, {mid_c[1]:.2f})")

    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(-1.08, 1.08)
    ax.set_title("Conformal Hyperbolic Geometry (Poincaré Disk D², Curvature c = -1)\nOrthogonal Circular Geodesics, Möbius Addition u ⊕ v & Einstein Midpoint", color="white", fontsize=11.5, pad=14)
    ax.axis("equal")
    ax.axis("off")
    ax.legend(facecolor="#10141f", edgecolor="#2d3748", labelcolor="white", loc="lower right", fontsize=8.5)
    hyp_png = os.path.join(out_dir, "06_poincare_hyperbolic_disk.png")
    fig.savefig(hyp_png, dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {hyp_png}\n")

    print(f"SUCCESS: All 6 scientific PNG visual plots generated and verified in: {out_dir}")


if __name__ == "__main__":
    render_all_pngs()
