# ============================================================================
# VERSORC-MATH: SCIENTIFIC MATPLOTLIB VISUALIZATION SUITE
# License: Apache-2.0 OR MIT
# ============================================================================

import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import Optional, List, Tuple

from ._versorc_math import (
    Cl3,
    PoincarePoint,
    get_e8_roots,
    get_lloyd_max_3bit_centroids,
)


def compute_e8_coxeter_basis() -> Tuple[np.ndarray, np.ndarray]:
    """
    Constructs the exact orthonormal 2D Coxeter plane basis (u1, u2) in R^8 from the
    8 Bourbaki simple roots of E8 via the product of simple reflections C = s_1 ... s_8,
    extracting the eigenspace corresponding to the primitive Coxeter eigenvalue exp(2*pi*i / 30).
    (Coxeter, Regular Polytopes, 1973; Conway & Sloane, Sphere Packings, Lattices and Groups).
    """
    simple_roots = np.array([
        [ 0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5,  0.5],
        [ 1.0,  1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
        [-1.0,  1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
        [ 0.0, -1.0,  1.0,  0.0,  0.0,  0.0,  0.0,  0.0],
        [ 0.0,  0.0, -1.0,  1.0,  0.0,  0.0,  0.0,  0.0],
        [ 0.0,  0.0,  0.0, -1.0,  1.0,  0.0,  0.0,  0.0],
        [ 0.0,  0.0,  0.0,  0.0, -1.0,  1.0,  0.0,  0.0],
        [ 0.0,  0.0,  0.0,  0.0,  0.0, -1.0,  1.0,  0.0],
    ], dtype=np.float64)

    coxeter_mat = np.eye(8, dtype=np.float64)
    for alpha in simple_roots:
        refl = np.eye(8, dtype=np.float64) - 2.0 * np.outer(alpha, alpha) / np.dot(alpha, alpha)
        coxeter_mat = coxeter_mat @ refl

    eigvals, eigvecs = np.linalg.eig(coxeter_mat)
    exponents = np.angle(eigvals) * 30.0 / (2.0 * np.pi)
    idx = int(np.argmin(np.abs(exponents - 1.0)))
    v = eigvecs[:, idx]
    u1 = np.real(v)
    u2 = np.imag(v)
    u1 = u1 / np.linalg.norm(u1)
    u2 = u2 - np.dot(u2, u1) * u1
    u2 = u2 / np.linalg.norm(u2)
    return u1, u2


def plot_lloyd_max_centroids(
    samples: Optional[np.ndarray] = None,
    bins: int = 80,
    title: str = "Clifford 3-Bit Lloyd-Max Centroids & Decision Boundaries",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Renders the 8 Lloyd-Max 3-bit centroids against the standard normal distribution N(0, 1)
    or RMS-normalized rotated weights R(W)/sigma, visualizing optimal minimum-MSE quantization.
    """
    centroids = get_lloyd_max_3bit_centroids()
    sorted_centroids = np.sort(centroids)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=120)

    # Plot theoretical Gaussian density N(0, 1)
    x = np.linspace(-3.5, 3.5, 1000)
    pdf = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2)
    ax.plot(x, pdf, "k-", lw=1.8, label="Gaussian N(0, 1) Reference PDF")

    if samples is not None:
        flat = np.asarray(samples, dtype=np.float32).ravel()
        rms = float(np.sqrt(np.mean(flat**2)))
        flat_norm = flat / rms if rms > 1e-6 else flat
        ax.hist(
            flat_norm,
            bins=bins,
            density=True,
            alpha=0.35,
            color="#2b5c8f",
            edgecolor="none",
            label="Normalized Rotated Distribution R(W) / σ",
        )

    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
    for i, c in enumerate(sorted_centroids):
        ax.axvline(
            c,
            color=palette[i % len(palette)],
            linestyle="--",
            lw=1.5,
            label=f"Level {i}: {c:+.4f}" if samples is None else None,
        )

    # Midpoint decision boundaries
    for i in range(len(sorted_centroids) - 1):
        mid = (sorted_centroids[i] + sorted_centroids[i + 1]) / 2.0
        ax.axvline(
            mid,
            color="#888888",
            linestyle=":",
            lw=0.9,
            alpha=0.7,
            label="Decision Boundaries" if i == 0 and samples is None else None,
        )

    ax.set_title(title, fontsize=12, pad=12)
    ax.set_xlabel("Normalized Value (Standard Deviations σ)", fontsize=10)
    ax.set_ylabel("Probability Density", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_e8_coxeter_plane(
    title: str = "E8 Root System Projection on 2D Coxeter Plane (240 Roots, D30 Symmetry)",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Projects the 240 minimal roots of the E8 lattice in R^8 onto the true 2D Coxeter plane,
    revealing 8 concentric rings of 30 vertices with exact 30-fold dihedral symmetry.
    """
    roots = get_e8_roots().astype(np.float64)  # Shape: (240, 8)
    u1, u2 = compute_e8_coxeter_basis()

    proj_x = roots @ u1
    proj_y = roots @ u2

    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    fig.patch.set_facecolor("#0f111a")
    ax.set_facecolor("#0f111a")

    # Connect true E8 root graph edges: roots with inner product <r_i, r_j> = 1 (distance sqrt(2) in R^8)
    gram = roots @ roots.T
    for i in range(240):
        neighbors = np.where(np.abs(gram[i] - 1.0) < 1e-4)[0]
        for j in neighbors:
            if i < j:
                ax.plot(
                    [proj_x[i], proj_x[j]],
                    [proj_y[i], proj_y[j]],
                    color="#4e88c7",
                    alpha=0.06,
                    lw=0.45,
                )

    radii = np.sqrt(proj_x**2 + proj_y**2)
    ax.scatter(
        proj_x,
        proj_y,
        c=radii,
        cmap="plasma",
        s=28,
        alpha=0.95,
        edgecolors="white",
        linewidths=0.35,
        zorder=5,
    )

    ax.set_title(title, color="white", fontsize=11, pad=15)
    ax.axis("equal")
    ax.axis("off")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    return fig


def plot_clifford_isotropy_comparison(
    original_weights: np.ndarray,
    rotated_weights: np.ndarray,
    title: str = "Outlier Smearing via Dyadic Walsh Versor Isometry: W vs R(W)",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Renders side-by-side probability density of RMS-normalized raw weights vs.
    versor-rotated weights, demonstrating the complete elimination of tail outliers
    and alignment with the 8 Lloyd-Max centroids.
    """
    orig_flat = np.asarray(original_weights, dtype=np.float32).ravel()
    rot_flat = np.asarray(rotated_weights, dtype=np.float32).ravel()
    rms = float(np.sqrt(np.mean(rot_flat**2)))
    orig_norm = orig_flat / rms if rms > 1e-6 else orig_flat
    rot_norm = rot_flat / rms if rms > 1e-6 else rot_flat

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=120)

    bins = 100
    ax1.hist(orig_norm, bins=bins, density=True, alpha=0.5, color="#d9534f", label="Raw Weights W / σ (Outliers)")
    ax1.hist(rot_norm, bins=bins, density=True, alpha=0.5, color="#2b5c8f", label="Rotated R(W) / σ (Gaussian)")
    ax1.set_title("Probability Density (Linear)", fontsize=11)
    ax1.set_xlabel("Normalized Weight (σ units)", fontsize=10)
    ax1.set_ylabel("Density", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.legend()

    ax2.hist(orig_norm, bins=bins, density=True, alpha=0.5, color="#d9534f", log=True, label="Raw W / σ (Extreme Tails)")
    ax2.hist(rot_norm, bins=bins, density=True, alpha=0.5, color="#2b5c8f", log=True, label="Rotated R(W) / σ (Bounded)")
    for c in np.sort(get_lloyd_max_3bit_centroids()):
        ax2.axvline(c, color="#ffaa00", linestyle="--", lw=1.1, alpha=0.8)
    ax2.set_title("Probability Density (Logarithmic Tails & Centroids)", fontsize=11)
    ax2.set_xlabel("Normalized Weight (σ units)", fontsize=10)
    ax2.set_ylabel("Log Density", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.legend()

    fig.suptitle(title, fontsize=12, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_poincare_disk(
    points: Optional[np.ndarray] = None,
    title: str = "Poincaré Hyperbolic Disk Model D^2 (Orthogonal Geodesics)",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Renders 2D points inside the conformal Poincaré disk model with exact hyperbolic
    geodesic circular arcs orthogonal to the boundary circle S^1.
    """
    fig, ax = plt.subplots(figsize=(6, 6), dpi=120)

    theta = np.linspace(0, 2 * np.pi, 500)
    ax.plot(np.cos(theta), np.sin(theta), "k-", lw=1.5, label="Boundary S^1 (|z| = 1)")
    ax.fill(np.cos(theta), np.sin(theta), color="#f5f7fa", zorder=0)

    for r in [0.3, 0.5, 0.7, 0.9]:
        ax.plot(r * np.cos(theta), r * np.sin(theta), color="#c0c4cc", linestyle=":", lw=0.8)

    if points is not None:
        pts = np.asarray(points, dtype=np.float32)
        assert pts.shape[1] == 2, "Points must be of shape (N, 2)"
        norms = np.linalg.norm(pts, axis=1)
        valid_pts = pts[norms < 1.0]
        p_objs = [PoincarePoint([float(p[0]), float(p[1])]) for p in valid_pts]
        for i in range(len(p_objs)):
            for j in range(i + 1, len(p_objs)):
                arc = np.array(p_objs[i].geodesic_arc(p_objs[j], 60))
                ax.plot(arc[:, 0], arc[:, 1], color="#fd7e14", lw=1.4, alpha=0.75)
        ax.scatter(valid_pts[:, 0], valid_pts[:, 1], c="#2b5c8f", s=45, zorder=5, label="Hyperbolic Points")

    ax.set_title(title, fontsize=11, pad=10)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.axis("equal")
    ax.axis("off")
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_bivector_blade(
    u: np.ndarray,
    v: np.ndarray,
    origin: Optional[np.ndarray] = None,
    title: str = "Geometric Algebra 2-Blade: Bivector u ∧ v",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Renders a 2-blade (bivector u ∧ v) in Cl(3,0) using VersorC-Math native Cl3 operations:
    - 1-blades u and v as vector arrows
    - 2-blade outer product B = u ^ v as an oriented planar parallelogram
    - Canonical Hodge dual *(u ^ v) = (u ^ v) * I^-1 via Cl3.dual()
    """
    u_vec = np.asarray(u, dtype=np.float64).ravel()
    v_vec = np.asarray(v, dtype=np.float64).ravel()
    assert len(u_vec) == 3 and len(v_vec) == 3, "Vectors u and v must be 3-dimensional"

    orig = np.zeros(3, dtype=np.float64) if origin is None else np.asarray(origin, dtype=np.float64).ravel()

    u_mv = Cl3.vector(float(u_vec[0]), float(u_vec[1]), float(u_vec[2]))
    v_mv = Cl3.vector(float(v_vec[0]), float(v_vec[1]), float(v_vec[2]))
    biv = u_mv ^ v_mv
    dual_mv = biv.dual()
    dual_arr = dual_mv.to_numpy()
    dual = np.array([dual_arr[1], dual_arr[2], dual_arr[4]], dtype=np.float64)
    area = float(biv.norm())

    fig = plt.figure(figsize=(8, 7), dpi=120)
    ax = fig.add_subplot(111, projection="3d")

    verts = [np.array([orig, orig + u_vec, orig + u_vec + v_vec, orig + v_vec])]
    poly = Poly3DCollection(verts, alpha=0.35, facecolor="#2b5c8f", edgecolor="#1a3a5c", lw=1.5)
    ax.add_collection3d(poly)

    ax.quiver(orig[0], orig[1], orig[2], u_vec[0], u_vec[1], u_vec[2],
              color="#d9534f", lw=2.2, arrow_length_ratio=0.1, label=f"1-Blade u ({u_vec[0]:.1f}, {u_vec[1]:.1f}, {u_vec[2]:.1f})")
    ax.quiver(orig[0], orig[1], orig[2], v_vec[0], v_vec[1], v_vec[2],
              color="#28a745", lw=2.2, arrow_length_ratio=0.1, label=f"1-Blade v ({v_vec[0]:.1f}, {v_vec[1]:.1f}, {v_vec[2]:.1f})")

    center = orig + (u_vec + v_vec) / 2.0
    if area > 1e-6:
        ax.quiver(center[0], center[1], center[2], dual[0], dual[1], dual[2],
                  color="#6f42c1", lw=2.0, linestyle="--", arrow_length_ratio=0.12,
                  label=f"Hodge Dual (u ∧ v)I⁻¹ (|B| = {area:.2f})")

    all_pts = np.vstack([orig, orig + u_vec, orig + v_vec, orig + u_vec + v_vec, center + dual])
    min_xyz = np.min(all_pts, axis=0) - 0.3
    max_xyz = np.max(all_pts, axis=0) + 0.3
    max_range = np.max(max_xyz - min_xyz) / 2.0
    mid_xyz = (max_xyz + min_xyz) / 2.0
    ax.set_xlim(mid_xyz[0] - max_range, mid_xyz[0] + max_range)
    ax.set_ylim(mid_xyz[1] - max_range, mid_xyz[1] + max_range)
    ax.set_zlim(mid_xyz[2] - max_range, mid_xyz[2] + max_range)

    ax.set_xlabel("e1", fontsize=10)
    ax.set_ylabel("e2", fontsize=10)
    ax.set_zlabel("e3", fontsize=10)
    ax.set_title(title, fontsize=11, pad=12)
    ax.legend(loc="upper left", frameon=True)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_rotor_orbit(
    rotation_axis: np.ndarray,
    theta_rad: float,
    probe_vector: np.ndarray,
    num_steps: int = 60,
    title: str = "Clifford Rotor Action: v' = R v R~",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Renders the continuous rotation orbit of a probe vector under the action of a Cl(3,0) rotor
    computed directly via R.sandwich(v0):
        R(t) = cos(t/2) - B_unit sin(t/2)
        v(t) = R(t) v_0 R(t)~
    """
    n = np.asarray(rotation_axis, dtype=np.float64).ravel()
    n_norm = np.linalg.norm(n)
    assert n_norm > 1e-6, "Rotation axis must be non-zero"
    n_unit = n / n_norm

    v0 = np.asarray(probe_vector, dtype=np.float64).ravel()
    assert len(v0) == 3, "Probe vector must be 3-dimensional"

    v0_mv = Cl3.vector(float(v0[0]), float(v0[1]), float(v0[2]))
    n_mv = Cl3.vector(float(n_unit[0]), float(n_unit[1]), float(n_unit[2]))
    b_unit = n_mv.dual()  # Unit bivector plane orthogonal to n_unit

    t_vals = np.linspace(0, theta_rad, num_steps)
    orbit = np.zeros((num_steps, 3), dtype=np.float64)
    for i, t in enumerate(t_vals):
        rotor = Cl3.from_scalar(float(math.cos(t * 0.5))) - b_unit * float(math.sin(t * 0.5))
        vt_arr = rotor.sandwich(v0_mv).to_numpy()
        orbit[i] = [vt_arr[1], vt_arr[2], vt_arr[4]]

    fig = plt.figure(figsize=(8, 7), dpi=120)
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(orbit[:, 0], orbit[:, 1], orbit[:, 2], color="#fd7e14", lw=2.8, label=f"Rotor Orbit (θ = {np.degrees(theta_rad):.1f}°)")
    ax.quiver(0, 0, 0, v0[0], v0[1], v0[2], color="#2b5c8f", lw=2.2, arrow_length_ratio=0.1, label="Initial Vector v(0)")

    v_final = orbit[-1]
    ax.quiver(0, 0, 0, v_final[0], v_final[1], v_final[2], color="#28a745", lw=2.2, arrow_length_ratio=0.1, label="Rotated v(θ) = R v(0) R~")
    ax.quiver(0, 0, 0, n_unit[0], n_unit[1], n_unit[2], color="#6f42c1", lw=1.8, linestyle=":", arrow_length_ratio=0.15, label="Bivector Dual Axis *(B)")

    all_pts = np.vstack([np.zeros(3), v0, v_final, n_unit, orbit])
    min_xyz = np.min(all_pts, axis=0) - 0.3
    max_xyz = np.max(all_pts, axis=0) + 0.3
    max_range = np.max(max_xyz - min_xyz) / 2.0
    mid_xyz = (max_xyz + min_xyz) / 2.0
    ax.set_xlim(mid_xyz[0] - max_range, mid_xyz[0] + max_range)
    ax.set_ylim(mid_xyz[1] - max_range, mid_xyz[1] + max_range)
    ax.set_zlim(mid_xyz[2] - max_range, mid_xyz[2] + max_range)

    ax.set_xlabel("e1", fontsize=10)
    ax.set_ylabel("e2", fontsize=10)
    ax.set_zlabel("e3", fontsize=10)
    ax.set_title(title, fontsize=11, pad=12)
    ax.legend(loc="upper left", frameon=True)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig
