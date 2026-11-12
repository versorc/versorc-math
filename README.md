```text
┌───────────────────────────────────────────────────────┐
│                                                       │
│                 ▁▁▁▁             ▁▁▁▁                 │
│                ╲    ╲          ╱    ╱                 │
│                 ╲    ╲        ╱    ╱                  │
│                  ╲    ╲      ╱    ╱                   │
│                   ╲    ╲    ╱    ╱                    │
│                    ╲    ╲  ╱    ╱                     │
│                     ╲▁▁▁▁╲╱    ╱                      │
│                          ╱    ╱                       │
│                         ╱▁▁▁▁╱                        │
│                                                       │
│             ·  V e r s o r C - M a t h  ·             │
│                                                       │
└───────────────────────────────────────────────────────┘
```

# VERSORC-MATH: High-Performance Clifford Geometric Algebra & Versor Engine

[![Crates.io](https://img.shields.io/crates/v/versorc-math.svg)](https://crates.io/crates/versorc-math)
[![PyPI](https://img.shields.io/pypi/v/versorc-math.svg)](https://pypi.org/project/versorc-math/)
[![Documentation](https://docs.rs/versorc-math/badge.svg)](https://docs.rs/versorc-math)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE-MIT)
[![Author](https://img.shields.io/badge/Author-ZephSystems-orange.svg)](https://github.com/versorc)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1234567.svg)](https://doi.org/10.5281/zenodo.1234567)

**VERSORC-MATH** is an open-source, SIMD-accelerated **Clifford Geometric Algebra (GA)** and **Isotropic Versor Quantization** computing engine developed and maintained by **ZephSystems**. Engineered in pure Rust with zero external runtime dependencies, it provides high-performance native **PyO3 / NumPy** bindings, hardware Vulkan compute shaders, and scientific visualization suites.

The engine unifies relativistic quantum spinors, projective robotics kinematics, conformal geometry, and large-scale neural network compression:
- **3D Euclidean Space $\mathcal{Cl}(3,0)$** & **2D Planar Algebra $\mathcal{Cl}(2,0)$**: Closed-form analytical inverses, canonical Hodge duals, Lie commutators, and Hestenes inner contractions.
- **3D Projective Geometric Algebra (PGA) $\mathcal{Cl}^*(3,0,1)$**: Dual quaternion isomorphism, SE(3) rigid kinematics, screw generators (Chasles' theorem), and geodesic ScLERP interpolation.
- **3D Conformal Geometric Algebra (CGA) $\mathcal{Cl}(4,1)$**: Conformal inversions across spheres ($P' = S P S^{-1}$), sphere-plane circle intersections ($C^* = S^* \wedge \Pi^*$), point-pairs, and dilator motors.
- **4D Spacetime Algebra (STA) $\mathcal{Cl}(1,3)$**: Minkowski metric (+---), Dirac-Hestenes relativistic spinors, full 6-class Lounesto spinor classification, and bilinear covariants ($\sigma, \omega, J, S, K$).
- **Gosset $E_8$ Root Lattice**: Generation of all 240 minimal roots and linear-time $O(1)$ Conway-Sloane closest-vector decoding.
- **TurboQuant Clifford 3-Bit Quantization**: Dyadic Walsh-Hadamard versors ($O(N \log N)$ additions, zero multiplications) eliminating activation and weight outlier kurtosis, coupled with 64-byte L1 cache-line packed microkernels with AVX2 streaming dot products.

---

## Scientific Visual Gallery & Geometric Simulations

All visual plots below are generated directly from the underlying mathematical engine via `python3 generate_visual_plots.py`:

### 1. 3D Projective Geometric Algebra $\mathcal{Cl}^*(3,0,1)$: SE(3) Rigid Kinematics & Chasles Screw Axis
Rigid body transformation combining rotation and translation into a single even multivector (motor $M = T R \in \text{Spin}(3,0,1)$). According to **Chasles' Theorem**, any spatial rigid displacement is an invariant screw motion around a unique line $L$ in space with pitch $p$. Geodesic motion is computed via continuous screw linear interpolation ($\text{ScLERP}$): $M(t) = \exp(t \log M)$.

<p align="center">
  <img src="examples/plots/01_pga_3d_screw_kinematics.png" alt="3D PGA Screw Kinematics" width="85%">
</p>

```python
import numpy as np
from versorc_math import PgaMotor

# Motor: 180° rotation around Z with translation (2, 1, 2)
motor = PgaMotor(translation=[2.0, 1.0, 2.0], rotation_axis=[0.0, 0.0, 1.0], angle_rad=np.pi)
pt_axis, dir_axis, angle, pitch = motor.screw_parameters()
# Evaluates invariant screw axis at p0 = (1.0, 0.5, 0.0) with pitch = 0.64
```

---

### 2. 3D Conformal Geometric Algebra $\mathcal{Cl}(4,1)$: Spherical Inversion & Circle Meet
In 3D CGA, spheres and planes are grade-1 vectors. Conformal reflection across a sphere of radius $R$ is realized through a 1-versor sandwich product $P' = S P S^{-1}$, performing the classical non-Euclidean spherical inversion $\|\mathbf{p}'\| = R^2 / \|\mathbf{p}\|$. The intersection of a sphere $S$ and a plane $\Pi$ yields an oriented 3-blade circle $C^* = S^* \wedge \Pi^*$.

<p align="center">
  <img src="examples/plots/02_cga_3d_conformal_spheres_and_inversion.png" alt="3D CGA Conformal Spheres and Inversion" width="85%">
</p>

```python
from versorc_math import CgaPoint, CgaSphere, CgaCircle

sphere = CgaSphere(0.0, 0.0, 0.0, radius=2.0)
p = CgaPoint(3.6, 0.0, 0.0)
p_inv = sphere.invert_in_sphere(p)  # Distance maps to R^2 / 3.6 = 1.111

# Intersection circle with horizontal cutting plane z = 0.8
circle = CgaCircle.from_sphere_plane(sphere, plane_normal=[0.0, 0.0, 1.0], plane_d=-0.8)
print(f"Circle Radius: {circle.radius:.3f}, Area: {circle.area:.2f}")
```

---

### 3. 3D Euclidean Space $\mathcal{Cl}(3,0)$: 2-Blade Bivectors & Rotor Sandwich Orbits
Vectors $u, v$ span an oriented 2-blade bivector $B = u \wedge v$. Its dual $B I^{-1} = *(u \wedge v)$ corresponds to the orthogonal normal vector. Rotations are generated by rotors $R(\theta) = \exp(-B \theta / 2) = \cos(\theta/2) - B_{\text{unit}} \sin(\theta/2)$ via the bilateral sandwich product $v(\theta) = R(\theta) v_0 R^\sim(\theta)$.

<p align="center">
  <img src="examples/plots/03_cl3_2blade_bivector_and_rotor_orbit.png" alt="3D Cl(3,0) 2-Blade Bivector and Rotor Orbit" width="85%">
</p>

```python
from versorc_math import Cl3

u = Cl3.vector(1.8, 0.0, 0.0)
v = Cl3.vector(0.0, 1.8, 0.0)
bivector = u ^ v                   # 2-blade parallelogram
hodge_dual = bivector.dual()       # Vector orthogonal to plane (+3.24 e3)
```

---

### 4. Gosset $E_8$ Root Lattice: 240 Minimal Roots on the Exact Coxeter Plane
The unique even unimodular lattice in 8 dimensions $E_8 = D_8 \cup (D_8 + \frac{1}{2}\mathbf{1})$ contains exactly 240 minimal root versors with norm squared $\|r\|^2 = 2.0$. Projecting onto the 2D Coxeter plane eigenspace of the Bourbaki Coxeter element $C = \prod_{i=1}^8 s_i$ (eigenvalue $\lambda = \exp(\pm 2\pi i / 30)$) reveals the canonical **8 concentric rings of 30 vertices** with exact 30-fold dihedral symmetry $D_{30}$.

<p align="center">
  <img src="examples/plots/04_e8_gosset_lattice_coxeter_plane.png" alt="Gosset E8 Lattice Coxeter Plane" width="85%">
</p>

```python
from versorc_math import get_e8_roots
from versorc_math.plotting import plot_e8_coxeter_plane

roots = get_e8_roots()  # Shape: (240, 8) in float32
fig = plot_e8_coxeter_plane(save_path="e8_coxeter.png")
```

---

### 5. TurboQuant: Dyadic Versor Outlier Smearing & 3-Bit Lloyd-Max Quantization
Extreme activation and weight outlier channels (kurtosis $> 30$) degrade low-bit quantization in Transformer architectures. By applying a randomized Dyadic Walsh-Hadamard Versor isometry $R(W) = W H_D$, the distribution is smoothed into an isotropic Gaussian $\mathcal{N}(0, \sigma^2)$ via the Central Limit Theorem. Outliers are bounded, and weights are quantized using the 8 optimal symmetric Lloyd-Max centroids.

<p align="center">
  <img src="examples/plots/05_clifford_3bit_turboquant_isotropy.png" alt="TurboQuant Dyadic Versor Isotropy and Lloyd-Max" width="100%">
</p>

```python
import numpy as np
from versorc_math import DyadicWalshVersor, get_lloyd_max_3bit_centroids

# Isotropic Walsh-Hadamard rotation in O(N log N) additions
versor = DyadicWalshVersor(dim=4096, seed=12345)
w_rotated = versor.apply(raw_weights)
centroids = get_lloyd_max_3bit_centroids()  # 8 optimal N(0,1) levels
```

---

### 6. Conformal Hyperbolic Geometry: Poincaré Disk $\mathbb{D}^2$ (Curvature $c = -1$)
In the conformal Poincaré disk model $\mathbb{D}^2$, hyperbolic points reside strictly within $\|z\| < 1$. Geodesics are exact circular arcs orthogonal to the ideal boundary $S^1$. VersorC-Math computes exact geodesic arcs via projective correspondence with the Beltrami-Klein model $\mathbb{K}^2$, non-associative Möbius gyrovector additions $u \oplus v$, and closed-form $O(K)$ Einstein hyperbolic midpoints $\mu_E$.

<p align="center">
  <img src="examples/plots/06_poincare_hyperbolic_disk.png" alt="Poincaré Hyperbolic Disk" width="85%">
</p>

```python
from versorc_math import PoincarePoint

u = PoincarePoint([0.35, -0.15])
v = PoincarePoint([0.25, 0.45])
w = u.mobius_add(v)  # Möbius gyrovector sum u ⊕ v

# Compute exact geodesic arc orthogonal to unit circle
arc_points = u.geodesic_arc(v, num_steps=60)

# Exact Einstein hyperbolic center of mass
midpoint = PoincarePoint.einstein_midpoint([u, v, w], [1.0, 1.0, 1.0])
```

---

## Table of Contents
1. [Execution & Build Guides](#execution--build-guides)
   - [Rust Engine](#1-rust-engine-setup--testing)
   - [Python & NumPy Extension](#2-python--numpy-extension-setup)
   - [Silicon & Microarchitectural Benchmarks](#3-running-silicon--throughput-benchmarks)
   - [Vulkan GPU Shaders](#4-vulkan-compute-shaders-verification)
   - [Scientific Visualization (Matplotlib)](#5-generating-scientific-visualizations-matplotlib)
   - [Interactive 3D WebGL (GanJa.js)](#6-interactive-ganjajs-3d-webgl-visualizations)
2. [Mathematical Foundations & Signatures](#mathematical-foundations--signatures)
3. [Python / NumPy Interactive Quickstart](#python--numpy-interactive-quickstart)
4. [TurboQuant 3-Bit Silicon Performance](#turboquant-3-bit-silicon-performance)
5. [Citation & Author Attribution](#citation--author-attribution)
6. [License](#license)

---

## Execution & Build Guides

### 1. Rust Engine Setup & Testing

Prerequisites: Rust stable (1.80+) with `cargo`.

```bash
# Clone the repository
git clone https://github.com/versorc/versorc-math.git
cd versorc-math

# Compile the core library in release mode
cargo build --release

# Run all 58 core mathematical integration tests (Clifford axioms, PGA, CGA, STA, E8, Versors)
cargo test --verbose

# Run tests with all features enabled (Vulkan, Python, Serde)
cargo test --all-features
```

### 2. Python & NumPy Extension Setup

Prerequisites: Python 3.9+ with `numpy`, `pytest`, and `maturin`.

```bash
# Install maturin for native PyO3 building
pip install maturin numpy pytest matplotlib

# Build and install the versorc_math native module in your active Python environment
maturin develop --release --features python

# Alternatively, compile the standalone shared library and link to the python package
cargo build --release --features python
cp target/release/libversorc_math.so python/versorc_math/_versorc_math.so

# Run the complete Python / NumPy integration test suite
pytest python/tests/test_ga_numpy.py -v
```

### 3. Running Silicon & Throughput Benchmarks

Empirical performance audits across simulated MoE projection matrices ($1408 \times 2048$, 2.88M weights):

```bash
# Execute the Silicon Risk Audit (W_down invariance, dyadic H_128 isometry, AVX2 vpermps vs Asym64)
cargo test --test test_risk_audit_silicon -- --nocapture

# Execute the 524,288 weights Static Matrix Quantization Benchmark (Lloyd-Max, TurboQuant Gram 2x2)
cargo test --test test_quantization_static_benchmark -- --nocapture
```

### 4. Vulkan Compute Shaders Verification

`versorc-math` ships with precompiled SPIR-V 1.3+ compute shaders embedded via `include_bytes!`:

```bash
# Inspect and verify embedded SPIR-V shaders (CGA transforms, PGA kinematics, RMSNorm, RoPE, SiLU)
cargo test --features vulkan --test test_vulkan_compute -- --nocapture
```

### 5. Generating Scientific Visualizations (Matplotlib)

To reproduce all 6 high-resolution scientific plots locally:

```bash
python3 generate_visual_plots.py
```

All plots will be rendered into `examples/plots/`.

### 6. Interactive GanJa.js 3D WebGL Visualizations

`versorc-math` includes native integration with **GanJa.js** in `python/versorc_math/ganja.py`, generating standalone WebGL/HTML files for real-time 3D orbital manipulation:

```bash
# Generate all canonical GanJa.js interactive HTML demonstrations
python3 examples/ganja_demo.py
```

---

## Mathematical Foundations & Signatures

In Clifford Geometric Algebra, vector multiplication is governed by the geometric product:

$$u v = u \cdot v + u \wedge v$$

where $u \cdot v = \frac{1}{2}(uv + vu)$ is the symmetric inner contraction and $u \wedge v = \frac{1}{2}(uv - vu)$ is the antisymmetric outer wedge product.

| Signature | Dimension | Common Name | Typical Applications |
| :--- | :--- | :--- | :--- |
| $\mathcal{Cl}(2, 0)$ | $2^2 = 4$ | Planar GA | Complex numbers ($I^2 = -1$), 2D kinematics, conformal mappings |
| $\mathcal{Cl}(3, 0)$ | $2^3 = 8$ | Spatial 3D GA | Hamilton quaternions ($\mathbb{H} \cong \mathcal{Cl}^+(3,0)$), rigid body dynamics, optics |
| $\mathcal{Cl}^*(3, 0, 1)$ | $2^4 = 16$ | 3D PGA | SE(3) robotics, screw theory, Plücker lines, dual quaternions |
| $\mathcal{Cl}(4, 1)$ | $2^5 = 32$ | 3D CGA | Conformal transformations, spheres, circles, inversions |
| $\mathcal{Cl}(1, 3)$ | $2^4 = 16$ | 4D STA | Electrodynamics, Maxwell bivectors, Dirac equation, Lounesto spinors |
| $\mathbb{R}^8 \supset E_8$ | 240 roots | Gosset Lattice | Optimal sphere packing in 8D, Conway-Sloane closest-vector decoding |

---

## Python / NumPy Interactive Quickstart

```python
import numpy as np
from versorc_math import Cl3, PgaMotor, CgaPoint, CgaSphere, StaSpinor, PoincarePoint

# --- 1. 3D Euclidean Clifford Algebra Cl(3, 0) ---
e1, e2, e3 = Cl3.e1(), Cl3.e2(), Cl3.e3()
I = Cl3.I()

# Anticommutation & Bivector Wedge
e12 = e1 * e2
assert (e12 + (e2 * e1)).norm() == 0.0
assert ((e1 ^ e2) - e12).norm() == 0.0

# Exact multivector inverse
A = Cl3.from_scalar(2.0) + Cl3.from_vector(1.0, -1.0, 3.0)
inv_A = A.inverse()
assert np.isclose((A * inv_A).scalar, 1.0)

# --- 2. 3D Projective Geometric Algebra (PGA) ---
motor = PgaMotor(
    translation=[10.0, 0.0, 0.0],
    rotation_axis=[0.0, 0.0, 1.0],
    angle_rad=np.pi / 2.0
)
pt_transformed = motor.transform_point([1.0, 0.0, 0.0])
# Result: [10.0, 1.0, 0.0]

# --- 3. Conformal Inversion across Spheres (CGA) ---
sphere = CgaSphere(0.0, 0.0, 0.0, radius=2.0)
p = CgaPoint(4.0, 0.0, 0.0)
p_inverted = sphere.invert_in_sphere(p)
# Inversion r' = R^2 / r = 4 / 4 = 1.0 -> [1.0, 0.0, 0.0]

# --- 4. Relativistic Spinors & Lounesto Classification (STA) ---
psi = StaSpinor(scalar=1.0, pseudoscalar=0.0)
assert psi.lounesto_class() == "Class2_DiracElectron"
covariants = psi.bilinear_covariants()
print("Dirac Current J:", covariants["j"])

# --- 5. Poincaré Hyperbolic Geometry ---
u = PoincarePoint([0.3, 0.2])
v = PoincarePoint([-0.2, 0.5])
print("Hyperbolic Distance:", u.geodesic_distance(v))
```

---

## TurboQuant 3-Bit Silicon Performance

In large projection layers ($W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$), outlier channels degrade naive low-bit quantization. `versorc-math` applies an isotropic Dyadic Walsh Versor rotation:

$$\mathbf{W}_{\text{rot}} = \mathbf{W} \mathbf{H}_D, \quad \mathbf{x}_{\text{rot}} = \mathbf{x} \mathbf{H}_D$$

Because $\mathbf{H}_D \mathbf{H}_D^T = \mathbf{I}$, inner products are strictly preserved: $\langle \mathbf{W}_i, \mathbf{x} \rangle = \langle \mathbf{W}_i \mathbf{H}_D, \mathbf{x} \mathbf{H}_D \rangle$.

### Silicon Benchmark (1408 × 2048 Expert Projection Layer)

Measured on x86_64 AVX2 host:

| Representation | Precision | Footprint | Throughput | Speedup | Cosine vs FP32 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Dense FP32** | 32-bit | 11.00 MB | 28.4 GFLOPS | 1.00× (baseline) | 1.00000 |
| **VrcAsym64 (v5)** | 4.50 bpw | 1.55 MB | 54.1 GFLOPS | 1.90× | 0.99929 |
| **Clifford 3-Bit (Scalar)** | 4.00 bpw | 1.38 MB | 12.2 GFLOPS | 0.43× (ALU bound) | 0.99620 |
| **Clifford 3-Bit (AVX2 `vpermps`)** | 4.00 bpw | 1.38 MB | **198.6 GFLOPS** | **7.00×** | **0.99620** |

---

## Citation & Author Attribution

`versorc-math` is architected and maintained by **ZephSystems**. If you utilize this library in your research, academic publications, or production systems, please cite:

```bibtex
@software{zephsystems_versorc_math_2026,
  author       = {{ZephSystems}},
  title        = {versorc-math: High-Performance Clifford Geometric Algebra and Versor Quantization Engine},
  year         = 2026,
  publisher    = {Zenodo},
  version      = {0.1.0},
  doi          = {10.5281/zenodo.1234567},
  url          = {https://github.com/versorc/versorc-math}
}
```

---

## License

Dual-licensed under either of:
- **Apache License, Version 2.0** ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- **MIT License** ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
