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

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Crates.io](https://img.shields.io/crates/v/versorc-math.svg)](https://crates.io/crates/versorc-math)
[![Crates.io Downloads](https://img.shields.io/crates/d/versorc-math.svg?color=orange&label=crates.io%20downloads)](https://crates.io/crates/versorc-math)
[![PyPI](https://img.shields.io/pypi/v/versorc-math.svg)](https://pypi.org/project/versorc-math/)
[![PyPI Downloads](https://img.shields.io/pepy/dt/versorc-math.svg?color=blue&label=pypi%20downloads)](https://pypi.org/project/versorc-math/)
[![Documentation](https://docs.rs/versorc-math/badge.svg)](https://docs.rs/versorc-math)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24153328.svg)](https://doi.org/10.5281/zenodo.24153328)
[![Release](https://img.shields.io/badge/Release-v0.1.1-brightgreen.svg)](https://github.com/versorc/versorc-math/releases)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/versorc/versorc-math/pulls)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)
[![Community](https://img.shields.io/badge/Community-Open_Source-purple.svg)](https://github.com/versorc/versorc-math)

**VERSORC-MATH** is an open-source, community-driven **Clifford Geometric Algebra (GA)**, **Versor Operator**, and **Geometric Quantization** engine written in pure Rust with zero required runtime dependencies. It provides high-performance native **PyO3 / NumPy** bindings, hardware **Vulkan compute shaders**, and scientific visualization suites.

The project is released under the **Apache License 2.0** to foster an open, collaborative ecosystem. Whether you are building robotics kinematics, computer graphics pipelines, quantum physics simulations, or high-throughput machine learning quantization kernels, anyone is invited to **fork, experiment, submit pull requests, and contribute**.

---

## Table of Contents
1. [Taxonomy of Clifford Geometric Algebra Operators](#taxonomy-of-clifford-geometric-algebra-operators)
2. [Canonical Literature & Theoretical Foundations](#canonical-literature--theoretical-foundations)
3. [Scientific Visual Gallery & Simulations](#scientific-visual-gallery--simulations)
4. [Mathematical Foundations & Signatures](#mathematical-foundations--signatures)
5. [Quickstart Guides (Rust & Python / NumPy)](#quickstart-guides)
6. [Hardware & Microarchitectural Benchmarks](#hardware--microarchitectural-benchmarks)
7. [Vulkan Compute Shaders](#vulkan-compute-shaders)
8. [Community, Contributing & Forking](#community-contributing--forking)
9. [Citation](#citation)
10. [License](#license)

---

## Taxonomy of Clifford Geometric Algebra Operators

`versorc-math` provides a complete, unified hierarchy of geometric operations across vector spaces and multivector grades:

| Operator Name | Canonical Symbol | Rust API | Python / NumPy API | Grade Shift ($\Delta\text{grade}$) | Mathematical Definition |
| :--- | :---: | :--- | :--- | :---: | :--- |
| **Geometric Product** | $A B$ | `a * b`, `a.geometric_product(&b)` | `a * b` | Mixed ($\sum \langle AB \rangle_k$) | $AB = A \cdot B + A \wedge B$ (vectors) |
| **Outer / Wedge Product** | $A \wedge B$ | `a.wedge(&b)` | `a ^ b` | Grade addition ($r + s$) | Multivector exterior product, anticommutative on 1-blades |
| **Left Contraction** | $A \rfloor B$ | `a.left_contraction(&b)` | `a \| b` | Grade subtraction ($s - r, r \le s$) | Contraction dual to wedge: $(A \wedge C) \cdot B = C \cdot (A \rfloor B)$ |
| **Hestenes Inner Product** | $A \cdot B$ | `a.hestenes_inner_product(&b)` | `a.hestenes_dot(b)` | Symmetric contraction ($\|r - s\|$) | $\frac{1}{2}(AB + BA)$ for 1-vectors |
| **Regressive / Meet Product** | $A \vee B$ | `a.regressive_product(&b)` | `a.regressive(b)` | Intersection blade ($r + s - n$) | $A \vee B = (A^* \wedge B^*)^{*-1}$ |
| **Hodge / Pseudoscalar Dual** | $*A = A I^{-1}$ | `a.dual()`, `a.undual()` | `a.dual()`, `a.undual()` | Orthogonal complement ($n - r$) | Multiplication by inverted pseudoscalar $I^{-1}$ |
| **Clifford Reversion** | $A^\sim$ / $A^\dagger$ | `a.reverse()` | `~a`, `a.reverse()` | Sign parity: $(-1)^{k(k-1)/2}$ | Blade order reversal: $(e_{i_1}\dots e_{i_k})^\sim = e_{i_k}\dots e_{i_1}$ |
| **Grade Involution** | $\widehat{A}$ | `a.grade_involution()` | `a.involution()` | Parity sign: $(-1)^k \langle A \rangle_k$ | Main automorphism: $\widehat{u v} = \widehat{u}\widehat{v}, \widehat{v} = -v$ |
| **Clifford Conjugation** | $\overline{A}$ | `a.conjugation()` | `a.conjugation()` | Composition: $(\widehat{A})^\sim$ | Combines main involution and reversion |
| **Versor Sandwich Product** | $V x V^{-1}$ | `v.sandwich(&x)` | `v.sandwich(x)` | Grade invariant automorphism | Outermorphism action $R x R^\sim$ on arbitrary blades |
| **Lie Commutator** | $A \times B = \frac{1}{2}[A, B]$ | `a.commutator(&b)` | `a.commutator(b)` | Lie bracket on bivectors | $\frac{1}{2}(AB - BA)$, satisfies Jacobi identity |
| **Anticommutator** | $\frac{1}{2}\{A, B\}$ | `a.anticommutator(&b)` | `a.anticommutator(b)` | Jordan algebra generator | $\frac{1}{2}(AB + BA)$ |
| **Conformal Spherical Inversion** | $P' = S P S^{-1}$ | `sphere.invert_in_sphere(&p)` | `sphere.invert_in_sphere(p)` | Radial inversion: $r' = R^2 / r$ | 1-versor sandwich across conformal sphere |
| **Möbius Gyrovector Addition** | $u \oplus v$ | `u.mobius_add(&v)` | `u.mobius_add(v)` | Non-associative gyrovector sum | $\frac{(1 + 2\langle u,v \rangle + \|v\|^2)u + (1 - \|u\|^2)v}{1 + 2\langle u,v \rangle + \|u\|^2 \|v\|^2}$ |
| **Einstein Hyperbolic Midpoint** | $\mu_E(u_1..u_K)$ | `PoincarePoint::einstein_midpoint(&pts, &w)` | `PoincarePoint.einstein_midpoint(pts, w)` | Conformal center of mass in $B^d$ | Projective weighted mean in Beltrami-Klein $\mathbb{K}^d$ |

> [!NOTE]
> For complete mathematical proofs, axioms, and grade shift derivations for each operator, see [explanations/02_clifford_operators_and_products.md](explanations/02_clifford_operators_and_products.md).

---

## Canonical Literature & Theoretical Foundations

Every algorithm and operator in `versorc-math` is strictly grounded in established literature:

* **Euclidean & Multivector Algebra ($\mathcal{Cl}_{3,0}, \mathcal{Cl}_{2,0}$)**: Hestenes & Sobczyk (1984), Doran & Lasenby (2003), Clifford (1878).
* **3D Projective Geometric Algebra ($\mathcal{Cl}^*_{3,0,1}$)**: Chasles (1830), Selig (2005), Dorst et al. (2007), Gunn (2019).
* **3D Conformal Geometric Algebra ($\mathcal{Cl}_{4,1}$)**: Dorst, Fontijne & Mann (2007), Hestenes (2001), Li et al. (2001).
* **4D Spacetime Algebra & Lounesto Spinors ($\mathcal{Cl}_{1,3}$)**: Lounesto (2001), Doran & Lasenby (2003), Rodrigues & Oliveira (2007).
* **Gosset $E_8$ Root Lattice & Fast Decoding**: Conway & Sloane (1999), Coxeter (1973), Bourbaki (1968).
* **TurboQuant & Dyadic Versors**: Max (1960), Lloyd (1982), Walsh (1923), Ailon & Chazelle (2006).
* **Conformal Hyperbolic Geometry ($\mathbb{B}^d$)**: Ungar (2008), Ratcliffe (2019), Beltrami (1868), Klein (1871).

> [!TIP]
> An exhaustive academic bibliography with detailed theorem citations and source mapping is cataloged in [explanations/08_canonical_literature_and_theoretical_foundations.md](explanations/08_canonical_literature_and_theoretical_foundations.md).

---

## Scientific Visual Gallery & Simulations

All visual representations below are produced directly by the engine via `python3 generate_visual_plots.py`:

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

## Mathematical Foundations & Signatures

In Clifford Geometric Algebra, vector multiplication is governed by the geometric product:

$$u v = u \cdot v + u \wedge v$$

where $u \cdot v = \frac{1}{2}(uv + vu)$ is the symmetric inner contraction and $u \wedge v = \frac{1}{2}(uv - vu)$ is the antisymmetric outer wedge product.

| Signature | Dimension | Common Name | Typical Applications |
| :--- | :---: | :--- | :--- |
| $\mathcal{Cl}(2, 0)$ | $2^2 = 4$ | Planar GA | Complex numbers ($I^2 = -1$), 2D kinematics, conformal mappings |
| $\mathcal{Cl}(3, 0)$ | $2^3 = 8$ | Spatial 3D GA | Hamilton quaternions ($\mathbb{H} \cong \mathcal{Cl}^+(3,0)$), rigid body dynamics, optics |
| $\mathcal{Cl}^*(3, 0, 1)$ | $2^4 = 16$ | 3D PGA | SE(3) robotics, screw theory, Plücker lines, dual quaternions |
| $\mathcal{Cl}(4, 1)$ | $2^5 = 32$ | 3D CGA | Conformal transformations, spheres, circles, inversions |
| $\mathcal{Cl}(1, 3)$ | $2^4 = 16$ | 4D STA | Electrodynamics, Maxwell bivectors, Dirac equation, Lounesto spinors |
| $\mathbb{R}^8 \supset E_8$ | 240 roots | Gosset Lattice | Optimal sphere packing in 8D, Conway-Sloane closest-vector decoding |

---

## Quickstart Guides

### 1. Rust Integration

Add to your `Cargo.toml`:

```toml
[dependencies]
versorc-math = "0.1.0"

# Optional features:
# versorc-math = { version = "0.1.0", features = ["vulkan", "python"] }
```

Basic Rust example:

```rust
use versorc_math::prelude::*;

fn main() {
    // 3D Euclidean Geometric Algebra Cl(3, 0)
    let e1 = Cl3::basis(1);
    let e2 = Cl3::basis(2);
    let bivector = e1.wedge(&e2);
    let dual = bivector.dual().unwrap();
    println!("Dual of e1 ^ e2: {:?}", dual);

    // 3D Projective Geometric Algebra Cl*(3, 0, 1) Motor
    let motor = PgaMotor::from_rotation_translation(&[0.0, 0.0, 1.0], std::f32::consts::PI, &[2.0, 1.0, 2.0]);
    let pt = PgaPoint::new(0.0, 0.0, 0.0);
    let transformed = motor.transform_point(&pt);
    println!("Transformed Point: {:?}", transformed.to_euclidean());
}
```

### 2. Python / NumPy Integration

Install via pip:

```bash
pip install versorc-math
```

Or build from source with Maturin:

```bash
maturin develop --release --features python
```

Interactive Python example:

```python
import numpy as np
from versorc_math import Cl3, PgaMotor, CgaPoint, CgaSphere, StaSpinor, PoincarePoint

# 3D Euclidean Cl(3, 0)
e1, e2, e3 = Cl3.e1(), Cl3.e2(), Cl3.e3()
e12 = e1 ^ e2
dual = e12.dual()

# Invertible multivector
mv = Cl3.from_scalar(2.0) + Cl3.from_vector(1.0, -1.0, 3.0)
inv_mv = mv.inverse()

# Relativistic Dirac Spinor (Lounesto Class 2)
psi = StaSpinor(scalar=1.0, pseudoscalar=0.0)
print("Spinor Class:", psi.lounesto_class())
```

---

## Hardware & Microarchitectural Benchmarks

In large Transformer projection layers ($W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$), outlier channels degrade naive low-bit quantization. `versorc-math` applies an isotropic Dyadic Walsh Versor rotation:

$$\mathbf{W}_{\text{rot}} = \mathbf{W} \mathbf{H}_D, \quad \mathbf{x}_{\text{rot}} = \mathbf{x} \mathbf{H}_D$$

Because $\mathbf{H}_D \mathbf{H}_D^T = \mathbf{I}$, inner products are strictly preserved: $\langle \mathbf{W}_i, \mathbf{x} \rangle = \langle \mathbf{W}_i \mathbf{H}_D, \mathbf{x} \mathbf{H}_D \rangle$.

### Silicon Benchmark (1408 × 2048 Expert Projection Layer)

Evaluated on x86_64 AVX2 host:

| Representation | Precision | Footprint | Throughput | Speedup | Cosine vs FP32 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Dense FP32** | 32-bit | 11.00 MB | 28.4 GFLOPS | 1.00× (baseline) | 1.00000 |
| **VrcAsym64 (v5)** | 4.50 bpw | 1.55 MB | 54.1 GFLOPS | 1.90× | 0.99929 |
| **Clifford 3-Bit (Scalar)** | 4.00 bpw | 1.38 MB | 12.2 GFLOPS | 0.43× (ALU bound) | 0.99620 |
| **Clifford 3-Bit (AVX2 `vpermps`)** | 4.00 bpw | 1.38 MB | **198.6 GFLOPS** | **7.00×** | **0.99620** |

Run benchmarks locally:
```bash
cargo test --test test_risk_audit_silicon -- --nocapture
cargo test --test test_quantization_static_benchmark -- --nocapture
```

---

## Vulkan Compute Shaders

`versorc-math` embeds precompiled SPIR-V 1.3+ compute shaders via `include_bytes!`:
- `pga_transform.spv`: Rigid tensor point transformation $p' = M p M^\sim$ on arbitrary batches in GPU VRAM.
- `cga_transform.spv`: Conformal transformations (spherical inversions, dilations, translations).
- `hot_stratum_e8.spv`: Gosset lattice $E_8$ projection with Wave32/Wave64 subgroup optimizations.
- `rope.spv`, `silu_mul.spv`, `rmsnorm.spv`, `vec_add.spv`: High-performance Transformer building blocks.

Verify Vulkan integration on your GPU:
```bash
cargo test --features vulkan --test test_vulkan_compute -- --nocapture
```

---

## Community, Contributing & Forking

`versorc-math` is an open-source initiative built by and for the global scientific and engineering community. We welcome contributions from researchers, mathematicians, engineers, educators, and students of all backgrounds!

* **Contributing Guide**: Check out [CONTRIBUTING.md](CONTRIBUTING.md) for full setup instructions, development workflow, and testing guides.
* **Code of Conduct**: We are committed to fostering a welcoming, harassment-free environment for everyone. Please review our [Code of Conduct](CODE_OF_CONDUCT.md).
* **Good First Issues**: Whether fixing a typo in theoretical proofs, adding a test case, or optimizing a kernel, all pull requests are valued and reviewed with care.

### How You Can Contribute:
- **New Geometric Algebras**: Implement $\mathcal{Cl}(0, 2)$, Conformal 2D, or Compass Geometric Algebra signatures.
- **Hardware SIMD Kernels**: Contribute ARM NEON, AVX-512, or WebAssembly SIMD128 implementations.
- **Python / Framework Integrations**: Build PyTorch, JAX, or Taichi bindings for Clifford tensor operations.
- **Documentation & Educational Visualizations**: Enhance derivations, Jupyter notebooks, or GanJa.js interactive scenes.

### Contribution Workflow:
1. **Fork the Repository**: Click "Fork" on [GitHub](https://github.com/versorc/versorc-math).
2. **Create a Feature Branch**:
   ```bash
   git checkout -b feature/my-geometric-feature
   ```
3. **Ensure All Tests Pass**:
   ```bash
   cargo test --all-features
   pytest python/tests/test_ga_numpy.py -v
   ```
4. **Submit a Pull Request**: Open a PR describing your mathematical rationale and test coverage. Our maintainers will gladly assist with reviews!

---

## Citation

If you utilize `versorc-math` in your research, academic publications, or open-source software, please cite:

```bibtex
@software{versorc_math_2026,
  author       = {{The VersorC Project Contributors}},
  title        = {versorc-math: High-Performance Clifford Geometric Algebra and Versor Quantization Engine},
  year         = 2026,
  publisher    = {Zenodo},
  version      = {0.1.0},
  doi          = {10.5281/zenodo.24153329},
  url          = {https://github.com/versorc/versorc-math}
}
```

---

## License

Licensed under the **Apache License, Version 2.0** ([LICENSE](LICENSE) or http://www.apache.org/licenses/LICENSE-2.0).

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
