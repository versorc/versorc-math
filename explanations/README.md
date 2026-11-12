# VersorC Math (`versorc-math`) — Architectural Overview & Documentation Map

Welcome to **VersorC Math (`versorc-math`)**, a low-level, high-performance Clifford Geometric Algebra and Versor Operations library engineered in pure Rust with zero external runtime dependencies.

---

## 1. Mission & Purpose

`versorc-math` provides the mathematical foundation and low-level computational primitives for:
1. **Clifford Geometric Algebra ($\mathcal{Cl}_{p,q,r}$)**: Exact multivector representations, metric signatures, blade bitmask indexing, geometric products, wedge products, directional contractions, and the complete 15-operator taxonomy.
2. **The Versor Group ($\mathbf{\Gamma}_{p,q}$)**: Hamilton's classical quaternions $\mathbb{H}$, bivector exponential rotors $R = \exp(-B/2)$, and sandwich transformations $V x \widetilde{V}$.
3. **High-Dimensional Linear Algebra Reduction**: Overcoming the $2^n$ dimension limit of general multivectors by factorizing dense $O(n^2)$ matrices into:
   - **1-Versors (Householder Cascades):** $O(n)$ compute and storage.
   - **2-Versors (Planar Spinor Banks):** Multi-frequency orthogonal phase rotations ($C\ell$-RoPE).
   - **Dyadic Versors (Walsh-Hadamard IP-RHT):** Incoherence outlier smearing in $O(n \log n)$ additions without floating-point multiplications.
4. **Root Lattices & Vector Quantization**: The 8-dimensional Gosset $E_8$ lattice (all 240 minimal root versors) and Conway & Sloane's linear-time closest-vector projection algorithm.
5. **Conformal Hyperbolic Geometry**: Poincaré ball $\mathbb{B}^d$ geodesics, Möbius gyrovector addition, and Klein manifold projections.

---

## 2. Documentation Map

```
explanations/
├── README.md                                                 # Architectural Master Map (This Document)
├── 01_geometric_algebra_and_versors.md                       # Clifford Algebras Cl_{p,q,r}, Lipschitz Group, Hamilton Quaternions
├── 02_clifford_operators_and_products.md                      # Unified 15-Operator Taxonomy, Derivations, and Involutions
├── 03_high_dimensional_versor_reductions.md                  # Householder Cascades, Planar Spinor Banks, Dyadic Walsh-Hadamard
├── 04_lattice_e8_and_conway_sloane.md                        # 240 Minimal Roots of E8, D8 Cosets, Conway-Sloane O(8) Decoder
├── 05_verification_and_ground_truth_test_battery.md          # Ground-Truth Fixtures, Algebraic Invariants & Tolerances
├── 06_clifford_3bit_turboquant_specification.md              # Hardware 64-Byte Cache Line Packing & Lloyd-Max Centroids
├── 07_specialized_hardware_kernels_and_design.md             # AVX2/AVX-512 vpermps Streaming and Vulkan Compute Shaders
└── 08_canonical_literature_and_theoretical_foundations.md    # Exhaustive Academic Bibliography, Theorems & Primary Sources
```

---

## 3. Crate Architecture

```
00_versorc-math/
├── Cargo.toml
├── explanations/          # In-depth theoretical documentation and operator derivations
├── vendor_reference/      # Permissive (Apache-2.0 / MIT) reference implementations
├── src/
│   ├── lib.rs             # Crate root and prelude
│   ├── signature.rs       # Const-generic metric signatures (p, q, r)
│   ├── blade.rs           # Bitmask blade indices and Cayley sign tables
│   ├── multivector.rs     # Exact multivectors for low/medium dimensions
│   ├── pga/               # 3D Projective Geometric Algebra Cl*(3,0,1)
│   ├── cga/               # 3D Conformal Geometric Algebra Cl(4,1)
│   ├── sta/               # 4D Spacetime Algebra Cl(1,3) & Lounesto Spinors
│   ├── versor/
│   │   ├── mod.rs         # VersorAction trait and sandwich operators
│   │   ├── hamilton.rs    # Hamilton quaternions H, Slerp, S^3 geodesics
│   │   ├── rotor.rs       # Exponential planar rotors R = exp(-B/2)
│   │   ├── householder.rs # 1-Versor reflection operators and cascades
│   │   ├── dyadic.rs      # Fast Walsh-Hadamard Transform (FWHT) in O(N log N)
│   │   └── planar_bank.rs # Multi-frequency 2-versor spinor banks (RoPE)
│   ├── lattice/
│   │   ├── mod.rs
│   │   └── e8.rs          # 240 minimal roots and Conway-Sloane decoder
│   ├── poincare/
│   │   └── mod.rs         # Hyperbolic geometry in Poincare Ball B^d
│   ├── quantization/
│   │   └── clifford_3bit.rs # Cache-line aligned 3-bit TurboQuant
│   └── reduction/
│       ├── mod.rs
│       └── cartan_dieudonne.rs # Orthogonal matrix factorizer into 1-versores
├── shaders/               # Precompiled SPIR-V 1.3+ Vulkan compute kernels
└── tests/                 # Integration test battery (58 ground-truth test suites)
```
