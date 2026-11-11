# VersorC Math (`versorc-math`)

Low-level, high-performance Clifford Geometric Algebra, Hamilton Versors, and High-Dimensional Matrix Reductions in pure Rust.

## Features
- **Arbitrary Metric Signatures $\mathcal{Cl}_{p,q,r}$**: Const-generic signatures for Euclidean, Minkowski, and Projective (PGA) algebras.
- **Complete Clifford Operators**: Geometric product, exterior wedge $\wedge$, left/right contractions $\rfloor, \lfloor$, reversions $\widetilde{A}$, grade involutions $\widehat{A}$, and Clifford conjugations $\overline{A}$.
- **Versor Group $\mathbf{\Gamma}_{p,q}$**:
  - Hamilton Quaternions ($\mathbb{H}$) with $S^3$ Slerp geodesic interpolation.
  - Planar Rotors $R = \exp(-B/2)$ and exact sandwich transformations $V x \widetilde{V}$.
  - 1-Versors (Householder Reflection Cascades) reducing $O(n^2)$ matrix transforms to $O(kn)$ operations.
  - 2-Versors (Orthogonal Planar Spinor Banks) generalizing multi-frequency RoPE.
  - Dyadic Walsh-Hadamard Versors in $O(n \log n)$ additions (zero multiplications).
- **Gosset $E_8$ Root Lattice**:
  - Exact 240 minimal root generation (112 integer roots + 128 half-integer roots).
  - Conway & Sloane linear-time $O(8)$ closest-vector decoding.
- **Conformal Hyperbolic Geometry**:
  - Poincaré Ball $\mathbb{B}^d$ geodesics and Möbius gyrovector addition.
- **Zero Allocations & Hardware SIMD**: In-place transformations with AVX2 vectorization.
