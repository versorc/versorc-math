# Chapter 08: Canonical Mathematical Literature & Theoretical Foundations

This document provides the exhaustive theoretical mapping, historical bibliography, and mathematical derivations connecting each module of `versorc-math` to canonical academic literature.

---

## 1. Euclidean Space $\mathcal{Cl}(3,0)$ & Planar Algebra $\mathcal{Cl}(2,0)$

### 1.1 Academic References
* **Hestenes, D., Sobczyk, G.** (1984). *Clifford Algebra to Geometric Calculus: A Unified Language for Mathematics and Physics*. Fundamental Theories of Physics, Vol. 1. D. Reidel Publishing Company, Dordrecht / Boston / Lancaster.
  - *Chapter 1*: Multivectors and Axioms (§1.1–§1.4).
  - *Chapter 2*: Geometric Calculus and Grade Projections.
* **Doran, C., Lasenby, A.** (2003). *Geometric Algebra for Physicists*. Cambridge University Press, Cambridge, UK.
  - *Chapter 2*: Geometric Algebra Fundamentals, multivector products, and Hodge duality.
* **Clifford, W.K.** (1878). *Applications of Grassmann's Extensive Algebra*. American Journal of Mathematics, 1(4):350–358.

### 1.2 Mathematical Theorems & Implementations
In `versorc-math` ([src/multivector.rs](src/multivector.rs)):
* **Axiomatic Geometric Product**: For $u, v \in V$, $uv = u \cdot v + u \wedge v$.
* **Determinant & Exact Multivector Inverses**: For $\mathcal{Cl}(3,0)$, the multivector $M = \langle M \rangle_0 + \langle M \rangle_1 + \langle M \rangle_2 + \langle M \rangle_3$ possesses a closed-form analytical inverse:
  $$M^{-1} = \frac{M^\dagger \overline{M} \widehat{M}}{\det(M)}$$
  which avoids numerical linear system solving.
* **Grade Projections**: Exact $k$-blade extraction $\langle M \rangle_k$ implemented via compile-time bitmask manipulation.

---

## 2. 3D Projective Geometric Algebra (PGA) $\mathcal{Cl}^*(3,0,1)$

### 2.1 Academic References
* **Chasles, M.** (1830). *Note sur les propriétés générales du système de deux corps semblables entre eux et placés d'une manière quelconque dans l'espace*. Bulletin des Sciences Mathématiques, Physiques et Chimiques, 14:321–326.
* **Selig, J.M.** (2005). *Geometric Fundamentals of Robotics*. Monographs in Computer Science. Springer, New York, 2nd Edition.
  - *Chapter 9*: Lie Groups and Screw Theory.
  - *Chapter 11*: Dual Quaternions and Rigid Body Displacements.
* **Dorst, L., Fontijne, D., Mann, S.** (2007). *Geometric Algebra for Computer Science: An Object-Oriented Approach to Geometry*. Morgan Kaufmann / Elsevier.
  - *Chapter 11*: 3D Projective Geometric Algebra.
* **Gunn, C.** (2019). *Geometric Algebra for Computer Graphics: Plane-based Projective Geometric Algebra*. IEEE Computer Graphics and Applications.

### 2.2 Mathematical Theorems & Implementations
In `versorc-math` ([src/pga/](src/pga/)):
* **Dual-Quaternion Isomorphism**: The even subalgebra $\mathcal{Cl}^+_{3,0,1}$ is naturally isomorphic to Clifford biquaternions $\mathbb{H} \otimes \mathbb{D}$ ($\epsilon^2 = 0$).
* **Chasles' Theorem**: Every spatial rigid motion $M \in \text{Spin}(3,0,1)$ decomposes into an invariant screw axis $L$ with translation pitch $p$ and rotation angle $\theta$:
  $$M = \exp\left(-\frac{\theta + \epsilon d}{2} L\right)$$
* **ScLERP (Screw Linear Interpolation)**: Continuous constant-speed geodesic trajectory interpolation on the $\text{SE}(3)$ manifold:
  $$M(t) = (M_1 M_0^{-1})^t M_0 = \exp(t \log(M_1 M_0^{-1})) M_0$$

---

## 3. 3D Conformal Geometric Algebra (CGA) $\mathcal{Cl}(4,1)$

### 3.1 Academic References
* **Dorst, L., Fontijne, D., Mann, S.** (2007). *Geometric Algebra for Computer Science*. Morgan Kaufmann.
  - *Chapter 13*: Conformal Geometric Algebra.
  - *Chapter 14*: Conformal Transformations and Inversions.
  - *Chapter 15*: Conformal Intersections and the Meet Operator.
* **Hestenes, D.** (2001). *Old Wine in New Bottles: A New Algebraic Framework for Computational Geometry*. In *Geometric Algebra with Applications in Science and Engineering*, Birkhäuser Boston, pp. 3–17.
* **Li, H., Hestenes, D., Rockwood, A.** (2001). *Generalized Homogeneous Coordinates for Computational Geometry*. In *Geometric Computing with Clifford Algebras*, Springer, pp. 27–52.

### 3.2 Mathematical Theorems & Implementations
In `versorc-math` ([src/cga/mod.rs](src/cga/mod.rs)):
* **Minkowski Null Basis**:
  $$e_+^2 = +1, \quad e_-^2 = -1, \quad n_o = \frac{1}{2}(e_- - e_+), \quad n_\infty = e_- + e_+$$
  with properties $n_o^2 = n_\infty^2 = 0$ and $n_o \cdot n_\infty = -1$.
* **Geometric Objects as Blades**:
  - Point: $P = \mathbf{x} + \frac{1}{2}\|\mathbf{x}\|^2 n_\infty + n_o$
  - Sphere: $S = \mathbf{c} + \frac{1}{2}(\|\mathbf{c}\|^2 - R^2) n_\infty + n_o$
  - Plane: $\Pi = \mathbf{n} + d\, n_\infty$
  - Circle: $C^* = S^* \wedge \Pi^*$ (meet bivector of dual representations)
* **Conformal Spherical Inversion**: Inverting across sphere $S$ is computed through a 1-versor sandwich product:
  $$P' = S P S^{-1}$$
  mapping $\|\mathbf{p}'\| = R^2 / \|\mathbf{p}\|$ with conformal angular invariance.

---

## 4. 4D Spacetime Algebra (STA) $\mathcal{Cl}(1,3)$ & Lounesto Spinors

### 4.1 Academic References
* **Lounesto, P.** (2001). *Clifford Algebras and Spinors*. London Mathematical Society Lecture Note Series 286. Cambridge University Press, 2nd Edition.
  - *Chapter 12*: Spinor Fields and the Bilinear Covariants.
  - *Chapter 13*: The Lounesto Spinor Classification Scheme.
* **Doran, C., Lasenby, A.** (2003). *Geometric Algebra for Physicists*. Cambridge University Press.
  - *Chapter 8*: Spacetime Algebra and Relativistic Quantum Mechanics.
* **Rodrigues, W.A., Oliveira, E.C.** (2007). *The Many Faces of Maxwell, Dirac and Einstein Equations: A Clifford Bundle Approach*. Lecture Notes in Physics, Vol. 714. Springer-Verlag, Berlin / Heidelberg.

### 4.2 Mathematical Theorems & Implementations
In `versorc-math` ([src/sta/mod.rs](src/sta/mod.rs)):
* **Bilinear Covariants**: For any Dirac spinor $\psi \in \mathcal{Cl}(1,3)$, five physical currents are generated:
  1. Scalar current: $\sigma = \langle \psi \widetilde{\psi} \rangle_0$
  2. Pseudoscalar current: $\omega = \langle \psi \widetilde{\psi} \rangle_4$
  3. Vector current: $J = \psi \gamma_0 \widetilde{\psi} \in \langle \mathcal{Cl} \rangle_1$
  4. Spin bivector current: $S = \psi \gamma_{12} \widetilde{\psi} \in \langle \mathcal{Cl} \rangle_2$
  5. Chiral axial vector: $K = \psi \gamma_3 \widetilde{\psi} \in \langle \mathcal{Cl} \rangle_3$
* **Lounesto Spinor Classification**: Evaluates algebraic constraints on $(\sigma, \omega, S)$ to deterministically classify any spinor into:
  - **Class 1 (Regular Dirac Electron)**: $\sigma \neq 0$, $\omega \neq 0$, $S \neq 0$.
  - **Class 2 (Dirac Pseudoscalar)**: $\sigma = 0$, $\omega \neq 0$, $S \neq 0$.
  - **Class 3 (Dirac Scalar)**: $\sigma \neq 0$, $\omega = 0$, $S \neq 0$.
  - **Class 4 (Flag-Dipole Spinor)**: $\sigma = \omega = 0$, $S \neq 0$, $K \neq 0$.
  - **Class 5 (Majorana Flag-Pole Spinor)**: $\sigma = \omega = 0$, $S \neq 0$, $K = 0$.
  - **Class 6 (Weyl Dipole Spinor)**: $\sigma = \omega = S = 0$.

---

## 5. Gosset $E_8$ Root Lattice & Conway-Sloane Fast Decoder

### 5.1 Academic References
* **Conway, J.H., Sloane, N.J.A.** (1999). *Sphere Packings, Lattices and Groups*. Grundlehren der mathematischen Wissenschaften 290. Springer-Verlag, New York, 3rd Edition.
  - *Chapter 8*: The Gosset Lattice $E_8$ and Leech Lattice $\Lambda_{24}$.
  - *Chapter 20*: Fast Quantizing and Decoding Algorithms for Gosset and Leech Lattices.
* **Coxeter, H.S.M.** (1973). *Regular Polytopes*. Dover Publications, New York, 3rd Edition.
  - *Chapter 11*: The Gosset 8-Polytope $4_{21}$.
  - *Chapter 12*: The Coxeter Plane and Dihedral Projections.
* **Bourbaki, N.** (1968). *Groupes et algèbres de Lie: Chapitres 4, 5 et 6*. Hermann, Paris.
  - *Planche VII*: Système de racines de type $E_8$.
* **Moody, R.V., Patera, J.** (1993). *Voronoi and Delaunay cells of root lattices: Classification of their facets and vertices by Dynkin diagrams*. Journal of Physics A: Mathematical and General, 26(18):4581.

### 5.2 Mathematical Theorems & Implementations
In `versorc-math` ([src/lattice/e8.rs](src/lattice/e8.rs)):
* **Minimal Root System**: The 240 root versors in $\mathbb{R}^8$ with squared norm $\|r\|^2 = 2.0$ are constructed as:
  $$E_8 = D_8 \cup \left( D_8 + \frac{1}{2} \mathbf{1} \right)$$
  - 112 integer roots: $(\pm 1, \pm 1, 0, 0, 0, 0, 0, 0)$ with even sign changes.
  - 128 half-integer roots: $\left( \pm \frac{1}{2}, \dots, \pm \frac{1}{2} \right)$ with an even number of minus signs.
* **Coxeter Plane Projection**: Projecting $E_8$ roots onto the 2D invariant eigenspace of the Coxeter element $\lambda = \exp(\pm 2\pi i / 30)$ produces exactly 8 concentric rings of 30 vertices with 30-fold rotational dihedral symmetry $D_{30}$.
* **Conway-Sloane $O(8)$ Linear Decoder**: Fast nearest-lattice point quantization:
  $$\text{decode}_{E_8}(x) = \arg\min_{y \in \{f(x, D_8), f(x - \frac{1}{2}\mathbf{1}, D_8) + \frac{1}{2}\mathbf{1}\}} \|x - y\|^2$$
  where $f(x, D_8)$ rounds to the nearest even-parity integer vector in $O(8)$ operations without table lookups.

---

## 6. TurboQuant: Dyadic Versor Isometry & Lloyd-Max Centroids

### 6.1 Academic References
* **Max, J.** (1960). *Quantizing for Minimum Distortion*. IRE Transactions on Information Theory, 6(1):7–12.
* **Lloyd, S.** (1982). *Least Squares Quantization in PCM*. IEEE Transactions on Information Theory, 28(2):129–137.
* **Walsh, J.L.** (1923). *A Closed Set of Normal Orthogonal Functions*. American Journal of Mathematics, 45(1):5–24.
* **Ailon, N., Chazelle, B.** (2006). *Approximate Nearest Neighbors and the Fast Johnson-Lindenstrauss Transform*. In Proceedings of the 38th Annual ACM Symposium on Theory of Computing (STOC '06), pp. 557–563.

### 6.2 Mathematical Theorems & Implementations
In `versorc-math` ([src/quantization/clifford_3bit.rs](src/quantization/clifford_3bit.rs) & [src/versor/dyadic.rs](src/versor/dyadic.rs)):
* **Central Limit Theorem & Outlier Smearing**:
  Applying a randomized Fast Walsh-Hadamard Transform (FWHT) versor rotation:
  $$W_{\text{rot}} = W \mathbf{H}_D, \quad \mathbf{H}_D = \frac{1}{\sqrt{N}} H_N \mathbf{D}$$
  where $\mathbf{D} = \text{diag}(\pm 1)$ is a pseudo-random Rademacher sign matrix, maps high-kurtosis activation distributions into isotropic Gaussian distributions $\mathcal{N}(0, \sigma^2)$ in $O(N \log N)$ additions without multiplications.
* **Optimal 3-Bit Lloyd-Max Centroids for $\mathcal{N}(0, 1)$**:
  Precomputed zero-bias quantizer levels minimizing mean-squared error $\mathbb{E}[(X - Q(X))^2]$:
  $$\mathcal{C}_{3\text{-bit}} = \{-2.152, -1.344, -0.756, -0.245, +0.245, +0.756, +1.344, +2.152\}$$
* **Hardware Cache Line Packing**:
  Packs 128 quantized weights (3 bits each) plus FP16 scale and offset into exactly 64 bytes (`#[repr(C, align(64))]`), guaranteeing zero cache line straddling and optimal AVX2/AVX-512 `vpermps` register streaming.

---

## 7. Conformal Hyperbolic Geometry & Poincaré Ball

### 7.1 Academic References
* **Ungar, A.A.** (2008). *Analytic Hyperbolic Geometry and Albert Einstein's Special Theory of Relativity*. World Scientific Publishing Co., Singapore.
  - *Chapter 2*: Gyrogroups and Möbius Gyrovector Spaces.
  - *Chapter 6*: Hyperbolic Midpoints and Barycentric Coordinates.
* **Ratcliffe, J.G.** (2019). *Foundations of Hyperbolic Manifolds*. Graduate Texts in Mathematics 149. Springer Nature, Cham, Switzerland, 3rd Edition.
  - *Chapter 4*: The Poincaré Ball Model and Conformal Isometries.
* **Beltrami, E.** (1868). *Saggio di interpretazione della geometria non-euclidea*. Giornale di Matematiche, 6:284–312.
* **Klein, F.** (1871). *Über die sogenannte Nicht-Euklidische Geometrie*. Mathematische Annalen, 4(4):573–625.

### 7.2 Mathematical Theorems & Implementations
In `versorc-math` ([src/poincare/mod.rs](src/poincare/mod.rs)):
* **Möbius Gyrovector Addition**: In the open ball $\mathbb{B}^d = \{x \in \mathbb{R}^d : \|x\| < 1\}$:
  $$u \oplus v = \frac{(1 + 2\langle u, v \rangle + \|v\|^2) u + (1 - \|u\|^2) v}{1 + 2\langle u, v \rangle + \|u\|^2 \|v\|^2}$$
  Satisfies the gyrocommutative and gyroassociative axioms under Thomas precession gyration $\text{gyr}[u, v]$.
* **Poincaré Geodesic Distance**:
  $$d_{\mathbb{B}}(u, v) = \text{arcosh}\left(1 + \frac{2\|u - v\|^2}{(1 - \|u\|^2)(1 - \|v\|^2)}\right) = 2 \text{artanh}(\|-u \oplus v\|)$$
* **Einstein Hyperbolic Midpoint $\mu_E$**:
  Computed via projective correspondence to the Beltrami-Klein model $\mathbb{K}^d$:
  $$x_K = \frac{2 x_P}{1 + \|x_P\|^2}, \quad x_P = \frac{x_K}{1 + \sqrt{1 - \|x_K\|^2}}$$
  The weighted center of mass in Klein coordinates is linear:
  $$\mathbf{m}_K = \frac{\sum_{i=1}^K w_i \gamma_i \mathbf{x}_{K, i}}{\sum_{i=1}^K w_i \gamma_i}$$
  with Lorentz factors $\gamma_i = \frac{1}{\sqrt{1 - \|\mathbf{x}_{K, i}\|^2}}$, mapped back to $\mathbb{B}^d$ in $O(K d)$ operations.
