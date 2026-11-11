# Chapter 05: Verification & Ground-Truth Test Battery

In accordance with strict empirical rigor, `versorc-math` mandates comprehensive, automated test fixtures. Every algebraic operator, metric property, and high-dimensional reduction is tested against known analytical identities and numerical references.

---

## 1. Test Invariants & Tolerance Thresholds

| Invariant Category | Target Tolerance ($\epsilon$) | Analytical Identity |
| :--- | :---: | :--- |
| **Clifford Associativity** | $1.0 \times 10^{-7}$ (f32) / $1.0 \times 10^{-14}$ (f64) | $(A B) C = A (B C)$ |
| **Clifford Basis Squares** | Exactly $0.0$ | $e_i^2 = \eta_{ii} \in \{-1, 0, +1\}$ |
| **Basis Anticommutativity** | Exactly $0.0$ | $e_i e_j + e_j e_i = 0 \;\; (\forall i \ne j)$ |
| **Hamilton Quaternions** | Exactly $0.0$ | $\mathbf{i}^2 = \mathbf{j}^2 = \mathbf{k}^2 = \mathbf{i}\mathbf{j}\mathbf{k} = -1$ |
| **Versor Isometry (Norm)** | $1.0 \times 10^{-6}$ | $\|V x \widetilde{V}\| = \|x\|$ |
| **Versor Distance Preservation** | $1.0 \times 10^{-6}$ | $\|V x \widetilde{V} - V y \widetilde{V}\| = \|x - y\|$ |
| **Householder Reflection Invariant** | $1.0 \times 10^{-6}$ | $\mathcal{H}_v(\mathcal{H}_v(x)) = x$ (Involution) |
| **Dyadic Hadamard Unitarity** | $1.0 \times 10^{-6}$ | $\frac{1}{n} H_n H_n^T = I_n$ |
| **Parseval Energy Conservation** | $1.0 \times 10^{-6}$ | $\|H_n x\|_2 = \sqrt{n} \|x\|_2$ |
| **$E_8$ Root Norm** | $1.0 \times 10^{-6}$ | $\|r\|^2 = 2.0 \;\; (\forall r \in \text{Roots}(E_8))$ |
| **$E_8$ Root Count** | Exact integer | $\text{count} = 240$ (112 integer + 128 half-integer) |

---

## 2. Ground-Truth Test Matrix

The test suite in `tests/` executes the following targeted suites:

1. **`test_clifford_axioms.rs`**:
   - Tests geometric product, wedge product, left and right contractions, reversions, grade involutions, and grade projections for $\mathcal{Cl}_{2,0}$, $\mathcal{Cl}_{3,0}$, $\mathcal{Cl}_{0,2}$, and Minkowski $\mathcal{Cl}_{1,3}$.
2. **`test_hamilton_quaternions.rs`**:
   - Tests Hamilton multiplication table, non-commutativity $\mathbf{i}\mathbf{j} = -\mathbf{j}\mathbf{i} = \mathbf{k}$, unit quaternion normalization, vector rotation, and spherical linear interpolation (`slerp`) along the geodesic on $S^3$.
3. **`test_versor_sandwich.rs`**:
   - Tests composition of rotors: $(R_2 R_1) x (R_2 R_1)^\sim = R_2 (R_1 x \widetilde{R}_1) \widetilde{R}_2$.
   - Confirms angle and norm invariance under 10,000 randomized 3D/4D test vectors.
4. **`test_householder_cascade.rs`**:
   - Decomposes a random orthogonal $8 \times 8$ and $64 \times 64$ matrix into Householder 1-versores and asserts numerical equivalence against standard matrix multiplication.
5. **`test_e8_lattice_roots.rs`**:
   - Enumerates all 240 minimal vectors of $E_8$.
   - Asserts even coordinate sum for $D_8$ components and even negative-sign count for $(D_8 + \frac{1}{2}\mathbf{1})$ components.
   - Tests Conway-Sloane closest-vector decoding against perturbation vectors.
6. **`test_dyadic_walsh.rs`**:
   - Asserts Fast Walsh-Hadamard Transform (FWHT) numerical equivalence between scalar and AVX2 vector paths.
   - Confirms zero allocation in in-place transformation.
7. **`test_poincare_hyperbolic.rs`**:
   - Tests Möbius gyrovector addition $u \oplus_c v$, confirming gyrocommutativity $u \oplus v = \operatorname{gyr}[u, v](v \oplus u)$, boundary projection, and hyperbolic distance triangle inequality.
