# Chapter 02: Clifford Operators and Products

This chapter documents the operator taxonomy and algebraic contracts implemented in `versorc-math`. Every operator is defined with mathematical exactness and mapped directly to hardware-efficient algorithmic implementations.

---

## 1. Unified Taxonomy of the 15 GA Operators

`versorc-math` provides a comprehensive, mathematically closed operator suite spanning all multivector grades:

| Operator Name | Canonical Symbol | Rust API | Python / NumPy API | Grade Shift ($\Delta\text{grade}$) | Mathematical Definition | Canonical Reference |
| :--- | :---: | :--- | :--- | :---: | :--- | :--- |
| **Geometric Product** | $A B$ | `a * b`, `a.geometric_product(&b)` | `a * b` | Mixed ($\sum \langle AB \rangle_k$) | $AB = A \cdot B + A \wedge B$ (vectors) | Clifford (1878), Hestenes (1984) §1.1 |
| **Outer / Wedge Product** | $A \wedge B$ | `a.wedge(&b)` | `a ^ b` | Grade addition ($r + s$) | Multivector exterior product, anticommutative on 1-blades | Grassmann (1844), Hestenes §1.2 |
| **Left Contraction** | $A \rfloor B$ | `a.left_contraction(&b)` | `a \| b` | Grade subtraction ($s - r, r \le s$) | Contraction dual to wedge: $(A \wedge C) \cdot B = C \cdot (A \rfloor B)$ | Lounesto (2001) §3.1, Dorst (2007) §4.2 |
| **Hestenes Inner Product** | $A \cdot B$ | `a.hestenes_inner_product(&b)` | `a.hestenes_dot(b)` | Symmetric contraction ($\|r - s\|$) | $\frac{1}{2}(AB + BA)$ for 1-vectors | Hestenes & Sobczyk (1984) §1.2 |
| **Regressive / Meet Product** | $A \vee B$ | `a.regressive_product(&b)` | `a.regressive(b)` | Intersection blade ($r + s - n$) | $A \vee B = (A^* \wedge B^*)^{*-1}$ | Dorst et al. (2007) §5.3, Hestenes §2.3 |
| **Hodge / Pseudoscalar Dual** | $*A = A I^{-1}$ | `a.dual()`, `a.undual()` | `a.dual()`, `a.undual()` | Orthogonal complement ($n - r$) | Multiplication by inverted pseudoscalar $I^{-1}$ | Hestenes & Sobczyk (1984) §1.4 |
| **Clifford Reversion** | $A^\sim$ / $A^\dagger$ | `a.reverse()` | `~a`, `a.reverse()` | Sign parity: $(-1)^{k(k-1)/2}$ | Blade order reversal: $(e_{i_1}\dots e_{i_k})^\sim = e_{i_k}\dots e_{i_1}$ | Lounesto (2001) §2.4, Dorst §3.6 |
| **Grade Involution** | $\widehat{A}$ | `a.grade_involution()` | `a.involution()` | Parity sign: $(-1)^k \langle A \rangle_k$ | Main automorphism: $\widehat{u v} = \widehat{u}\widehat{v}, \widehat{v} = -v$ | Hestenes (1984) §1.3, Lounesto §2.4 |
| **Clifford Conjugation** | $\overline{A}$ | `a.conjugation()` | `a.conjugation()` | Composition: $(\widehat{A})^\sim$ | Combines main involution and reversion | Lounesto (2001) §2.4 |
| **Versor Sandwich Product** | $V x V^{-1}$ | `v.sandwich(&x)` | `v.sandwich(x)` | Grade invariant automorphism | Outermorphism action $R x R^\sim$ on arbitrary blades | Cartan (1938), Hestenes (1984) §3.2 |
| **Lie Commutator** | $A \times B = \frac{1}{2}[A, B]$ | `a.commutator(&b)` | `a.commutator(b)` | Lie bracket on bivectors | $\frac{1}{2}(AB - BA)$, satisfies Jacobi identity | Doran & Lasenby (2003) §2.4 |
| **Anticommutator** | $\frac{1}{2}\{A, B\}$ | `a.anticommutator(&b)` | `a.anticommutator(b)` | Jordan algebra generator | $\frac{1}{2}(AB + BA)$ | Lounesto (2001) §3.3 |
| **Conformal Spherical Inversion** | $P' = S P S^{-1}$ | `sphere.invert_in_sphere(&p)` | `sphere.invert_in_sphere(p)` | Radial inversion: $r' = R^2 / r$ | 1-versor sandwich across conformal sphere | Dorst, Fontijne, Mann (2007) §14.3 |
| **Möbius Gyrovector Addition** | $u \oplus v$ | `u.mobius_add(&v)` | `u.mobius_add(v)` | Non-associative gyrovector sum | $\frac{(1 + 2\langle u,v \rangle + \|v\|^2)u + (1 - \|u\|^2)v}{1 + 2\langle u,v \rangle + \|u\|^2 \|v\|^2}$ | Ungar (2008) §2.2, Ratcliffe (2019) §4.5 |
| **Einstein Hyperbolic Midpoint** | $\mu_E(u_1..u_K)$ | `PoincarePoint::einstein_midpoint(&pts, &w)` | `PoincarePoint.einstein_midpoint(pts, w)` | Conformal center of mass in $B^d$ | Projective weighted mean in Beltrami-Klein $\mathbb{K}^d$ | Ungar (2008) §6.1, Ratcliffe §4.6 |

---

## 2. Fundamental Products

### 2.1 The Geometric Product ($A B$)
The core associative product uniting inner and outer products. For 1-vectors:
$$a b = a \cdot b + a \wedge b$$
For homogeneous blades $A_r, B_s$:
$$A_r B_s = \sum_{k=|r-s|}^{r+s} \langle A_r B_s \rangle_k \quad (\Delta k = 2)$$

### 2.2 Exterior Wedge Product ($A \wedge B$)
Represents the oriented subspace spanned by the operands:
$$A_r \wedge B_s = \langle A_r B_s \rangle_{r+s}$$
Properties:
1. Anticommutative on 1-vectors: $a \wedge b = -(b \wedge a)$
2. Nilpotent on linear dependence: $a \wedge a = 0$
3. Distributive and associative across multivectors.

### 2.3 Interior Contractions ($\rfloor$ and $\lfloor$)
- **Left Contraction ($A \rfloor B$)**: Grade $s - r$ ($r \le s$):
  $$A_r \rfloor B_s = \langle A_r B_s \rangle_{s-r}$$
- **Right Contraction ($A \lfloor B$)**: Grade $r - s$ ($r \ge s$):
  $$A_r \lfloor B_s = \langle A_r B_s \rangle_{r-s}$$

### 2.4 Hestenes Inner Product ($A \cdot B$)
Defined symmetrically for homogeneous blades $A_r, B_s$:
$$A_r \cdot B_s = \begin{cases} \langle A_r B_s \rangle_{|r-s|} & \text{if } r \neq 0, s \neq 0 \\ 0 & \text{if } r = 0 \text{ or } s = 0 \end{cases}$$

### 2.5 Regressive / Meet Product ($A \vee B$)
Computes the intersection of geometric subspaces in dimension $n$:
$$A \vee B = (A^* \wedge B^*)^{*-1}$$
where $X^*$ is the Hodge dual with respect to the pseudoscalar $I$.

---

## 3. Structural Automorphisms & Involutions

### 3.1 Reversion ($A^\sim$ or $A^\dagger$)
Reverses blade factor ordering:
$$\widetilde{e_{i_1} \dots e_{i_k}} = (-1)^{k(k-1)/2} e_{i_1} \dots e_{i_k}$$

### 3.2 Grade Involution ($\widehat{A}$)
Flips the sign of odd-grade elements:
$$\widehat{A} = \sum_{k=0}^n (-1)^k \langle A \rangle_k$$

### 3.3 Clifford Conjugation ($\overline{A}$)
Composition of reversion and grade involution:
$$\overline{A} = \widehat{\widetilde{A}} = \sum_{k=0}^n (-1)^{k(k+1)/2} \langle A \rangle_k$$

---

## 4. Lie, Jordan, and Hyperbolic Operators

### 4.1 Lie Commutator Bracket ($[A, B]$)
$$A \times B = \frac{1}{2}(AB - BA)$$
Forms a Lie algebra on the subspace of bivectors $\langle \mathcal{Cl} \rangle_2$, isomorphic to $\mathfrak{so}(p,q)$.

### 4.2 Anticommutator / Jordan Bracket ($\{A, B\}$)
$$\frac{1}{2}\{A, B\} = \frac{1}{2}(AB + BA)$$
Generates symmetric compositions and anti-Lie observable algebras.

### 4.3 Versor Sandwich Transformation ($V X V^{-1}$)
For any rotor $R \in \text{Spin}(p,q)$, the outermorphism:
$$X' = R X R^\sim$$
is an exact isometry, preserving grades, norms, and geometric angles.

### 4.4 Conformal Spherical Inversion
In 3D CGA $\mathcal{Cl}(4,1)$, reflecting a point $P$ across sphere $S$:
$$P' = S P S^{-1}$$
executes non-Euclidean spherical inversion $r' = R^2 / r$ preserving angles globally.

### 4.5 Möbius Gyrovector Addition & Einstein Midpoint
In the Poincaré hyperbolic model $\mathbb{B}^d$:
$$u \oplus v = \frac{(1 + 2\langle u, v \rangle + \|v\|^2) u + (1 - \|u\|^2) v}{1 + 2\langle u, v \rangle + \|u\|^2 \|v\|^2}$$
Non-associative gyrogroup addition enables calculating the Einstein center of mass $\mu_E$ in closed form via Beltrami-Klein projective mapping.
