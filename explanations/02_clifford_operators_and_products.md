# Chapter 02: Clifford Operators and Products

This chapter documents the operator taxonomy and algebraic contracts implemented in `versorc-math`. Every operator is defined with mathematical exactness and mapped directly to hardware-efficient algorithmic implementations.

---

## 1. The Geometric Product ($A B$)

The fundamental operation in Clifford algebra is the associative, distributive **geometric product**. For two 1-vectors $a, b \in V$:

$$a b = a \cdot b + a \wedge b$$

where:
- $a \cdot b = \frac{1}{2}(a b + b a) \in \mathbb{R}$ is the symmetric **interior scalar product** (grade 0).
- $a \wedge b = \frac{1}{2}(a b - b a) \in \langle \mathcal{Cl} \rangle_2$ is the antisymmetric **exterior wedge product** (grade 2).

For arbitrary homogeneous multivectors (blades) $A_r \in \langle \mathcal{Cl} \rangle_r$ and $B_s \in \langle \mathcal{Cl} \rangle_s$, their geometric product decomposes across grades:

$$A_r B_s = \sum_{k=|r-s|}^{r+s} \langle A_r B_s \rangle_k \quad \text{with step size } \Delta k = 2$$

---

## 2. Exterior Wedge Product ($A \wedge B$)

The Grassmann wedge product represents the oriented subspace spanned by the operands:

$$A_r \wedge B_s = \langle A_r B_s \rangle_{r+s}$$

### Algebraic Properties:
1. **Anticommutativity on vectors:** $a \wedge b = - (b \wedge a)$
2. **Nilpotence on linear dependence:** $a \wedge a = 0$
3. **Associativity:** $(A \wedge B) \wedge C = A \wedge (B \wedge C)$
4. **Distributivity:** $A \wedge (B + C) = A \wedge B + A \wedge C$

In `versorc-math`, the wedge product of two canonical basis blades $e_I$ and $e_J$ (where $I, J \subseteq \{1, \dots, n\}$ are bitmasks) is nonzero if and only if $I \cap J = \emptyset$:

$$e_I \wedge e_J = \begin{cases} (-1)^{\text{inversions}(I, J)} e_{I \cup J} & \text{if } I \cap J = \emptyset \\ 0 & \text{otherwise} \end{cases}$$

---

## 3. Interior Contraction Operators ($\rfloor$ and $\lfloor$)

Unlike the symmetric dot product, high-dimensional Geometric Algebra utilizes oriented contractions:

### 3.1 Left Contraction ($A \rfloor B$)
Also called the interior product or "A contracted onto B":

$$A_r \rfloor B_s = \begin{cases} \langle A_r B_s \rangle_{s-r} & \text{if } r \le s \\ 0 & \text{if } r > s \end{cases}$$

For a 1-vector $a$ and a 2-blade $B = u \wedge v$:
$$a \rfloor (u \wedge v) = (a \cdot u) v - (a \cdot v) u$$
This projects the vector $a$ into the plane of $B$ and rotates it by $90^\circ$ clockwise.

### 3.2 Right Contraction ($A \lfloor B$)
"A contracted by B":

$$A_r \lfloor B_s = \begin{cases} \langle A_r B_s \rangle_{r-s} & \text{if } r \ge s \\ 0 & \text{if } r < s \end{cases}$$

---

## 4. Fundamental Involutions & Automorphisms

`versorc-math` provides three structural linear automorphisms:

### 4.1 Reverse ($\widetilde{A}$ or $A^\dagger$)
Reverses the order of all vector factors in every blade:
$$\widetilde{e_{i_1} e_{i_2} \cdots e_{i_k}} = e_{i_k} \cdots e_{i_2} e_{i_1} = (-1)^{\frac{k(k-1)}{2}} e_{i_1} \cdots e_{i_k}$$

| Grade $k$ | Sign $(-1)^{k(k-1)/2}$ | Description |
| :---: | :---: | :--- |
| **0** (Scalar) | $+1$ | Invariant |
| **1** (Vector) | $+1$ | Invariant |
| **2** (Bivector) | $-1$ | Anti-self-adjoint ($\widetilde{B} = -B$) |
| **3** (Trivector)| $-1$ | Anti-self-adjoint |
| **4** (Quadvector)| $+1$ | Self-adjoint |

### 4.2 Grade Involution / Main Automorphism ($\widehat{A}$)
Flips the sign of odd-grade elements:
$$\widehat{A} = \sum_{k=0}^n (-1)^k \langle A \rangle_k$$
This preserves algebra grading: $\widehat{A B} = \widehat{A} \widehat{B}$.

### 4.3 Clifford Conjugation ($\overline{A}$)
The composition of reversion and grade involution:
$$\overline{A} = \widehat{\widetilde{A}} = \sum_{k=0}^n (-1)^{\frac{k(k+1)}{2}} \langle A \rangle_k$$
For quaternions $\mathbb{H}$, Clifford conjugation is identical to classical quaternion conjugation:
$$\overline{w + x \mathbf{i} + y \mathbf{j} + z \mathbf{k}} = w - x \mathbf{i} - y \mathbf{j} - z \mathbf{k}$$

---

## 5. Sandwich Product ($V X \widetilde{V}$)

The general action of an operator $V \in \mathbf{\Gamma}_{p,q}$ on any multivector element $X \in \mathcal{Cl}_{p,q}$ is computed by the two-sided **sandwich product**:

$$X' = V X \widetilde{V}$$

When $V$ is a unit rotor ($V \widetilde{V} = 1$), the sandwich product is:
1. **Norm-Preserving (Isometry):** $\|X'\| = \|X\|$
2. **Grade-Preserving (Automorphism):** $\langle X' \rangle_k = V \langle X \rangle_k \widetilde{V}$
3. **Structure-Preserving:** $V (X \wedge Y) \widetilde{V} = (V X \widetilde{V}) \wedge (V Y \widetilde{V})$

This guarantees that rotating geometric configurations preserves angles, areas, and volumes unconditionally.
