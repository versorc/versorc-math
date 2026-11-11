# Chapter 01: Geometric Algebra & The Versor Group

This chapter establishes the algebraic foundations of **VersorC Math (`versorc-math`)**, focusing on Clifford Geometric Algebras $\mathcal{Cl}_{p,q,r}$, the Lipschitz/Versor group $\mathbf{\Gamma}_{p,q}$, Hamilton's classical quaternions $\mathbb{H}$, and spinor transformations.

---

## 1. Mathematical Foundations: The Clifford Algebra $\mathcal{Cl}_{p,q,r}$

Let $V$ be a real vector space of dimension $n = p + q + r$ equipped with a symmetric bilinear form $B: V \times V \to \mathbb{R}$ and associated quadratic form $Q(v) = B(v, v)$.
The **metric signature** $(p, q, r)$ denotes:
- $p$: number of orthogonal basis vectors squaring to $+1$ (Euclidean space).
- $q$: number of orthogonal basis vectors squaring to $-1$ (anti-Euclidean / Minkowski time-like).
- $r$: number of orthogonal basis vectors squaring to $0$ (degenerate / Projective Geometric Algebra).

An orthogonal basis $\{e_1, e_2, \dots, e_n\}$ satisfies the canonical Clifford relations:

$$e_i e_j + e_j e_i = 2 B(e_i, e_j) = 2 \eta_{ij}$$

Specifically:
- $e_i^2 = +1$ for $1 \le i \le p$
- $e_i^2 = -1$ for $p < i \le p + q$
- $e_i^2 = 0$ for $p + q < i \le p + q + r$
- $e_i e_j = -e_j e_i$ for all $i \ne j$

The universal Clifford algebra $\mathcal{Cl}_{p,q,r}$ is a graded associative algebra of dimension $2^n$:

$$\mathcal{Cl}_{p,q,r} = \bigoplus_{k=0}^n \langle \mathcal{Cl}_{p,q,r} \rangle_k$$

where each subspace $\langle \mathcal{Cl}_{p,q,r} \rangle_k$ contains **$k$-blades** (homogeneous multivectors of grade $k$) with dimension $\binom{n}{k}$.

---

## 2. The Lipschitz Group and Definition of a Versor

In classical algebra, a **Versor** was introduced by William Rowan Hamilton (1843) in the context of quaternions $\mathbb{H}$ as an operator that alters the *direction* (turns) of a vector without altering its *magnitude* (length).

In Clifford Geometric Algebra, Hamilton's definition generalizes rigorously to the **Lipschitz Group** (also called the **Clifford Group** or **Versor Group** $\mathbf{\Gamma}_{p,q}$):

### Definition: $k$-Versor
A **versor** $V \in \mathcal{Cl}_{p,q,r}$ is any multivector that can be expressed as the geometric product of $k$ non-null invertible 1-vectors $v_1, v_2, \dots, v_k \in V$:

$$V = v_1 v_2 \cdots v_k, \quad \text{where } Q(v_i) \ne 0 \;\; \forall i \in \{1, \dots, k\}$$

The integer $k$ is called the **versor order** (or grade parity):
- **1-Versor (Reflection / Householder operator):** A single vector $v \in V$. Acting on a vector $x$ via sandwich reflection:
  $$x' = -v x v^{-1} = x - 2 \frac{B(x, v)}{Q(v)} v$$
- **2-Versor (Rotor / Planar turn):** The product of two vectors $R = u v$. An even versor with unit norm ($R \widetilde{R} = 1$) that performs an exact orthogonal rotation in the plane spanned by $u \wedge v$.
- **$k$-Versor (General orthogonal transformation):** By the **Cartan-Dieudonné Theorem**, every orthogonal transformation $T \in O(p, q)$ on an $n$-dimensional space can be represented as the composition of at most $n$ reflections (i.e., a $k$-versor with $k \le n$).

---

## 3. Hamilton Quaternions as an Even Clifford Subalgebra $\mathcal{Cl}_{0,2}^+ \cong \mathbb{H}$

In Hamilton's notation, a quaternion $q \in \mathbb{H}$ is given by:

$$q = w + x \mathbf{i} + y \mathbf{j} + z \mathbf{k}, \quad \mathbf{i}^2 = \mathbf{j}^2 = \mathbf{k}^2 = \mathbf{i}\mathbf{j}\mathbf{k} = -1$$

In Clifford Geometric Algebra, $\mathbb{H}$ is isomorphic to the even subalgebra of the 3D Euclidean space $\mathcal{Cl}_{3,0}^+$ (or $\mathcal{Cl}_{0,2}$):

$$\mathbf{i} \longleftrightarrow -e_2 e_3, \quad \mathbf{j} \longleftrightarrow -e_3 e_1, \quad \mathbf{k} \longleftrightarrow -e_1 e_2$$

Each basis element $\mathbf{i}, \mathbf{j}, \mathbf{k}$ is **not an imaginary scalar**, but a **directed physical plane (bivector)**.
Squaring the bivectors reveals the algebraic origin of $-1$:

$$(-e_1 e_2)^2 = e_1 e_2 e_1 e_2 = -e_1 e_1 e_2 e_2 = -(+1)(+1) = -1$$

A unit quaternion $|q| = 1$ is a **2-versor (rotor)** $R \in \mathbf{Spin}(3)$. The rotation of a 3D vector $\vec{v} = v_1 e_1 + v_2 e_2 + v_3 e_3$ by angle $\theta$ around unit axis $\vec{u} = u_x e_1 + u_y e_2 + u_z e_3$ is the sandwich product:

$$R = \cos\left(\frac{\theta}{2}\right) - \vec{u}^* \sin\left(\frac{\theta}{2}\right)$$
$$\vec{v}' = R \, \vec{v} \, R^\dagger$$

where $R^\dagger = \widetilde{R}$ is the Clifford reverse.

---

## 4. Spinors and the Exponential Map

Every rotor $R \in \mathbf{Spin}(p, q)$ can be generated from an exponentiated 2-blade (bivector) $B \in \langle \mathcal{Cl}_{p,q} \rangle_2$:

$$R = \exp\left(-\frac{B}{2}\right) = \sum_{k=0}^\infty \frac{(-B/2)^k}{k!}$$

When $B^2 < 0$ (elliptic plane, Euclidean signature), let $\|B\| = \theta$:

$$R = \cos\left(\frac{\theta}{2}\right) - \frac{B}{\|B\|} \sin\left(\frac{\theta}{2}\right)$$

When $B^2 > 0$ (hyperbolic plane, Minkowski signature), let $\|B\| = \phi$:

$$R = \cosh\left(\frac{\phi}{2}\right) - \frac{B}{\|B\|} \sinh\left(\frac{\phi}{2}\right)$$

When $B^2 = 0$ (parabolic plane, Projective PGA signature):

$$R = 1 - \frac{B}{2}$$

This unified exponential map allows `versorc-math` to handle Euclidean rotations, Lorentz boosts, and Euclidean translations within the exact same operator architecture.
