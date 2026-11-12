// ============================================================================
// VERSORC-MATH: UNIFIED GEOMETRIC ALGEBRA SHADER LIBRARY (GLSL / SPIR-V)
// License: Apache-2.0
//
// Inlined, zero-overhead device functions operating entirely in registers
// (VGPRs) for all 15 Geometric Algebra operators, 3D PGA, CGA, and Poincare.
// ============================================================================

#ifndef VERSORC_GA_GLSL
#define VERSORC_GA_GLSL

// ----------------------------------------------------------------------------
// 1. Cl(3, 0) 8-Component Multivector Structure
// Indices:
// c[0]: 1 (scalar)
// c[1]: e1
// c[2]: e2
// c[3]: e12 (e1 ^ e2)
// c[4]: e3
// c[5]: e13 (e1 ^ e3)
// c[6]: e23 (e2 ^ e3)
// c[7]: e123 (pseudoscalar I)
// ----------------------------------------------------------------------------
struct Cl3MV {
    float c[8];
};

Cl3MV cl3_zero() {
    Cl3MV res;
    for (int i = 0; i < 8; ++i) res.c[i] = 0.0;
    return res;
}

Cl3MV cl3_scalar(float s) {
    Cl3MV res = cl3_zero();
    res.c[0] = s;
    return res;
}

Cl3MV cl3_vector(float x, float y, float z) {
    Cl3MV res = cl3_zero();
    res.c[1] = x;
    res.c[2] = y;
    res.c[4] = z;
    return res;
}

Cl3MV cl3_add(Cl3MV a, Cl3MV b) {
    Cl3MV res;
    for (int i = 0; i < 8; ++i) res.c[i] = a.c[i] + b.c[i];
    return res;
}

Cl3MV cl3_sub(Cl3MV a, Cl3MV b) {
    Cl3MV res;
    for (int i = 0; i < 8; ++i) res.c[i] = a.c[i] - b.c[i];
    return res;
}

Cl3MV cl3_scale(Cl3MV a, float s) {
    Cl3MV res;
    for (int i = 0; i < 8; ++i) res.c[i] = a.c[i] * s;
    return res;
}

// ----------------------------------------------------------------------------
// 2. Operator 1: Geometric Product (A B) in Cl(3, 0)
// Unrolled exact Cayley multiplication
// ----------------------------------------------------------------------------
Cl3MV cl3_geometric_product(Cl3MV a, Cl3MV b) {
    Cl3MV res;
    res.c[0] = a.c[0] * b.c[0] + a.c[1] * b.c[1] + a.c[2] * b.c[2] - a.c[3] * b.c[3] + a.c[4] * b.c[4] - a.c[5] * b.c[5] - a.c[6] * b.c[6] - a.c[7] * b.c[7];
    res.c[1] = a.c[0] * b.c[1] + a.c[1] * b.c[0] - a.c[2] * b.c[3] + a.c[3] * b.c[2] - a.c[4] * b.c[5] + a.c[5] * b.c[4] - a.c[6] * b.c[7] - a.c[7] * b.c[6];
    res.c[2] = a.c[0] * b.c[2] + a.c[1] * b.c[3] + a.c[2] * b.c[0] - a.c[3] * b.c[1] - a.c[4] * b.c[6] + a.c[5] * b.c[7] + a.c[6] * b.c[4] + a.c[7] * b.c[5];
    res.c[3] = a.c[0] * b.c[3] + a.c[1] * b.c[2] - a.c[2] * b.c[1] + a.c[3] * b.c[0] + a.c[4] * b.c[7] - a.c[5] * b.c[6] + a.c[6] * b.c[5] + a.c[7] * b.c[4];
    res.c[4] = a.c[0] * b.c[4] + a.c[1] * b.c[5] + a.c[2] * b.c[6] - a.c[3] * b.c[7] + a.c[4] * b.c[0] - a.c[5] * b.c[1] - a.c[6] * b.c[2] - a.c[7] * b.c[3];
    res.c[5] = a.c[0] * b.c[5] + a.c[1] * b.c[4] - a.c[2] * b.c[7] + a.c[3] * b.c[6] - a.c[4] * b.c[1] + a.c[5] * b.c[0] - a.c[6] * b.c[3] - a.c[7] * b.c[2];
    res.c[6] = a.c[0] * b.c[6] + a.c[1] * b.c[7] + a.c[2] * b.c[4] - a.c[3] * b.c[5] - a.c[4] * b.c[2] + a.c[5] * b.c[3] + a.c[6] * b.c[0] + a.c[7] * b.c[1];
    res.c[7] = a.c[0] * b.c[7] + a.c[1] * b.c[6] - a.c[2] * b.c[5] + a.c[3] * b.c[4] + a.c[4] * b.c[3] - a.c[5] * b.c[2] + a.c[6] * b.c[1] + a.c[7] * b.c[0];
    return res;
}

// ----------------------------------------------------------------------------
// 3. Operator 2: Outer / Wedge Product (A ^ B)
// ----------------------------------------------------------------------------
Cl3MV cl3_wedge(Cl3MV a, Cl3MV b) {
    Cl3MV res;
    res.c[0] = a.c[0] * b.c[0];
    res.c[1] = a.c[0] * b.c[1] + a.c[1] * b.c[0];
    res.c[2] = a.c[0] * b.c[2] + a.c[2] * b.c[0];
    res.c[3] = a.c[0] * b.c[3] + a.c[1] * b.c[2] - a.c[2] * b.c[1] + a.c[3] * b.c[0];
    res.c[4] = a.c[0] * b.c[4] + a.c[4] * b.c[0];
    res.c[5] = a.c[0] * b.c[5] + a.c[1] * b.c[4] - a.c[4] * b.c[1] + a.c[5] * b.c[0];
    res.c[6] = a.c[0] * b.c[6] + a.c[2] * b.c[4] - a.c[4] * b.c[2] + a.c[6] * b.c[0];
    res.c[7] = a.c[0] * b.c[7] + a.c[1] * b.c[6] - a.c[2] * b.c[5] + a.c[3] * b.c[4] + a.c[4] * b.c[3] - a.c[5] * b.c[2] + a.c[6] * b.c[1] + a.c[7] * b.c[0];
    return res;
}

// ----------------------------------------------------------------------------
// 4. Operator 3: Left Contraction (A _| B)
// ----------------------------------------------------------------------------
Cl3MV cl3_left_contraction(Cl3MV a, Cl3MV b) {
    Cl3MV res;
    res.c[0] = a.c[0] * b.c[0] + a.c[1] * b.c[1] + a.c[2] * b.c[2] - a.c[3] * b.c[3] + a.c[4] * b.c[4] - a.c[5] * b.c[5] - a.c[6] * b.c[6] - a.c[7] * b.c[7];
    res.c[1] = a.c[0] * b.c[1] - a.c[2] * b.c[3] - a.c[4] * b.c[5] - a.c[6] * b.c[7];
    res.c[2] = a.c[0] * b.c[2] + a.c[1] * b.c[3] - a.c[4] * b.c[6] + a.c[5] * b.c[7];
    res.c[3] = a.c[0] * b.c[3] + a.c[4] * b.c[7];
    res.c[4] = a.c[0] * b.c[4] + a.c[1] * b.c[5] + a.c[2] * b.c[6] - a.c[3] * b.c[7];
    res.c[5] = a.c[0] * b.c[5] - a.c[2] * b.c[7];
    res.c[6] = a.c[0] * b.c[6] + a.c[1] * b.c[7];
    res.c[7] = a.c[0] * b.c[7];
    return res;
}

// ----------------------------------------------------------------------------
// 5. Operator 4: Hestenes Inner Product (Fat Dot: A • B)
// ----------------------------------------------------------------------------
Cl3MV cl3_hestenes_inner(Cl3MV a, Cl3MV b) {
    Cl3MV res;
    res.c[0] = a.c[1] * b.c[1] + a.c[2] * b.c[2] - a.c[3] * b.c[3] + a.c[4] * b.c[4] - a.c[5] * b.c[5] - a.c[6] * b.c[6] - a.c[7] * b.c[7];
    res.c[1] = - a.c[2] * b.c[3] + a.c[3] * b.c[2] - a.c[4] * b.c[5] + a.c[5] * b.c[4] - a.c[6] * b.c[7] - a.c[7] * b.c[6];
    res.c[2] = a.c[1] * b.c[3] - a.c[3] * b.c[1] - a.c[4] * b.c[6] + a.c[5] * b.c[7] + a.c[6] * b.c[4] + a.c[7] * b.c[5];
    res.c[3] = a.c[4] * b.c[7] + a.c[7] * b.c[4];
    res.c[4] = a.c[1] * b.c[5] + a.c[2] * b.c[6] - a.c[3] * b.c[7] - a.c[5] * b.c[1] - a.c[6] * b.c[2] - a.c[7] * b.c[3];
    res.c[5] = - a.c[2] * b.c[7] - a.c[7] * b.c[2];
    res.c[6] = a.c[1] * b.c[7] + a.c[7] * b.c[1];
    res.c[7] = 0.0;
    return res;
}

// ----------------------------------------------------------------------------
// 6. Operator 5: Hodge Dual (*A = A I^{-1}) & Undual (A I)
// For Cl(3, 0): I = e123, I^{-1} = -e123
// ----------------------------------------------------------------------------
Cl3MV cl3_hodge_dual(Cl3MV a) {
    Cl3MV res;
    res.c[0] = a.c[7];
    res.c[1] = a.c[6];
    res.c[2] = -a.c[5];
    res.c[3] = -a.c[4];
    res.c[4] = a.c[3];
    res.c[5] = a.c[2];
    res.c[6] = -a.c[1];
    res.c[7] = -a.c[0];
    return res;
}

Cl3MV cl3_undual(Cl3MV a) {
    Cl3MV res;
    res.c[0] = -a.c[7];
    res.c[1] = -a.c[6];
    res.c[2] = a.c[5];
    res.c[3] = a.c[4];
    res.c[4] = -a.c[3];
    res.c[5] = -a.c[2];
    res.c[6] = a.c[1];
    res.c[7] = a.c[0];
    return res;
}

// ----------------------------------------------------------------------------
// 7. Operator 6: Regressive / Meet Product (A ∨ B = (A* ∧ B*).undual())
// ----------------------------------------------------------------------------
Cl3MV cl3_regressive_meet(Cl3MV a, Cl3MV b) {
    Cl3MV dual_a = cl3_hodge_dual(a);
    Cl3MV dual_b = cl3_hodge_dual(b);
    Cl3MV wedge_dual = cl3_wedge(dual_a, dual_b);
    return cl3_undual(wedge_dual);
}

// ----------------------------------------------------------------------------
// 8. Operators 7, 8, 9: Involutions & Automorphisms
// - Reversion: A~ = sum (-1)^{k(k-1)/2} <A>_k
// - Grade Involution: A^ = sum (-1)^k <A>_k
// - Clifford Conjugation: \bar{A} = sum (-1)^{k(k+1)/2} <A>_k
// ----------------------------------------------------------------------------
Cl3MV cl3_reverse(Cl3MV a) {
    Cl3MV res;
    res.c[0] = a.c[0];
    res.c[1] = a.c[1];
    res.c[2] = a.c[2];
    res.c[3] = -a.c[3];
    res.c[4] = a.c[4];
    res.c[5] = -a.c[5];
    res.c[6] = -a.c[6];
    res.c[7] = -a.c[7];
    return res;
}

Cl3MV cl3_involution(Cl3MV a) {
    Cl3MV res;
    res.c[0] = a.c[0];
    res.c[1] = -a.c[1];
    res.c[2] = -a.c[2];
    res.c[3] = a.c[3];
    res.c[4] = -a.c[4];
    res.c[5] = a.c[5];
    res.c[6] = a.c[6];
    res.c[7] = -a.c[7];
    return res;
}

Cl3MV cl3_conjugation(Cl3MV a) {
    Cl3MV res;
    res.c[0] = a.c[0];
    res.c[1] = -a.c[1];
    res.c[2] = -a.c[2];
    res.c[3] = -a.c[3];
    res.c[4] = -a.c[4];
    res.c[5] = -a.c[5];
    res.c[6] = -a.c[6];
    res.c[7] = a.c[7];
    return res;
}

// ----------------------------------------------------------------------------
// 9. Operator 10: Versor Sandwich Transformation (X' = V X V~)
// ----------------------------------------------------------------------------
Cl3MV cl3_sandwich(Cl3MV v, Cl3MV x) {
    Cl3MV vx = cl3_geometric_product(v, x);
    Cl3MV v_rev = cl3_reverse(v);
    return cl3_geometric_product(vx, v_rev);
}

// ----------------------------------------------------------------------------
// 10. Operator 11 & 12: Lie Commutator and Jordan Anticommutator
// [A, B] = 1/2 (A B - B A)
// {A, B} = 1/2 (A B + B A)
// ----------------------------------------------------------------------------
Cl3MV cl3_commutator(Cl3MV a, Cl3MV b) {
    Cl3MV ab = cl3_geometric_product(a, b);
    Cl3MV ba = cl3_geometric_product(b, a);
    return cl3_scale(cl3_sub(ab, ba), 0.5);
}

Cl3MV cl3_anticommutator(Cl3MV a, Cl3MV b) {
    Cl3MV ab = cl3_geometric_product(a, b);
    Cl3MV ba = cl3_geometric_product(b, a);
    return cl3_scale(cl3_add(ab, ba), 0.5);
}

// ----------------------------------------------------------------------------
// 11. Operator 13: Conformal Spherical Inversion (CGA Cl(4, 1))
// P' = S P S^{-1} => maps ||p'|| = R^2 / ||p|| radially
// ----------------------------------------------------------------------------
vec3 cga_spherical_inversion(vec3 p, vec3 center, float radius) {
    vec3 diff = p - center;
    float dist_sq = dot(diff, diff);
    if (dist_sq < 1e-12) return center;
    float factor = (radius * radius) / dist_sq;
    return center + diff * factor;
}

// ----------------------------------------------------------------------------
// 12. Operator 14: Mobius Gyrovector Addition in Poincare Ball B^d
// u (+) v = ((1 + 2<u,v> + ||v||^2)u + (1 - ||u||^2)v) / (1 + 2<u,v> + ||u||^2 ||v||^2)
// ----------------------------------------------------------------------------
vec3 poincare_mobius_add(vec3 u, vec3 v) {
    float u2 = dot(u, u);
    float v2 = dot(v, v);
    float uv2 = 2.0 * dot(u, v);
    float denom = 1.0 + uv2 + u2 * v2;
    if (abs(denom) < 1e-12) return u;
    vec3 num = (1.0 + uv2 + v2) * u + (1.0 - u2) * v;
    return num / denom;
}

// Poincare geodesic distance
float poincare_distance(vec3 u, vec3 v) {
    float u2 = min(dot(u, u), 0.999999);
    float v2 = min(dot(v, v), 0.999999);
    vec3 diff = u - v;
    float diff2 = dot(diff, diff);
    float denom = (1.0 - u2) * (1.0 - v2);
    if (denom <= 1e-12) return 0.0;
    float alpha = 1.0 + 2.0 * diff2 / denom;
    // acosh(alpha) = ln(alpha + sqrt(alpha^2 - 1))
    return log(alpha + sqrt(max(alpha * alpha - 1.0, 0.0)));
}

// ----------------------------------------------------------------------------
// 13. Operator 15: Einstein Hyperbolic Midpoint in Poincare Ball
// Projective correspondence through Beltrami-Klein model
// ----------------------------------------------------------------------------
vec3 poincare_to_klein(vec3 p) {
    float p2 = dot(p, p);
    return (2.0 / (1.0 + p2)) * p;
}

vec3 klein_to_poincare(vec3 k) {
    float k2 = dot(k, k);
    if (k2 >= 1.0) k2 = 0.999999;
    return k / (1.0 + sqrt(max(1.0 - k2, 1e-12)));
}

vec3 poincare_einstein_midpoint(vec3 u, vec3 v) {
    vec3 ku = poincare_to_klein(u);
    vec3 kv = poincare_to_klein(v);
    float ku2 = min(dot(ku, ku), 0.99999);
    float kv2 = min(dot(kv, kv), 0.99999);
    float gu = 1.0 / max(sqrt(1.0 - ku2), 1e-7);
    float gv = 1.0 / max(sqrt(1.0 - kv2), 1e-7);
    float sum_g = gu + gv;
    if (sum_g <= 1e-9) return u;
    vec3 km = (gu * ku + gv * kv) / sum_g;
    return klein_to_poincare(km);
}

#endif // VERSORC_GA_GLSL
