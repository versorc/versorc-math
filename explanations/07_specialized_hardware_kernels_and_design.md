# Specialized Hardware Kernels and Microarchitectural Design

**Repository Module:** `src/kernels/`  
**Active Implementations:**
- [`src/kernels/clifford_3bit_avx2.rs`](../src/kernels/clifford_3bit_avx2.rs): 3-bit SIMD AVX2 microkernel with `vpermps` + `vpsrlvd`.
- [`shaders/`](../shaders/): Vulkan SPIR-V compute shaders for parallel GPU execution.

---

## 1. Motivation and Microarchitectural Diagnosis

During initial risk auditing, a naive scalar loop for 3-bit unpacking took **10,829 µs**, performing 5.59× slower than `VrcAsym64` (1,934 µs) due to the ALU instruction overhead of serial bit-shifts and scalar conversions.

To eliminate this bottleneck and ensure 3-bit geometric quantization dominates in silicon, the specialized [`src/kernels/`](../src/kernels/) module was engineered according to three Data-Oriented Design (DOD) principles:
1. **Parallel SIMD Extraction**: Unpack 8 3-bit codes simultaneously in a single 256-bit AVX2 register without scalar branching.
2. **Single-Cycle Centroid Reconstruction via `vpermps`**: Map the 8 combinatorial states $\{-c_3 \dots +c_3\}$ directly into a 256-bit register table, eliminating sign-magnitude conditional branches and runtime conversions.
3. **Fused Multiply-Add (FMA) with Hoisted Scaling**: Accumulate dot products directly across vector registers and apply the BF16 block scale once per 16 weights.

---

## 2. AVX2 Instruction Pipeline (`vpermps`)

### 2.1 The 8-State Lookup Strategy
Each 3-bit quantized weight consists of 1 sign bit $s \in \{0, 1\}$ and 2 Lloyd-Max magnitude bits $m \in \{0, 1, 2, 3\}$.
The resulting 3-bit index is:
$$k = (s \ll 2) \mid m \in [0 \dots 7]$$

An AVX2 register `table` is constructed with the 8 exact centroids:
```rust
let table = _mm256_setr_ps(
    +c0, +c1, +c2, +c3, // Indices 0..3: Positive values (s = 0)
    -c0, -c1, -c2, -c3, // Indices 4..7: Negative values (s = 1)
);
```
**The packed 3-bit code serves natively as the permutation index for `vpermps`**. No sign masking or conditional instructions are needed.

### 2.2 Extraction and Permutation in 6 Instructions
For an 8-weight block (24 bits inside `word.0`):

1. **Broadcast (`vpbroadcastd`)**: Clones the 24 bits across all 8 32-bit lanes of an AVX2 register.
2. **Variable Shift (`vpsrlvd`)**: Shifts each lane $i \in [0 \dots 7]$ by $i \times 3$ bits using the precomputed vector `[0, 3, 6, 9, 12, 15, 18, 21]`.
3. **Bitwise AND (`vpand`)**: Applies `& 0x07` to isolate the 3-bit index in each lane.
4. **Centroid Permutation (`vpermps`)**: `_mm256_permutevar8x32_ps(table, indices)` extracts all 8 reconstructed floating-point values in **a single clock cycle**.
5. **Fused Multiply-Add (`vfmadd231ps`)**: Loads 8 floating-point activation values from $x$ and accumulates using FMA.

---

## 3. Empirical Silicon Benchmark (x86_64 AVX2 / DDR5)

Empirical release-mode benchmark across a **$1408 \times 2048$ (2,883,584 weights)** MoE expert projection layer (`tests/test_risk_audit_silicon.rs`):

```text
================================================================================================
 SILICON COMPARATIVE BENCHMARK (MoE Matrix 1408 x 2048 in RAM)
================================================================================================
 Implementation / Codec                    MatVec Latency       Bandwidth         Cosine vs FP32
------------------------------------------------------------------------------------------------
 VrcAsym64 (4.50 bpw - v5 Production)     1,934.0 µs (1.93 ms)  0.84 GB/s         0.99929
 Clifford3Bit (4.00 bpw - Scalar Loop)   10,829.0 µs (10.8 ms)  0.13 GB/s         0.99620
 Clifford3Bit (4.00 bpw - AVX2 KERNEL)     510.0 µs (0.51 ms)  2.82 GB/s         0.99620
================================================================================================
 AVX2 KERNEL PERFORMANCE GAIN:
 -> 21.23x faster than scalar 3-bit unpacking.
 -> 3.79x FASTER THAN VrcAsym64 (0.51 ms vs 1.93 ms) with 11.1% less memory bandwidth.
================================================================================================
```

---

## 4. Kernel Usage in High-Performance Runtimes

```rust
use versorc_math::kernels::matvec_clifford_3bit_avx2;
use versorc_math::versor::DyadicWalshVersor;

// During runtime expert layer evaluation:
let versor = DyadicWalshVersor::new(cols, 42);

// Evaluates y = W * x directly over 64-byte L1 cache lines
// Performs single activation rotation and AVX2 streaming kernel
matvec_clifford_3bit_avx2(
    &expert_cache_lines, // &[Clifford3BitCacheLine64]
    1408,                // rows
    2048,                // cols
    &activation_x,       // &[f32]
    &mut output_y,       // &mut [f32]
    &versor,
);
```
