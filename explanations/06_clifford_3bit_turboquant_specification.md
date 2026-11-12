# Technical Specification: 3-Bit Geometric Quantization and TurboQuant for VersorC-MoE

**Document ID:** `06_clifford_3bit_turboquant_specification.md`  
**Status:** Architecture Proposal & Transition Design (With Dual v5/v6 Fallback)  
**Date:** November 2026  
**Authors:** VersorC Core Architecture Team  

---

## 1. Executive Summary and Motivation

This specification formalizes the integration bridge between the low-level mathematical engine **`00_versorc-math`** and the heterogeneous distributed inference runtime **`02_versorc-moe`**.

### 1.1 The Operational Bottleneck in `02_versorc-moe`
Currently, the `.vrc v5` container in `versorc-moe` employs the **`VrcAsym64`** codec (4.50 bits per weight, 36 bytes per 64 weights). While this codec delivers high fidelity (cosine similarity 0.9964), it imposes severe pressure on memory-constrained systems (e.g., DDR5 hosts and 6 GB VRAM GPUs like RTX 4050 Laptop):
- Sparse MoE architectures (DeepSeek-MoE 16B, OLMoE 1B-7B, Qwen3-Next) transfer gigabytes of expert parameters per token.
- In the autoregressive decode phase (batch size = 1), performance is **100% memory-bandwidth bound**.
- Naive 1-bit and 2-bit quantization schemes suffer catastrophic angular collapse (cosine 0.694 on 1 KiB $E_8$ codebooks and 0.917 on unrotated Lloyd-Max) or severe execution stalls due to software scalar unpacking loops.

### 1.2 The Geometric Solution
Leveraging the **Clifford Dual Isometry** and the Walsh-Hadamard **Dyadic Versor** ($R \in \mathrm{Spin}(N)$) engineered in `00_versorc-math`:
- Channel activation and weight outliers are eliminated via Central Limit Theorem isotropic smearing with zero floating-point multiplications ($O(N \log N)$ additions and subtractions).
- Quantizes to **3.0 bits** (8 Lloyd-Max levels) reaching **0.98755** fidelity, or **3.5 bits (TurboQuant)** with $2 \times 2$ Gram orthogonal projection reaching **0.99815**.
- Aligned to hardware primitives: **16 weights per 64-bit word** (`Clifford3BitWord64`) and **128 weights per 64-byte L1 cache line** (`Clifford3BitCacheLine64`).

---

## 2. System Architecture & Inter-Library Contract

```text
  ┌────────────────────────────────────────────────────────────────────────┐
  │                           00_versorc-math                              │
  │                    (Low-Level Canonical GA Engine)                     │
  ├────────────────────────────────────────────────────────────────────────┤
  │  - DyadicWalshVersor (R in Spin(N), isometry R^T R = I, AVX2 / CPU)    │
  │  - Clifford3BitWord64 (64 bits: 16 weights x 3-bit + BF16 scale)       │
  │  - Clifford3BitCacheLine64 (64 bytes: 8 words = 128 weights)           │
  │  - TurboQuantBlock32 (32 weights x 3-bit base + 32-bit QJL + 2x BF16)  │
  │  - AOT Solvers: Least-Squares Scaling & 2x2 Gram Projection            │
  │  - Vulkan SPIR-V compute shaders for parallel GPU decompression        │
  └───────────────────────────────────┬────────────────────────────────────┘
                                      │ Imported as crate dependency:
                                      │ versorc-math = { path = "..." }
                                      ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │                           02_versorc-moe                               │
  │                    (Heterogeneous MoE Runtime)                         │
  ├───────────────────────────────────┬────────────────────────────────────┤
  │       versorc-compiler            │         versorc-runtime            │
  ├───────────────────────────────────┼────────────────────────────────────┤
  │ - Crystalizes GGUF -> .vrc v6     │ - Top-K routing over F32 router    │
  │ - Rotates weights AOT: R(W)       │ - Rotates activation once: R(x)    │
  │ - Packs into 3-bit / TurboQuant   │ - Direct streaming dot_128         │
  │ - Fallback: compiles v5 (asym64)  │ - Fallback: executes v5 native     │
  └───────────────────────────────────┴────────────────────────────────────┘
```

`02_versorc-moe` **does not reimplement Clifford mathematics**. It consumes the public types and functions exported by `versorc_math::quantization::*` and `versorc_math::versor::*`.

---

## 3. AOT Compiler Changes (`versorc-compiler`)

### 3.1 Container Format Evolution: `.vrc v6`
The `.vrc` binary format extends support for the new codec while preserving full backward compatibility:

```rust
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VrcCodecType {
    /// Legacy v5 codec: Asymmetric 64 (4.50 bpw, 36 bytes / 64 weights)
    Asym64 = 0,
    /// Codec v6: Clifford 3-Bit Lloyd-Max (3.00 bpw payload + BF16 scale)
    Clifford3Bit = 1,
    /// Codec v6: Clifford TurboQuant 3.5-Bit (3-bit base + 1-bit QJL residual)
    TurboQuant3_5 = 2,
}
```

In the container header `VrcHeader` (64-byte aligned):
- `magic`: `*b"VRC5"` for legacy fallback or `*b"VRC6"`.
- Reserved field `codec_type`: stores the `VrcCodecType` discriminant.

### 3.2 Geometry and Clean Divisibility (Zero Padding)
In standard MoE model dimensions:
- $d_{\text{model}} = 2048$
- $d_{\text{ff\_exp}} = 1408$ (DeepSeek-MoE) or $2048$ (OLMoE / Qwen)

Both dimensions are **clean multiples of 128**:
$$2048 / 128 = 16 \text{ cache lines of 64 bytes}$$
$$1408 / 128 = 11 \text{ cache lines of 64 bytes}$$

Every weight row across Gate, Up, and Down projections maps to an exact integral count of 64-byte `Clifford3BitCacheLine64` structs with zero padding required.

---

## 4. Inference Runtime Changes (`versorc-runtime`)

### 4.1 SwiGLU Execution Cycle with Dual Isometry
During forward execution of an MoE layer for token activation $x \in \mathbb{R}^{d_{\text{model}}}$:

1. **Routing**: The router evaluates $x$ and selects the top $K$ active experts and weights.
2. **Single Input Activation Rotation**:
   $$\tilde{x} = R_{d_{\text{model}}}(x)$$
   Evaluated in a thread-local reusable buffer ($< 0.8\,\mu\text{s}$ on AVX2 CPU using butterfly additions/subtractions).
3. **Gate and Up Evaluation in Rotated Domain**:
   $$h_{\text{gate}}[r] = \sum_{l=0}^{L-1} \text{line\_gate}[r, l].\text{dot\_128}(\tilde{x}[l \cdot 128 \dots])$$
   $$h_{\text{up}}[r] = \sum_{l=0}^{L-1} \text{line\_up}[r, l].\text{dot\_128}(\tilde{x}[l \cdot 128 \dots])$$
   *Critical rule*: Compressed weights in memory are never unrotated; operations occur directly on 3-bit cache lines.
4. **Intermediate Non-Linear Activation**:
   $$h = \text{SiLU}(h_{\text{gate}}) \odot h_{\text{up}} \in \mathbb{R}^{d_{\text{ff\_exp}}}$$
5. **Activation Rotation for Down Projection**:
   $$\tilde{h} = R_{d_{\text{ff\_exp}}}(h)$$
6. **Down Projection and Accumulation**:
   $$y_{\text{rot}}[r] = \sum_{l=0}^{L-1} \text{line\_down}[r, l].\text{dot\_128}(\tilde{h}[l \cdot 128 \dots])$$
7. **Final Layer Unrotation**:
   $$y = R_{d_{\text{model}}}^{-1}\left( \sum_{k} w_k \cdot y_{\text{rot}}^{(k)} \right)$$

---

## 5. Dual-Track Fallback Architecture (v5 / v6)

To guarantee stability, the engine implements a transparent dual-track dispatch:

```text
                       [ .vrc File ]
                             │
                             ▼  VRC Header Reader
                    header.codec_type?
                             │
            ┌────────────────┴────────────────┐
            │                                 │
      VrcCodecType::Asym64          VrcCodecType::Clifford3Bit
      (v5 Legacy / Fallback)        (v6 Clifford Isometry)
            │                                 │
            ▼                                 ▼
      Legacy v5 Pipeline            New v6 Pipeline
    - No activation rotation      - versor.apply_in_place(x)
    - 36B / 64w unpacking         - Streaming dot_128 (64B / 128w)
    - Scale & bias accumulation   - Zero intermediate buffers
```

---

## 6. Quantitative Silicon Benchmark Summary

Empirical release-mode measurements across an MoE expert layer ($1408 \times 2048$, 2,883,584 weights) from `tests/test_risk_audit_silicon.rs`:

| Parameter | VrcAsym64 (v5) | Clifford3Bit (v6 Scalar) | Clifford3Bit (v6 AVX2 Kernel) |
|---|---|---|---|
| **Footprint (bpw)** | 4.50 bpw | 4.00 bpw | 4.00 bpw |
| **Row Bytes ($d=2048$)** | 1,152 B | 1,024 B (-11.1%) | 1,024 B (-11.1%) |
| **MatVec Latency** | 1,934.0 µs (1.93 ms) | 10,829.0 µs (10.8 ms) | **510.0 µs (0.51 ms)** |
| **Throughput / Bandwidth** | 0.84 GB/s | 0.13 GB/s | **2.82 GB/s (3.79× faster)** |
| **Cosine vs FP32** | 0.99929 | 0.99620 | 0.99620 |
