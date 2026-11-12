# Contributing to VersorC-Math

Welcome to the **VersorC-Math** open-source project! We are thrilled to have you join our community. Whether you are an expert in Clifford Geometric Algebra, a systems engineer optimizing SIMD/GPU kernels, a Python data scientist, or a student eager to learn and improve documentation, **your contributions are genuinely welcome**.

---

## 🌟 Our Community Ethos

- **Warm & Inclusive**: We believe in creating a welcoming, encouraging environment where anyone can ask questions, propose ideas, and learn without judgment.
- **Academic & Empirical Rigor**: We value mathematical precision, empirical test validation, and zero mock data.
- **Collaborative Ownership**: `versorc-math` is a public good licensed under **Apache 2.0**. We recognize and celebrate all contributors.

---

## 🚀 How to Get Started

### 1. Find or Propose Something to Work On
- **Good First Issues**: Look for issues tagged `good-first-issue` or `documentation` on our GitHub tracker.
- **New Geometric Signatures**: Implement or extend support for other algebras (e.g. $\mathcal{Cl}(0,2)$, Compass GA, Double Conformal GA).
- **SIMD & Acceleration**: Implement or benchmark ARM NEON, AVX-512, or WebAssembly SIMD kernels.
- **Python / Framework Wrappers**: Expand PyTorch, JAX, or NumPy operator integrations.
- **Documentation & Educational Visualizations**: Add new GanJa.js scenes, interactive notebooks, or clarify proofs in `explanations/`.

### 2. Development Setup

#### Prerequisites
- **Rust Toolchain**: Stable (1.80+) via [rustup.rs](https://rustup.rs).
- **Python**: 3.9+ with `numpy` and `pytest`.
- **Maturin & uv**: For building local Python bindings (`pip install maturin` or `uv`).

#### Clone & Build
```bash
git clone https://github.com/versorc/versorc-math.git
cd versorc-math

# Build Rust core
cargo build

# Run all 58+ unit and integration tests
cargo test --all-features
```

#### Build Python Bindings Locally
```bash
# Using maturin
maturin develop --release --features python

# Run Python test battery
pytest python/tests/ -v
```

---

## 🛠️ Contribution Workflow

1. **Fork the Repository**: Click the "Fork" button on [GitHub](https://github.com/versorc/versorc-math).
2. **Create a Topic Branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Write Tests First**: Every new feature or operator should include ground-truth test assertions verifying algebraic axioms (e.g., associativity, metric signatures, grade projection consistency).
4. **Ensure Code Quality**:
   ```bash
   cargo fmt --check
   cargo clippy --all-features
   cargo test --all-features
   pytest python/tests/ -v
   ```
5. **Commit Your Changes**: Use clear, conventional commit messages (e.g., `feat(cga): add circle-sphere tangent intersection`, `docs(poincare): clarify gyrovector derivation`).
6. **Open a Pull Request**: Submit your PR on GitHub! Be sure to describe your changes, the mathematical rationale, and any relevant literature references.

---

## 💬 Getting Help & Discussions

- **GitHub Discussions**: Share ideas, ask mathematical questions, or suggest new architectures.
- **Issues**: Report bugs or suggest concrete improvements using the issue templates.
- **Code of Conduct**: All participants are expected to follow our [Code of Conduct](CODE_OF_CONDUCT.md).

Thank you for helping build the future of Clifford Geometric Computing!
