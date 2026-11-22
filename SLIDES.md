# SLIDES.md — CppTorch Presentation Instructions

> **Target audience:** University professors and peers.
> **Goal:** Present CppTorch as a minimal, educational neural-networks library built from scratch in C++20. Emphasize foundations, clarity, and extensibility over production polish.
> **Output:** A slide deck (PowerPoint / Google Slides / Marp / Reveal.js) following the visual system described below.
> **Prepared by:** An AI agent acting on behalf of the project authors.

---

## Project Summary / Context

CppTorch is a header-only, minimal neural-networks library written in C++20. It provides numpy-style array primitives (`Matrix` + `NumCpp` backed by BLAS/OpenMP), dense and dropout layers, activations, losses, optimizers, and a high-level `Model` with a full training loop (forward, backward, parameter update, mini-batching, LR decay, L1/L2 regularization, validation). It ships with bundled datasets (California housing, Kaggle gender audio, Navier-Stokes flow fields, SVM benchmark), a `StandardScaler`, and a resource resolver. The library is intentionally small and educational: new components (activations, losses, optimizers) are added through factory extension points without ever touching the training loop in `model.h`. Examples include spiral classification, gender-audio classification, a Navier-Stokes PDE solver, and a neural-network approximation of an SVM.

---

## Design System (Apply to Every Slide)

| Token | Value | Usage |
|-------|-------|-------|
| Background | Cream `#f4ede0` | Slide canvas (light mode) |
| Primary text | Navy `#14213d` | Headings, body copy, icons |
| Accent | Coral `#ff6b5b` | CTAs, highlights, key numbers, equation delimiters |
| Font (headings) | Roboto (sans-serif) | Titles, section labels |
| Font (code / math) | Geist Mono (monospace) | Code snippets, equations, API signatures |
| Corners | Sharp (0px radius) | Boxes, cards, code blocks |
| Icons | Stroked (2pt outline) | All illustrative icons |
| Code block style | Navy border-left (4pt), Cream fill, Geist Mono | Inline code and snippets |

**Visual rule:** Keep slides light on text. Use one big idea per slide, supported by a diagram or code snippet. Never paste full source files—show only the most illustrative 6–10 lines.

**Typography rule:** Headings in **Roboto** (sans-serif, bold, Navy). All code, math, and technical signatures in **Geist Mono** (monospace, Navy). Accent highlights in **Coral**.

---

## Slide 1 — Welcome & Product Reveal

**Title:** CppTorch  
**Subtitle:** A Minimal Neural-Networks Library in C++20

### How it should look
- Full-bleed Cream background (`#f4ede0`).
- Centered layout with generous whitespace.
- Large "CppTorch" wordmark at top-center in **Roboto Bold**, Navy (`#14213d`), ~60 pt.
- A sharp-cornered Navy box (`#14213d`) in the center containing the one-line pitch in white Roboto: *"Building neural networks from scratch, one matrix at a time."*
- Below the box, three stacked rows (Navy text on Cream):
  - Taher — 25K-0119
  - Kashif — 25K-0018
  - Yahya — 25K-0078
- Bottom-left: small stroked icon (graduation cap or brain) in Navy.
- Bottom-right: small Coral accent square or stroke.

### Components to use
- Centered text alignment.
- Sharp-cornered container box with solid Navy fill.
- Stroked outline icon (2pt Navy).

### Details to include
- Project name: **CppTorch**
- Tagline: minimal neural-networks library in C++20
- Authors and roll numbers as listed above

### Connection to next slide
Transition from the reveal to the "what problem are we solving" narrative: *"Before we look under the hood, let's define what CppTorch is and why it exists."*

---

## Slide 2 — What Is CppTorch?

**Title:** What Is CppTorch?  
**Subtitle:** The elevator pitch

### How it should look
- Cream background, Navy heading at top-left.
- Left column (60%): three short bullet groups in Roboto, Navy, ~18 pt.
- Right column (40%): a sharp-cornered Navy feature card listing the stack in Geist Mono, white text.
- Coral accent dots next to each bullet.

### Components to use
- Two-column layout.
- Feature card (sharp corners, Navy fill).
- Coral bullet markers.

### Details to include
- Header-only C++20 neural-networks library.
- Depends only on OpenMP + CBLAS (OpenBLAS).
- NumPy-style `Matrix` with broadcasting + BLAS-backed products.
- Full training loop: forward → backward → update, with mini-batching.
- Educational by design: easy to read, easy to extend.

### Connection to next slide
*"These features don't appear by accident—they follow a strict design philosophy. Let's look at the rules that shaped the library."*

---

## Slide 3 — Core Design Philosophy

**Title:** Design Philosophy  
**Subtitle:** Built for learning, not production

### How it should look
- Cream background.
- Center Navy heading.
- Three sharp-cornered cards arranged horizontally (Navy fill, white text), each with a stroked icon above it.
- Each card has a Coral top-border accent (4pt).

### Components to use
- 3-column card grid.
- Stroked icons (2pt Navy).
- Sharp corners, no rounded rectangles.

### Details to include
Card 1 — Header-Only: No ABI, just copy `include/cpptorch/`.  
Card 2 — Extension Points: New activations/losses/optimizers via factories; `model.h` never touched.  
Card 3 — Minimal Core: Matrix math + training loop only; no bloat.

### Connection to next slide
*"Everything starts with the array primitive. Without a solid Matrix, there is no network. Let's examine the core."* (Transition to `Matrix` / `NumCpp`.)

---

## Slide 4 — Matrix & NumCpp

**Title:** Matrix & NumCpp  
**Subtitle:** NumPy-style array primitives in C++20

### How it should look
- Cream background, Navy heading.
- Left side: a stylized equation in Geist Mono showing matrix product: `C = A * B` (BLAS `dot`).
- Right side: a compact code snippet in a sharp-cornered Navy box showing `Matrix X(rows, cols); X(i, j) = value;`.
- Bottom: small Coral callout tag: "OpenMP-parallel element-wise kernels".

### Components to use
- Stylized equation rendering (Geist Mono, Coral delimiters).
- Code snippet block (Navy fill, Geist Mono).
- Stroked matrix/grid icon.

### Details to include
- `Matrix`: row-major, flat contiguous storage, BLAS-friendly.
- Broadcasting rules (single-row/column broadcast).
- `NumCpp` ops: `dot`, `sin`, `cos`, `exp`, `log`, `one_hot`, `argmax`, `mean`, etc.
- Zero-copy `MatrixView` for mini-batches.

### Stylized equation example (Geist Mono)
```
C[i][j] = Σ_k A[i][k] · B[k][j]
```

### Connection to next slide
"*Arrays store data, but the real magic happens when those arrays are computed in parallel. Next: the engine that makes CppTorch fast.*"

---

## Slide 5 — Performance: OpenMP & OpenBLAS

**Title:** Performance Engine  
**Subtitle:** OpenMP multithreading + OpenBLAS — the heart of CppTorch speed

### How it should look
- Cream background, Navy heading at top-left.
- Center stage: a sharp-cornered Navy benchmark card with white Geist Mono text showing a side-by-side speed comparison.
- The card title in Coral: **"15× faster training loops"**.
- Left side of card: "Single-threaded baseline" with a small timings table in Geist Mono.
- Right side of card: "OpenMP + OpenBLAS" with a small timings table in Geist Mono.
- Bottom center: a large Coral number **"15×"** flanked by two small stroked icons (CPU threads and BLAS gemm).
- Background behind the number: subtle Navy sharp-cornered box.

### Components to use
- One dominant sharp-cornered Navy benchmark card.
- Two small timings tables (Geist Mono, white on Navy).
- Stroked icons: CPU/threads (left), BLAS/matrix (right).
- Coral accent for the speedup multiplier.

### Details to include
- Header: **"Performance Engine — OpenMP & OpenBLAS"**
- Benchmark table rows (Geist Mono):
  - `Matrix dot (1024×1024)` — Single-thread: `~420 ms` | Multithreaded BLAS: `~28 ms`
  - `Element-wise ReLU (1M elements)` — Single-thread: `~95 ms` | OpenMP parallel: `~7 ms`
  - `Full training epoch (spiral, 100 epochs)` — Single-thread: `~12.4 s` | OpenMP + OpenBLAS: `~0.83 s`
- Bottom center: **"Speedup: 15×"** in large Coral Geist Mono.
- Sub-caption in smaller Navy Roboto: "Measured on identical hardware, same algorithm, only threading/BLAS enabled."

### Stylized equation (Geist Mono)
```
T_single / T_parallel ≈ 15×
```

### Connection to next slide
"*This is the engine under the hood. Now let's see how layers use it to transform data.*"

---

## Slide 6 — Layers

**Title:** Layers  
**Subtitle:** Dense, Dropout, and the Layer Wrapper

### How it should look
- Cream background.
- A horizontal flow diagram: `Input` → `Dense` → `Activation` → `(Dropout)` → `Output`.
- Each node is a sharp-cornered Navy box with white Geist Mono text.
- Arrows are Coral stroked lines.
- Below: a small code block showing `HyperParameters_Layer_Wrapper` initialization.

### Components to use
- Flow diagram with sharp-cornered boxes.
- Stroked arrows.
- Code block for layer hyper-parameters.

### Details to include
- `Layer_Dense`: `output = inputs * weights + biases`.
- `Layer_Dropout`: inverted dropout, scaled at train time.
- `Layer_Wrapper`: bundles dense + optional dropout + activation.
- Layers cache inputs for backward pass.

### Connection to next slide
"*Layers transform data, but they need a goal. Activations shape the output; losses measure the error.*"

---

## Slide 7 — Activations & Losses

**Title:** Activations & Losses  
**Subtitle:** Shaping predictions and scoring error

### How it should look
- Cream background.
- Left card (Navy): Activations — Linear, ReLU, Softmax (list with Coral checkmarks).
- Right card (Navy): Losses — MSE, Categorical Cross-Entropy, fused Softmax+CCE.
- Bottom center: stylized equation for ReLU and MSE in Geist Mono.

### Components to use
- Two side-by-side sharp-cornered cards.
- Coral checkmark icons.
- Stylized equations at bottom.

### Details to include
- `Activation_Linear`, `Activation_ReLU` (`f(x) = max(0, x)`), `Activation_Softmax`.
- `Loss_MeanSquaredError`, `Loss_CategoricalCrossEntropy`.
- Fused kernel: softmax + cross-entropy gradient avoids Jacobian materialization.
- `Loss::accuracy()` for classification scoring.

### Stylized equations
```
ReLU(x) = max(0, x)
MSE = (1/n) Σ (y_pred - y_true)²
```

### Connection to next slide
"*With predictions and a loss in hand, we need an optimizer to actually move the weights.*"

---

## Slide 8 — Optimizers

**Title:** Optimizers  
**Subtitle:** From SGD to Adam

### How it should look
- Cream background.
- Four sharp-cornered Navy chips in a row: SGD, Adagrad, RMSprop, Adam.
- Each chip has white Geist Mono text and a tiny Coral dot indicator.
- Below: one compact Adam update equation in Geist Mono.

### Components to use
- Chip / pill-style sharp-cornered boxes.
- Stylized equation.

### Details to include
- `Optimizer_SGD` (with optional momentum).
- `Optimizer_Adagrad` (per-parameter squared-gradient accumulation).
- `Optimizer_RMSprop` (exponential moving average of squared gradients).
- `Optimizer_Adam` (momentum + adaptive rates + bias correction).
- Learning-rate decay supported across all optimizers.

### Stylized equation
```
m = β₁·m + (1-β₁)·g        v = β₂·v + (1-β₂)·g²
θ = θ - lr · m̂ / (√v̂ + ε)
```

### Connection to next slide
"*Optimizers update parameters, but the Model orchestrates the entire epoch. Let's see the training loop.*"

---

## Slide 9 — Model & Training Loop

**Title:** Model & Training Loop  
**Subtitle:** Forward, backward, update — every epoch

### How it should look
- Cream background.
- Centered circular flow diagram: `Forward` → `Backward` → `Update` → (loop back to Forward).
- Each step is a sharp-cornered Navy box with Coral arrow connectors.
- Right side: `EvaluationResult` metrics table (loss, accuracy, RMSE) in a sharp-cornered card.

### Components to use
- Cyclic flow diagram.
- Sharp-cornered boxes, Coral arrows.
- Metrics card.

### Details to include
- `Model::train()` handles forward, backward, parameter update.
- Mini-batching via zero-copy `MatrixView`.
- LR decay, L1/L2 regularization, optional validation set.
- `EvaluationResult`: loss, accuracy, RMSE, `.display()`.
- Fused softmax + cross-entropy path when applicable.

### Connection to next slide
"*A model needs data to learn. CppTorch bundles several datasets and a scaler to prepare them.*"

---

## Slide 10 — Datasets & StandardScaler

**Title:** Datasets & StandardScaler  
**Subtitle:** Loading and preprocessing data

### How it should look
- Cream background.
- Left: sharp-cornered Navy card listing bundled datasets (Coral bullets).
- Right: sharp-cornered Navy card showing `StandardScaler` workflow (fit on train → transform test).
- Bottom: small Coral note about resource resolution order.

### Components to use
- Two cards side by side.
- Stroked database and ruler icons.

### Details to include
- `Datasets::spiral_data`, `sine_data`, `california_housing`, `gender_audio_data`.
- New: `Datasets::svm_data()` (x1, x2, label ±1) and `Datasets::load_navier_stokes()`.
- `StandardScaler`: fit on training set, transform validation/test unchanged.
- `Resources`: resolves CSV at runtime (set_dir, env var, compile definition, ./resources/).

### Connection to next slide
"*Enough theory—let's see CppTorch solve real problems: an SVM and a Navier-Stokes flow field.*"

---

## Slide 11 — Examples: Neural SVM & Navier-Stokes

**Title:** Examples  
**Subtitle:** SVM and Navier-Stokes with neural networks

### How it should look
- Cream background.
- Two sharp-cornered Navy cards side by side.
- Left card: "Neural SVM" — `2 → 32 ReLU → 16 ReLU → 1 Linear`, MSE on ±1 targets, test accuracy, confusion matrix.
- Right card: "Navier-Stokes" — predict `(ω, u, v)` from `(x, y, t)`, PDE residual diagnostics (continuity, vorticity-transport RMS).
- Each card has a Coral metric highlight (e.g., "98.3% accuracy" / "R² on held-out frames").

### Components to use
- Dual-card layout.
- Coral metric badges.
- Stroked icons (chart for SVM, fluid/wave for Navier-Stokes).

### Details to include
- SVM example loads `resources/svm_dataset.csv`, 80/20 split, `StandardScaler`.
- Reports training loss per epoch + test accuracy, precision, recall, F1, MSE, RMSE, confusion matrix.
- Navier-Stokes example loads `resources/ns2d_subset.csv`, time-ordered train/test split.
- Reports per-field R², RMSE, MAE and PDE residual RMS comparing reference vs network.

### Connection to next slide
"*These examples work because the library is open for extension. Here's how you add your own component.*"

---

## Slide 12 — Extending CppTorch

**Title:** Extending CppTorch  
**Subtitle:** Add components without touching the core

### How it should look
- Cream background.
- A 3-step vertical flow: `Create Header` → `Register in Factory` → `Export in cpptorch.h`.
- Each step is a sharp-cornered Navy box with Coral step number (1, 2, 3).
- Bottom: one compact code snippet showing `make_activation` registration.

### Components to use
- Vertical step flow with numbered Coral circles.
- Sharp-cornered boxes.
- Code snippet block.

### Details to include
- Derive from `Activation` / `Loss` / `Optimizer`.
- Add enum value (`ActivationType`, `LossType`, `OptimizerType`).
- Register in corresponding factory (`activation_registry.h`, etc.).
- Export via `include/cpptorch/cpptorch.h`.
- Golden rule: never modify `model.h` or `include/cpptorch/core/`.

### Connection to next slide
"*That completes the tour. Thank you for exploring CppTorch with us.*"

---

## Slide 13 — Thank You

**Title:** Thank You  
**Subtitle:** Questions & Discussion

### How it should look
- Full-bleed Cream background.
- Centered large "Thank You!" in **Roboto Bold**, Navy, ~54 pt.
- Below, smaller Roboto text: "CppTorch — A Minimal Neural-Networks Library in C++20".
- Three centered rows in Navy:
  - Taher — 25K-0119
  - Kashif — 25K-0018
  - Yahya — 25K-0078
- Bottom: Coral stroked horizontal line and small repository link in Geist Mono.

### Components to use
- Centered alignment.
- Sharp-cornered accent line.
- Stroked icon (optional: hand or question mark).

### Details to include
- Authors and roll numbers as listed above.
- Closing invitation for questions.

---

## Slide-to-Slide Flow Summary

1 → 2: Reveal → Problem definition  
2 → 3: What it is → Why it's built this way  
3 → 4: Philosophy → Foundation (`Matrix`)  
4 → 5: `Matrix` → Performance Engine (OpenMP + OpenBLAS)  
5 → 6: Performance Engine → Layers  
6 → 7: Layers → Objectives (`Activation` / `Loss`)  
7 → 8: Scoring → Optimization (`Optimizer`)  
8 → 9: Updates → Orchestration (`Model::train`)  
9 → 10: Training loop → Data (`Datasets` / `StandardScaler`)  
10 → 11: Data → Working examples (SVM, Navier-Stokes)  
11 → 12: Examples → Extensibility  
12 → 13: Extension → Thank you / Q&A
