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

### 3D Interactive Elements (Three.js + GSAP + React)
- **Object:** A low-poly rotating neural-network node cluster (react-three-fiber `InstancedMesh` spheres connected by `Line` segments) centered in the slide, Navy nodes (`#14213d`) with Coral accent edges (`#ff6b5b`).
- **Animation:** GSAP `to()` tween auto-rotating the cluster 360° over 12 seconds, `repeat: -1`, `ease: "linear"`; individual nodes pulse scale with a staggered GSAP `fromTo()` (0.8 → 1.2 → 0.8) on load.
- **Interaction:** `OrbitControls` disabled (auto-spin only); on click, nodes briefly flash Coral.
- **React integration:** Wrap scene in `<Canvas>` with `shadowMap` enabled; ambient light (Cream) + directional light (Navy) from top-left.

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

### 3D Interactive Elements (Three.js + GSAP + React)
- **Object:** Four floating sharp-cornered 3D cards (`BoxGeometry` with `EdgesGeometry` stroked outlines) arranged in a shallow arc, each Navy (`#14213d`) with white Geist Mono feature labels; cards represent: Header-Only, OpenMP+OpenBLAS, Full Training Loop, Extensible.
- **Animation:** GSAP `fromTo()` floating bobbing motion (y-axis ±0.3 units, 3s duration, `yoyo: true`, `repeat: -1`) staggered per card; on hover, card scales up 1.2× with Coral glow (`pointLight` intensity tween).
- **Interaction:** Pointer events via `react-three-drei` `<HoverProvider>`; clicking a card highlights its Coral border (4px stroke) and fades others to 40% opacity.
- **React integration:** `<Canvas>` with `camera={{ position: [0, 2, 6] }}`; `useFrame` for subtle drift; environment map in Cream tone.

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

### 3D Interactive Elements (Three.js + GSAP + React)
- **Object:** Three tall sharp-cornered 3D pillars (`BoxGeometry` with `EdgesGeometry` stroked outlines), Navy (`#14213d`) with Coral (`#ff6b5b`) top caps, arranged left-to-right; each pillar labeled in Geist Mono: Header-Only, Extension Points, Minimal Core.
- **Animation:** GSAP `from()` scale-in from 0 to 1 on load with `stagger: 0.2`, `ease: "back.out(1.7)"`; Coral caps pulse brightness with GSAP `to()` tween (`repeat: -1`, `yoyo: true`, 2s).
- **Interaction:** Hovering a pillar expands its width slightly and shows a Coral tooltip (HTML overlay positioned via `useSpring` / CSS); clicking a pillar centers camera on it with GSAP `to()` camera position tween.
- **React integration:** `<Canvas>` with `camera={{ position: [0, 3, 8] }}`; `useFrame` for slow auto-rotation (0.5°/frame); `useThree` for viewport-aware scaling.

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

### 3D Interactive Elements (Three.js + GSAP + React)
- **Object:** A floating 3D matrix grid (`InstancedMesh` of small Navy boxes representing cells) showing two matrices A and B side by side, with connecting lines (`Line`) from row of A to column of B converging into a result cell in matrix C; Coral dots highlight active computation cells.
- **Animation:** GSAP `to()` wave animation sweeping across matrix A cells (`stagger: 0.01`, `yoyo: true`, `repeat: -1`), each cell scaling up as the wave passes; connecting lines draw themselves with GSAP `drawSVG` style via `useFrame` progress tween.
- **Interaction:** Clicking a cell isolates its row/column (others fade to 20% opacity); slider (HTML range input) adjusts matrix size and regenerates the `InstancedMesh` count.
- **React integration:** `<Canvas>` with `camera={{ position: [6, 6, 6] }}` and `OrbitControls` enabled; `useFrame` for the wave phase offset; all geometry in Coral/Navy/Cream palette.

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

### 3D Interactive Elements (Three.js + GSAP + React)
- **Object:** A 3D racing bar chart: two groups of sharp-cornered 3D bars (`BoxGeometry`), left group Navy (`#14213d`) labeled "Single-threaded", right group Coral (`#ff6b5b`) labeled "OpenMP + OpenBLAS"; bars represent operations (Matrix dot, ReLU, Full epoch), with height proportional to time (taller = slower).
- **Animation:** GSAP timeline racing animation: all bars grow from 0 to their final height simultaneously, but Coral bars reach full height ~15× faster than Navy bars (staggered duration difference); a Coral "finish line" plane slams down when OpenMP bars complete.
- **Interaction:** Hovering a bar shows a tooltip with exact ms value (HTML overlay); a toggle button switches between "Training epoch" and "Single operation" views, regenerating bar heights with GSAP scale transitions.
- **React integration:** `<Canvas>` with `camera={{ position: [0, 4, 10] }}`; `useFrame` for subtle camera drift; `useThree` for responsive resize; Coral/Navy color-coded `MeshStandardMaterial`.

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

### 3D Interactive Elements (Three.js + GSAP + React)
- **Object:** A 3D neural network visualization: input layer (2 Navy spheres), two hidden layers (32 and 16 smaller spheres), output layer (1 Coral sphere); nodes connected by thin stroked lines (`Line` with `BufferGeometry`) representing weights; `MeshStandardMaterial` with Navy/Coral color coding per layer.
- **Animation:** GSAP signal pulse animation: a bright Coral dot travels along edges from input to output using `MotionPathPlugin` on a `useFrame`-driven tween; nodes flash (scale 1→1.4→1) as the pulse passes through them; `repeat: -1`, `duration: 3`.
- **Interaction:** Clicking a node highlights all connected edges (Coral glow) and fades unrelated nodes to 30% opacity; a dropdown (HTML) changes network depth and regenerates the node/edge layout with GSAP scale transitions.
- **React integration:** `<Canvas>` with `camera={{ position: [0, 4, 12] }}`; `OrbitControls` enabled; `useFrame` for pulse phase; `react-three-drei` `<Text>` labels for layer names.

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

### 3D Interactive Elements (Three.js + GSAP + React)
- **Object:** Two side-by-side 3D function graphs (`PlaneGeometry` with `MeshStandardMaterial` displacement): left graph shows ReLU (flat at 0 for x<0, linear ramp for x>0 in Navy), right graph shows MSE parabolic bowl (`z = (x-y)²` in Coral); both grids have stroked `EdgesGeometry` outlines.
- **Animation:** GSAP `to()` tween animates the displacement height of the ReLU graph from flat to full slope on load (`stagger: 0.1`, `ease: "power2.out"`); the MSE bowl rotates slowly via `useFrame` (0.3°/frame around Y axis).
- **Interaction:** A slider (HTML overlay) adjusts the ReLU alpha (leakiness) from 0 to 1, morphing the graph surface with GSAP scale transitions; clicking a point on the MSE bowl drops a Coral sphere marker showing the current loss value.
- **React integration:** `<Canvas>` with `camera={{ position: [0, 4, 6] }}`; `useFrame` for bowl rotation; `useThree` for responsive resize; colors strictly Navy/Coral/Cream.

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

### 3D Interactive Elements (Three.js + GSAP + React)
- **Object:** Four floating 3D optimizer chips (`BoxGeometry` with `EdgesGeometry` stroked outlines), Navy (`#14213d`) with white Geist Mono labels: SGD, Adagrad, RMSprop, Adam; each chip has a Coral sphere "particle" orbiting it representing momentum accumulation.
- **Animation:** GSAP `to()` animation moves each Coral particle in a distinct path: SGD (straight line), Adagrad (shrinking spiral), RMSprop (decaying sine wave), Adam (combined spiral+sine); particles loop with `repeat: -1`, staggered start delays.
- **Interaction:** Clicking a chip freezes the other three particles and expands the selected chip's particle trail into a visible `TubeGeometry` path showing its full trajectory; HTML labels show the optimizer name and update rule.
- **React integration:** `<Canvas>` with `camera={{ position: [0, 3, 8] }}`; `useFrame` for particle phase offsets; `react-three-drei` `<Text>` for labels; strict Navy/Coral palette.

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

### 3D Interactive Elements (Three.js + GSAP + React)
- **Object:** A 3D cyclic training loop diagram: three sharp-cornered 3D pillars (`BoxGeometry` with `EdgesGeometry`) labeled Forward, Backward, Update in Geist Mono; a Coral sphere travels around the triangle connecting them via `CatmullRomCurve3` path; `MeshStandardMaterial` Navy pillars with Coral path line.
- **Animation:** GSAP `to()` moves the sphere along the path (`duration: 4`, `repeat: -1`, `ease: "linear"`); each time the sphere reaches a pillar, the pillar flashes (scale 1→1.3→1) and emits a Coral particle burst via `useFrame`-driven `InstancedMesh` scale pulses.
- **Interaction:** Clicking a pillar pauses the sphere and displays the step's description in an HTML overlay; clicking again resumes; speed slider adjusts GSAP `timeScale()`.
- **React integration:** `<Canvas>` with `camera={{ position: [0, 5, 8] }}`; `OrbitControls` enabled; `useFrame` for particle emission; colors strictly Navy/Coral/Cream.

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

### 3D Interactive Elements (Three.js + GSAP + React)
- **Object:** A 3D data pipeline: a Navy cylinder (raw dataset) on the left feeds into a Coral cube (`StandardScaler`) in the center, which outputs to a Navy plane (scaled train/test splits) on the right; small floating spheres represent individual data samples moving through the pipeline; `EdgesGeometry` stroked outlines on all solids.
- **Animation:** GSAP `to()` moves sample spheres from cylinder → cube → plane in a staggered sequence (`stagger: 0.05`, `duration: 6`, `repeat: -1`); the Coral cube pulses (scale 1→1.2→1) when spheres pass through; spheres change color from Navy to Coral as they exit.
- **Interaction:** Clicking the cylinder regenerates a new random set of sample spheres; clicking the cube toggles "fit mode" where the cube temporarily expands to encompass all spheres; HTML dropdown selects dataset type (spiral/housing/audio/SVM).
- **React integration:** `<Canvas>` with `camera={{ position: [0, 4, 10] }}`; `useFrame` for sample sphere bobbing; `react-three-drei` `<Text>` labels for each stage; strict Navy/Coral/Cream palette.

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

### 3D Interactive Elements (Three.js + GSAP + React)
- **Object:** Split 3D scene: left side shows a 2D scatter plot extruded into 3D (`InstancedMesh` cylinders, Navy/Coral based on SVM class ±1) with a floating 3D decision plane (`PlaneGeometry`, semi-transparent Coral) cutting through; right side shows a 3D fluid grid (`InstancedMesh` small cubes colored by vorticity magnitude) with flow lines (`Line` segments) representing velocity vectors.
- **Animation:** GSAP `to()` animates the SVM decision plane rotating and translating to find the best split (`duration: 5`, `repeat: -1`, `yoyo: true`); the Navier-Stokes grid cells pulse color intensity with `useFrame`-driven color lerp based on a noise function; flow lines draw themselves progressively.
- **Interaction:** Clicking the decision plane snaps it to the optimal position and highlights correctly classified spheres in Coral; dragging (OrbitControls) rotates both views independently; HTML metric badges update live as animations progress.
- **React integration:** `<Canvas>` split via two `<canvas>` elements or a single `<Canvas>` with two `useRef` groups; `camera={{ position: [0, 3, 6] }}` for SVM side, `camera={{ position: [4, 4, 4] }}` for Navier-Stokes side; `useFrame` for animation loops; colors strictly Navy/Coral/Cream.

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

### 3D Interactive Elements (Three.js + GSAP + React)
- **Object:** Three sharp-cornered 3D floating steps (`BoxGeometry` with `EdgesGeometry` stroked outlines), Navy (`#14213d`) with Coral step numbers (1, 2, 3) floating above each; step labels in Geist Mono: "Create Header", "Register in Factory", "Export in cpptorch.h"; thin Coral stroked connector lines between steps.
- **Animation:** GSAP `from()` scale-in from 0 on load with `stagger: 0.3`, `ease: "back.out(1.7)"`; Coral step numbers bounce with GSAP `to()` (`yoyo: true`, `repeat: -1`, 1.5s duration); connector lines draw themselves via `useFrame` progress tween.
- **Interaction:** Clicking a step expands it to 2× size, fades others to 30% opacity, and shows the corresponding code snippet in an HTML overlay; a "Reset" button returns all steps to original size with GSAP scale transition.
- **React integration:** `<Canvas>` with `camera={{ position: [0, 3, 8] }}`; `useFrame` for gentle float animation; `react-three-drei` `<Text>` for labels; strict Navy/Coral/Cream palette.

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

### 3D Interactive Elements (Three.js + GSAP + React)
- **Object:** A central floating 3D CppTorch logo mark: interconnected Navy spheres (nodes) connected by Coral stroked lines forming a neural network node cluster in the center; three smaller orbiting rings around it representing Taher, Kashif, and Yahya; `EdgesGeometry` stroked outlines on all geometry.
- **Animation:** GSAP `to()` auto-rotates the central cluster 360° over 10 seconds (`repeat: -1`, `ease: "linear"`); the three orbiting rings rotate at different speeds via `useFrame`; nodes pulse scale with staggered GSAP `fromTo()` (0.9→1.3→0.9) on load.
- **Interaction:** Clicking the cluster triggers a Coral particle explosion (`InstancedMesh` of small spheres expanding outward); hovering any orbiting ring highlights the corresponding author name in an HTML overlay; the repository link glows Coral on hover.
- **React integration:** `<Canvas>` with `camera={{ position: [0, 0, 7] }}`; `OrbitControls` disabled (controlled auto-rotation); `useFrame` for ring orbits and pulse phases; ambient light (Cream) + directional light (Navy) for soft depth; strict Navy/Coral/Cream palette.

### Connection to next slide
None — this is the final slide. End the presentation here or loop back to Slide 1.


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
