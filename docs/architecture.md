# Architecture

This document explains how CppTorch is put together, so that a new contributor
can find the right place to change something.

## Design goals

1. **Small core.** The mathematical core — `Matrix`, `NumCpp` — is small enough
   to read in one sitting and has no dependencies beyond BLAS and OpenMP.
2. **Composable components.** Activations, losses and optimizers are separate,
   interchangeable pieces behind a common base class.
3. **A stable core.** Adding a component must never require touching the
   training loop. This is enforced by routing every component through a
   factory (see [Extension points](#extension-points)).
4. **Header-only.** Users can vendor a single directory, or consume the
   installed CMake package. There is no ABI to track.

## Layers of the library

```
                    ┌──────────────────────────────────────┐
                    │            cpptorch::Model           │   training loop
                    │  train() / predict() / evaluate()     │
                    └───────────────┬──────────────────────┘
                                    │ owns
                    ┌───────────────▼──────────────────────┐
                    │       cpptorch::Layer_Wrapper         │   layer plumbing
                    │  dense → [dropout] → activation        │
                    └───────┬───────────────────┬──────────┘
                            │                   │
          ┌─────────────────▼──────┐  ┌─────────▼──────────────┐
          │ cpptorch::Layer_Dense  │  │ cpptorch::Activation   │   components
          │ cpptorch::Layer_Dropout│  │ Linear / ReLU / Softmax │
          └────────────────────────┘  └────────────────────────┘
          ┌────────────────────────┐  ┌────────────────────────┐
          │    cpptorch::Loss      │  │  cpptorch::Optimizer   │   components
          │ MSE / Categorical CE   │  │ SGD/Adagrad/RMSprop/Adam│
          └────────────────────────┘  └────────────────────────┘
          ┌──────────────────────────────────────────────────────┐
          │  cpptorch::Matrix / cpptorch::MatrixView / NumCpp   │   core maths
          └──────────────────────────────────────────────────────┘
```

### Core maths — `core/`

`cpptorch::Matrix` stores its elements in a single flat `std::vector<double>`.
This is what makes the BLAS calls free: `cblas_dgemm` can be handed a pointer
into the buffer without any copy or stride arithmetic.

`cpptorch::MatrixView` is a non-owning window over a contiguous range of rows.
The training loop uses it to iterate over mini-batches without copying the
data out of the training matrix.

`cpptorch::NumCpp` holds the numpy-style operations. Operations that produce a
new matrix are named after the numpy function; operations that write into a
caller-provided buffer end in `_into`. The `_into` variants exist so that hot
loops can reuse a buffer instead of allocating on every iteration.

### Components — `layers/`, `losses/`, `optimizers/`

Every component family follows the same shape:

| File | Role |
|---|---|
| `<family>.h` | The abstract base class and the `<Family>Type` enum. |
| `<component>.h` | One concrete implementation, deriving from the base class. |
| `<family>_registry.h` | The `make_<family>()` factory that maps enum values to implementations. |

The base classes are deliberately small:

- `Activation` — `forward()` and `backward()`, plus the `output`, `inputs` and
  `dinputs` buffers.
- `Loss` — `forward()` (per-sample losses) and `backward()` (gradient with
  respect to the prediction).
- `Optimizer` — `update_params()` plus the learning-rate schedule hooks.

### The training loop — `model.h`

`cpptorch::Model` owns an ordered list of `Layer_Wrapper` objects, one loss and
one optimizer. `train()` runs three passes per epoch:

1. **Forward** — `Layer_Wrapper::forward()` for each layer. When the last layer
   uses a softmax activation and the loss is categorical cross-entropy, a fused
   gradient kernel is used instead of the generic path.
2. **Backward** — `Layer_Wrapper::backward()` for each layer, in reverse order.
3. **Update** — `Optimizer::update_params()` for each layer.

The loop never names a concrete activation, loss or optimizer; it only uses the
`ActivationType`, `LossType` and `OptimizerType` enums and the corresponding
factories. **This is the property that keeps the core stable.**

### Datasets — `datasets.h`

`cpptorch::Datasets` loads the bundled CSV files and splits them into
train/test/validation sets. File lookup goes through
`cpptorch::Resources`, which resolves the directory at runtime so that an
installed library does not depend on a build-time path.

## Extension points

| Family | Base class | Enum | Factory |
|---|---|---|---|
| Activation | `cpptorch::Activation` | `cpptorch::ActivationType` | `cpptorch::make_activation` |
| Loss | `cpptorch::Loss` | `cpptorch::LossType` | `cpptorch::make_loss` |
| Optimizer | `cpptorch::Optimizer` | `cpptorch::OptimizerType` | `cpptorch::make_optimizer` |
| Dataset | — | — | a static function on `cpptorch::Datasets` |

The full recipe, with code, is in
[CONTRIBUTING.md](../CONTRIBUTING.md#4-adding-a-new-component).

## Performance notes

- **BLAS.** Dense matrix products go through `cblas_dgemm`. The
  `dot_transpose_left` and `dot_transpose_right` variants avoid materialising a
  transposed copy, which matters for the backward pass.
- **OpenMP.** Element-wise kernels use `#pragma omp parallel for simd` with a
  size threshold, so small buffers do not pay the thread-spawn cost.
- **Fusion.** The softmax + cross-entropy pair is fused into a single kernel,
  avoiding the intermediate Jacobian.
- **Zero-copy batches.** Mini-batches are processed through `MatrixView`.

## Dependency policy

CppTorch depends on:

- **BLAS** (through the CBLAS API) — required, for dense matrix products.
- **OpenMP** — required, for the parallel kernels.

Nothing else. In particular the library does not depend on a linear-algebra
package, an image library, or a framework. This keeps the install footprint
small and the build fast.
