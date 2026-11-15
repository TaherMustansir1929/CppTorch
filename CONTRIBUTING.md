# Contributing to CppTorch

Thanks for taking the time to contribute! This document explains how to set up
the project, the conventions we follow, and — most importantly — **how to add a
new feature without breaking the core of the library**.

- [1. Getting set up](#1-getting-set-up)
- [2. Repository layout](#2-repository-layout)
- [3. The extension-point contract](#3-the-extension-point-contract)
- [4. Adding a new component](#4-adding-a-new-component)
- [5. Coding conventions](#5-coding-conventions)
- [6. Testing](#6-testing)
- [7. Pull request checklist](#7-pull-request-checklist)
- [8. Reporting bugs](#8-reporting-bugs)

---

## 1. Getting set up

### Prerequisites

| Dependency | Version |
|---|---|
| C++ compiler | GCC 11+, Clang 14+, MSVC 19.30+ |
| CMake | 3.21+ |
| OpenMP | bundled with most compilers |
| BLAS | OpenBLAS recommended |

```bash
# Debian / Ubuntu
sudo apt install build-essential cmake libopenblas-dev

# macOS
brew install cmake openblas

# Windows (vcpkg)
vcpkg install openblas
```

### Building

```bash
git clone https://github.com/TaherMustansir1929/CppTorch.git
cd CppTorch
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### Running the tests

```bash
cd build && ctest --output-on-failure
```

The test suite is a single binary with a dispatcher. To run one test by hand:

```bash
./build/tests/cpptorch_tests full_pass_with_optimizer_adam
```

---

## 2. Repository layout

```
CppTorch/
├── CMakeLists.txt              Library target, options, install/export rules
├── cmake/                      Package config template
├── include/cpptorch/           Public headers — the library itself
│   ├── cpptorch.h              Umbrella header
│   ├── core/                   Matrix, MatrixView, NumCpp
│   ├── layers/                 dense, dropout, activations, layer wrapper
│   ├── losses/                 loss functions
│   ├── optimizers/             optimizers
│   ├── utils/                  scaler, exceptions, resources, helpers
│   ├── datasets.h              dataset loaders
│   └── model.h                 high-level training API
├── examples/                   Runnable demos
├── tests/                      Test suite (CTest)
├── docs/                       Architecture and how-to guides
└── resources/                  Bundled CSV datasets
```

**Rule of thumb:** everything under `include/cpptorch/` is the public API and
must stay header-only, dependency-free (apart from BLAS/OpenMP) and free of
I/O side effects other than the dataset loaders.

---

## 3. The extension-point contract

CppTorch separates the *core* (the maths and the training loop) from the
*components* (activations, losses, optimizers). The core never needs to change
when a component is added.

| Component | Base class | Enum | Factory (plug-in point) |
|---|---|---|---|
| Activation | `cpptorch::Activation` | `cpptorch::ActivationType` | `cpptorch::make_activation` in `layers/activations/activation_registry.h` |
| Loss | `cpptorch::Loss` | `cpptorch::LossType` | `cpptorch::make_loss` in `losses/loss_registry.h` |
| Optimizer | `cpptorch::Optimizer` | `cpptorch::OptimizerType` | `cpptorch::make_optimizer` in `optimizers/optimizer_registry.h` |
| Dataset | — | — | add a static function to `cpptorch::Datasets` |

The guarantees the core relies on:

1. **`model.h` never changes** when a component is added. The training loop
   only calls the factories.
2. A component is **one self-contained header**. It has no knowledge of
   `Model`, and the core has no knowledge of the component beyond its enum
   value.
3. A component only depends on the base class it derives from and on
   `core/matrix.h`. It must not include `model.h`.
4. The enum value is the **only** identifier the core needs. Adding a value is
   backwards compatible.

---

## 4. Adding a new component

### 4.1 A new activation

**Step 1 — create the header** `include/cpptorch/layers/activations/leaky_relu.h`:

```cpp
#pragma once

#include "cpptorch/layers/activations/activation.h"

#include <cmath>

namespace cpptorch
{

/**
 * @brief Leaky ReLU activation, f(x) = x for x >= 0, alpha * x otherwise.
 */
class Activation_LeakyReLU : public Activation
{
    double m_alpha = 0.01;

public:
    explicit Activation_LeakyReLU(double alpha = 0.01) : m_alpha(alpha) {}

    void forward(const Matrix &inputs, bool cache_inputs = true) override
    {
        if (cache_inputs)
            this->inputs = inputs;
        this->output.resize(inputs.rows(), inputs.cols());
        for (size_t i = 0; i < inputs.size(); ++i)
            this->output[i] = inputs[i] > 0.0 ? inputs[i] : m_alpha * inputs[i];
    }

    void backward(const Matrix &dvalues) override
    {
        this->dinputs.resize(dvalues.rows(), dvalues.cols());
        for (size_t i = 0; i < this->output.size(); ++i)
            this->dinputs[i] = this->output[i] > 0.0 ? dvalues[i] : m_alpha * dvalues[i];
    }
};

} // namespace cpptorch
```

**Step 2 — add the enum value** in
`include/cpptorch/layers/activations/activation.h`:

```cpp
enum class ActivationType : std::uint8_t
{
    Linear = 0,
    ReLU = 1,
    Softmax = 2,
    LeakyReLU = 3,   // <- new
};
```

**Step 3 — register it** in
`include/cpptorch/layers/activations/activation_registry.h`:

```cpp
case ActivationType::LeakyReLU:
    return std::make_unique<Activation_LeakyReLU>();
```

**Step 4 — export it** by adding the header to `include/cpptorch/cpptorch.h`:

```cpp
#include "cpptorch/layers/activations/leaky_relu.h"
```

**Step 5 — test it.** Add `tests/full_pass_with_leaky_relu.cpp` following the
pattern of the other tests, register the file in `tests/CMakeLists.txt`, add the
name to `CPPTORCH_TEST_NAMES` and to the registry in `tests/main.cpp`.

That is it — `model.h` was not touched.

### 4.2 A new loss

Identical shape, with `cpptorch::Loss` as the base class,
`cpptorch::LossType` as the enum and `cpptorch::make_loss` as the factory.
The loss must implement `forward()` (per-sample losses) and `backward()`
(gradient with respect to the prediction).

To use a custom loss with a `Model`, either add it to the registry and map it
from a `ModelType`, or call `Model::set_loss()` with your own instance.

### 4.3 A new optimizer

Identical shape, with `cpptorch::Optimizer` as the base class,
`cpptorch::OptimizerType` as the enum and `cpptorch::make_optimizer` as the
factory. The optimizer must implement `update_params(Layer_Dense &)` and may
override `pre_update_params()` to customise the learning-rate schedule.

### 4.4 A new dataset

Add a static function to `cpptorch::Datasets` in `include/cpptorch/datasets.h`:

```cpp
static std::tuple<Matrix, Matrix> my_dataset(const std::string &filename = "my_data.csv")
{
    std::ifstream file = open_dataset(filename);
    if (!file.is_open())
        throw std::runtime_error("[Datasets::my_dataset] Failed to open CSV file: " + filename);
    // ... parse ...
    return {X, y};
}
```

`open_dataset()` resolves the path through `cpptorch::Resources`, so the file
is looked up in the resources directory, in the working directory, and finally
by bare name.

### 4.5 A new layer

A layer that is not "dense + activation" needs a new `Layer_Wrapper`. Keep the
change inside `include/cpptorch/layers/layer.h` and preserve the contract the
training loop depends on:

- `forward()` computes and caches the output,
- `backward()` fills `dinputs()` with the gradient with respect to the input,
- `dense()` exposes the parameters for the optimizer.

If the new layer does not own parameters, return an empty dense layer.

---

## 5. Coding conventions

The project follows the existing `.clang-format` and `.clang-tidy`
configurations. Run clang-format before committing:

```bash
clang-format -i include/cpptorch/**/*.h tests/*.cpp examples/*.cpp
```

Rules we do enforce:

- **Namespace** — everything public lives in `namespace cpptorch`. Nothing may
  leak into the global namespace.
- **Header guards** — `#pragma once`.
- **Self-contained headers** — every header includes what it uses and compiles
  on its own.
- **Documentation** — public classes and methods carry Doxygen comments
  (`@brief`, `@param`, `@return`, `@throws`).
- **Ownership** — `std::unique_ptr` for polymorphic bases (`Activation`, `Loss`,
  `Optimizer`); `std::shared_ptr` is not used.
- **Errors** — throw `std::invalid_argument` for bad arguments,
  `std::runtime_error` for runtime failures, `std::out_of_range` for indexing.
- **No exceptions in hot loops** — validation happens at the API boundary.
- **Deterministic tests** — a new test must not depend on wall-clock time or
  on the machine's core count.

### Performance guidelines

- Prefer `*_into` variants (`NumCpp::dot_into`, `NumCpp::sum_into`,
  `NumCpp::binomial_into`) in hot paths to avoid allocations.
- Use `#pragma omp parallel for simd if (size > THRESHOLD)` for element-wise
  loops. The threshold avoids spawning threads for tiny buffers.
- Reuse buffers instead of allocating inside loops.

---

## 6. Testing

Every change must keep the suite green:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
cd build && ctest --output-on-failure
```

A new component should come with:

1. A unit test in `tests/` that exercises the forward and backward passes.
2. A numerical check against an analytically known result where possible.
3. An entry in `tests/main.cpp` and in `tests/CMakeLists.txt`.

CI runs the suite on Linux, macOS and Windows for every push and pull request.

---

## 7. Pull request checklist

- [ ] The build is clean (`cmake --build build`).
- [ ] `ctest` passes.
- [ ] New public API is documented with Doxygen comments.
- [ ] New public API is mentioned in `README.md` (API reference section).
- [ ] `model.h` and `core/` are unchanged unless the change is the point of the
      PR — component additions must not touch the core.
- [ ] `clang-format` has been run on the changed files.
- [ ] `CHANGELOG.md` has an entry under *Unreleased*.

---

## 8. Reporting bugs

Open an issue using the **Bug report** template and include:

- the CppTorch version or commit,
- the compiler, OS and CMake version,
- a **minimal** reproducer,
- the full output, including any compiler warnings.

For numerical bugs, please include the input data and the expected versus the
actual result.
