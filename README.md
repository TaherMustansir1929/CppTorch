# CppTorch

[![CI](https://github.com/TaherMustansir1929/CppTorch/actions/workflows/ci.yml/badge.svg)](https://github.com/TaherMustansir1929/CppTorch/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://isocpp.org/)

A minimal neural networks library in C++20 with numpy-style array primitives.

CppTorch provides the building blocks of a small deep-learning framework: a
`Matrix` type with numpy-like broadcasting, a `NumCpp` namespace of array
operations backed by BLAS, dense and dropout layers, activations, losses,
optimizers, and a high-level `Model` with training and evaluation loops.

The library is **header-only** and depends only on **OpenMP** and a **CBLAS**
provider (OpenBLAS recommended).

---

## Table of contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
  - [1. Build and install from source](#1-build-and-install-from-source)
  - [2. Consume the installed package](#2-consume-the-installed-package)
  - [3. Use as a subdirectory](#3-use-as-a-subdirectory)
  - [4. Vendor the headers](#4-vendor-the-headers)
  - [5. Package managers](#5-package-managers)
- [Configuration](#configuration)
  - [CMake options](#cmake-options)
  - [BLAS and OpenMP](#blas-and-openmp)
  - [Locating the bundled datasets](#locating-the-bundled-datasets)
- [Quick start](#quick-start)
  - [Using CppTorch in a CMake project](#using-cpptorch-in-a-cmake-project)
  - [Using CppTorch without CMake](#using-cpptorch-without-cmake)
  - [The code](#the-code)
  - [Regression example](#regression-example)
- [Project layout](#project-layout)
- [API reference](#api-reference)
  - [Enums and configuration structs](#enums-and-configuration-structs)
  - [Core: `Matrix`](#core-matrix)
  - [Core: `MatrixView`](#core-matrixview)
  - [Core: `NumCpp`](#core-numcpp)
  - [Layers: `Layer_Dense`](#layers-layer_dense)
  - [Layers: `Layer_Dropout`](#layers-layer_dropout)
  - [Layers: `Layer_Wrapper`](#layers-layer_wrapper)
  - [Activations](#activations)
  - [Losses](#losses)
  - [Optimizers](#optimizers)
  - [Model](#model)
  - [Datasets](#datasets)
  - [Utilities](#utilities)
  - [Extension points](#extension-points)
- [Extending the library](#extending-the-library)
- [Building and testing](#building-and-testing)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- **numpy-style arrays** — `Matrix` with broadcasting, flat contiguous storage
  and BLAS-backed matrix products.
- **Full training loop** — `Model::train()` handles the forward pass, the
  backward pass and the parameter update, with mini-batching, learning-rate
  decay, L1/L2 regularization and an optional validation set.
- **Composable components** — activations, losses and optimizers are separate
  pieces behind a common base class and are plugged in through factories, so a
  new component never requires touching the training loop.
- **Performance** — OpenMP-parallel element-wise kernels, a fused
  softmax + cross-entropy gradient, zero-copy mini-batches and zero-allocation
  buffer reuse in the hot paths.
- **Header-only** — no ABI to track; consume it as a CMake package, a
  subdirectory, or a single copied header folder.
- **Tested** — a 15-test CTest suite that runs on Linux, macOS and Windows.
- **Datasets** — the California housing and Kaggle gender audio datasets ship
  with the library, along with synthetic spiral and sine generators.

---

## Requirements

| Dependency | Version | Notes |
|---|---|---|
| C++ compiler | GCC 11+, Clang 14+, MSVC 19.30+ | C++20 is required |
| CMake | 3.21+ | Only needed if you use the CMake integration |
| OpenMP | — | Usually bundled with the compiler |
| BLAS | — | OpenBLAS recommended (`libopenblas-dev`, `brew install openblas`) |

```bash
# Debian / Ubuntu
sudo apt install build-essential cmake libopenblas-dev

# macOS
brew install cmake openblas

# Windows (vcpkg)
vcpkg install openblas
```

---

## Installation

### 1. Build and install from source

```bash
git clone https://github.com/TaherMustansir1929/CppTorch.git
cd CppTorch

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
cmake --install build --prefix /your/prefix
```

This installs:

```
<prefix>/include/cpptorch/**/*.h
<prefix>/lib/cmake/cpptorch/cpptorchConfig.cmake
<prefix>/lib/cmake/cpptorch/cpptorchConfigVersion.cmake
<prefix>/lib/cmake/cpptorch/cpptorchTargets.cmake
```

### 2. Consume the installed package

```cmake
cmake_minimum_required(VERSION 3.21)
project(my_app LANGUAGES CXX)

find_package(cpptorch 0.1 REQUIRED)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE CppTorch::cpptorch)
```

`CppTorch::cpptorch` carries the include directory, the C++20 requirement and
the OpenMP/BLAS dependencies, so nothing else is needed in your
`CMakeLists.txt`.

`find_package` accepts a version range:

```cmake
find_package(cpptorch 0.1...<1.0 REQUIRED)
```

### 3. Use as a subdirectory

```cmake
add_subdirectory(external/CppTorch)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE CppTorch::cpptorch)
```

When the project is not the top-level one, `cpptorch_BUILD_TESTS`,
`cpptorch_BUILD_EXAMPLES` and `cpptorch_INSTALL` all default to `OFF`, so the
parent build stays clean.

### 4. Vendor the headers

CppTorch is header-only. Copy `include/cpptorch/` into your project and add it
to the include path:

```bash
cp -r external/CppTorch/include/cpptorch my_project/third_party/
```

```bash
g++ -std=c++20 -fopenmp -I my_project/third_party my_project/main.cpp -lopenblas -o my_app
```

### 5. Package managers

The repository ships a vcpkg manifest and a Conan recipe.

```bash
# vcpkg (manifest mode)
vcpkg install cpptorch

# Conan
conan create .
conan install --requires=cpptorch/0.1.0 -of build
```

---

## Configuration

### CMake options

| Option | Default | Description |
|---|---|---|
| `cpptorch_BUILD_TESTS` | `ON` when top-level | Build the test suite |
| `cpptorch_BUILD_EXAMPLES` | `ON` when top-level | Build the examples |
| `cpptorch_INSTALL` | `ON` when top-level | Generate the install rules |

```bash
cmake -S . -B build \
      -DCMAKE_BUILD_TYPE=Release \
      -Dcpptorch_BUILD_TESTS=OFF \
      -Dcpptorch_BUILD_EXAMPLES=OFF
```

### BLAS and OpenMP

The library needs a CBLAS provider and OpenMP. CMake locates both
automatically:

- `BLA_VENDOR` is set to `OpenBLAS` by default. Override it to use another
  provider, for example `-DBLA_VENDOR=MKL` or `-DBLA_VENDOR=AppleAccelerate`.
- `cblas.h` is searched for in the standard locations. If it lives somewhere
  unusual, point CMake at it: `-DCBLAS_INCLUDE_DIR=/path/to/include`.
- `OpenMP::OpenMP_CXX` is found through `find_package(OpenMP)`.

### Locating the bundled datasets

`Datasets` resolves the CSV files at runtime, in this order:

1. a directory installed with `cpptorch::Resources::set_dir("...")`,
2. the `CPPTORCH_RESOURCES_PATH` environment variable,
3. the `CPPTORCH_DEFAULT_RESOURCES_PATH` compile definition — set automatically
   for in-tree builds of the tests and the examples,
4. `./resources/` relative to the current working directory.

```cpp
#include <cpptorch/cpptorch.h>

int main()
{
    cpptorch::Resources::set_dir("/opt/cpptorch/share/resources/");
    // ...
}
```

```bash
CPPTORCH_RESOURCES_PATH=/opt/cpptorch/share/resources/ ./my_app
```

---

## Quick start

### Using CppTorch in a CMake project

```cmake
cmake_minimum_required(VERSION 3.21)
project(my_app LANGUAGES CXX)

find_package(cpptorch 0.1 REQUIRED)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE CppTorch::cpptorch)
```

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH=/your/prefix
cmake --build build
```

### Using CppTorch without CMake

```bash
g++ -std=c++20 -O3 -fopenmp -march=native \
    -I /your/prefix/include \
    main.cpp -lopenblas -o my_app
```

```bash
clang++ -std=c++20 -O3 -fopenmp \
    -I /your/prefix/include \
    main.cpp -lopenblas -o my_app
```

### The code

```cpp
#include <cpptorch/cpptorch.h>

#include <iostream>

int main()
{
    using namespace cpptorch;

    // 1. Data: a 3-class spiral with 100 samples per class.
    auto [X, y] = Datasets::spiral_data(100, 3);

    // 2. Model: 2 -> 64 (ReLU) -> 32 (ReLU) -> 3 (Softmax).
    Model model(ModelType::Classification);

    model.add_layer({.n_inputs = X.cols(), .n_neurons = 64, .activation_type = ActivationType::ReLU});
    model.add_layer({.n_inputs = 64, .n_neurons = 32, .activation_type = ActivationType::ReLU});
    model.add_layer({.n_inputs = 32, .n_neurons = 3, .activation_type = ActivationType::Softmax});

    // 3. Train with Adam and a decaying learning rate.
    model.set_optimizer({.optmzr_type = OptimizerType::Adam, .learning_rate = 0.01, .decay = 1e-3});
    model.train(X, y, /*n_epochs=*/100);

    // 4. Evaluate.
    auto result = model.evaluate(X, y);
    std::cout << "Accuracy: " << result.accuracy * 100.0 << "%\n";

    // 5. Predict.
    Matrix classes = model.predict_classes(X);
    std::cout << "First prediction: " << classes(0, 0) << "\n";

    return 0;
}
```

A complete, runnable version of this program lives in
[`examples/quickstart.cpp`](examples/quickstart.cpp).

### Regression example

```cpp
#include <cpptorch/cpptorch.h>

int main()
{
    using namespace cpptorch;

    auto [X, y] = Datasets::california_housing();

    StandardScaler scaler;
    scaler.fit_transform(X);

    Model model(ModelType::Regression);
    model.add_layer({.n_inputs = X.cols(), .n_neurons = 64, .activation_type = ActivationType::ReLU});
    model.add_layer({.n_inputs = 64, .n_neurons = 1, .activation_type = ActivationType::Linear});
    model.set_optimizer({.optmzr_type = OptimizerType::Adam, .learning_rate = 0.01, .decay = 1e-3});

    model.train(X, y, 20);

    auto result = model.evaluate(X, y);
    std::cout << "RMSE: " << result.rmse << "\n";
    return 0;
}
```

---

## Project layout

```
CppTorch/
├── CMakeLists.txt               Library target, options, install/export rules
├── cmake/
│   └── cpptorchConfig.cmake.in  Package config template
├── include/cpptorch/            Public headers — the library itself
│   ├── cpptorch.h               Umbrella header
│   ├── core/
│   │   ├── matrix.h             Matrix, MatrixView
│   │   └── numcpp.h             NumCpp
│   ├── layers/
│   │   ├── dense.h              Layer_Dense
│   │   ├── dropout.h            Layer_Dropout
│   │   ├── layer.h              Layer_Wrapper, HyperParameters_Layer_Wrapper
│   │   └── activations/
│   │       ├── activation.h         Activation, ActivationType
│   │       ├── activation_registry.h make_activation()
│   │       ├── linear.h             Activation_Linear
│   │       ├── relu.h               Activation_ReLU
│   │       └── softmax.h            Activation_Softmax
│   ├── losses/
│   │   ├── loss.h                Loss, LossType
│   │   ├── loss_registry.h       make_loss()
│   │   ├── categorical_crossentropy.h
│   │   ├── mean_squared_error.h
│   │   └── softmax_categoricalcrossentropy.h
│   ├── optimizers/
│   │   ├── optimizer.h           Optimizer, OptimizerType
│   │   ├── optimizer_registry.h  make_optimizer()
│   │   ├── sgd.h  adagrad.h  rmsprop.h  adam.h
│   ├── utils/
│   │   ├── exceptions.h          ShapeInvalidForOperation
│   │   ├── resources.h           Resources
│   │   ├── scaler.h              StandardScaler
│   │   └── utils.h               random helpers, readCSV
│   ├── datasets.h               Datasets
│   └── model.h                  Model
├── examples/
│   ├── CMakeLists.txt
│   ├── quickstart.cpp
│   └── gender_audio_classification.cpp
├── tests/                       Test suite (CTest)
│   ├── CMakeLists.txt
│   ├── main.cpp                 Test dispatcher
│   ├── tests.h
│   └── *.cpp
├── docs/
│   └── architecture.md
├── resources/                   Bundled CSV datasets
├── .github/                     Issue templates, PR template, CI workflow
├── CHANGELOG.md
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── LICENSE
├── README.md
├── conanfile.py
└── vcpkg.json
```

---

## API reference

Everything below lives in the `cpptorch` namespace.

### Enums and configuration structs

#### `ActivationType`

| Value | Meaning |
|---|---|
| `Linear` | Identity activation, `f(x) = x` |
| `ReLU` | Rectified linear unit, `f(x) = max(0, x)` |
| `Softmax` | Softmax normalisation across the feature axis |

#### `LossType`

| Value | Meaning |
|---|---|
| `MeanSquaredError` | Mean squared error, for regression |
| `CategoricalCrossEntropy` | Categorical cross-entropy, for classification |

#### `OptimizerType`

| Value | Meaning |
|---|---|
| `SGD` | Stochastic gradient descent, with optional momentum |
| `Adagrad` | Per-parameter squared-gradient accumulation |
| `RMSprop` | Exponentially averaged squared gradient |
| `Adam` | Momentum plus adaptive learning rates |

#### `ModelType`

| Value | Meaning |
|---|---|
| `Classification` | Cross-entropy loss, accuracy metrics |
| `Regression` | Mean squared error loss, RMSE metrics |

#### `HyperParameters_Layer_Wrapper`

| Field | Type | Default | Meaning |
|---|---|---|---|
| `n_inputs` | `size_t` | `0` | Number of input features |
| `n_neurons` | `size_t` | `0` | Number of neurons in the layer |
| `weight_regularizer_l1` | `double` | `0.0` | L1 penalty on the weights |
| `weight_regularizer_l2` | `double` | `0.0` | L2 penalty on the weights |
| `bias_regularizer_l1` | `double` | `0.0` | L1 penalty on the biases |
| `bias_regularizer_l2` | `double` | `0.0` | L2 penalty on the biases |
| `activation_type` | `ActivationType` | `Linear` | Activation applied after the dense projection |
| `dropout_rate` | `double` | `0.0` | Dropout probability; `0.0` disables dropout |

#### `HyperParameters_Optimizer`

| Field | Type | Default | Meaning |
|---|---|---|---|
| `optmzr_type` | `OptimizerType` | `SGD` | The optimization algorithm |
| `learning_rate` | `double` | `0.01` | Base learning rate |
| `decay` | `double` | `0.0` | Learning-rate decay per iteration |
| `momentum` | `double` | `0.0` | Momentum factor (SGD) |
| `epsilon` | `double` | `1e-7` | Numerical stability constant |
| `rho` | `double` | `0.9` | Decay of the squared-gradient cache (RMSprop) |
| `beta_1` | `double` | `0.9` | Decay rate of the first moment (Adam) |
| `beta_2` | `double` | `0.999` | Decay rate of the second moment (Adam) |

---

### Core: `Matrix`

A dense, row-major matrix of doubles. The data lives in a single flat,
contiguous buffer so rows can be handed to BLAS without a copy. Arithmetic
operators follow numpy's broadcasting rules: a matrix with a single row or a
single column is broadcast against its operand.

#### Constructors

| Signature | Description |
|---|---|
| `Matrix(size_t rows, size_t cols)` | Zero-filled matrix of the given shape |
| `Matrix()` | Empty 0×0 matrix |
| `Matrix(size_t rows, size_t cols, std::initializer_list<double> list)` | From a brace-init list; the size must match |
| `Matrix(size_t rows, size_t cols, const std::vector<double> &flat_data)` | From a flat buffer; the size must match |

Copy and move constructors and assignment operators are defaulted.

#### Accessors and mutators

| Method | Returns | Description |
|---|---|---|
| `shape()` | `std::tuple<size_t, size_t>` | The `(rows, cols)` pair |
| `rows()` | `size_t` | Number of rows |
| `cols()` | `size_t` | Number of columns |
| `size()` | `size_t` | Total number of elements |
| `data()` | `const double *` | Pointer to the first element |
| `getFlatData()` | `const std::vector<double> &` | The underlying flat buffer |
| `begin()`, `end()` | iterators | Iteration over the flat buffer (const and non-const) |
| `operator[](size_t i)` | `double &` | Element at the flat index `i` |
| `operator()(size_t r, size_t c)` | `double &` | Element at row `r`, column `c`; throws `std::out_of_range` |
| `rows_view(size_t first_row, size_t row_count)` | `MatrixView` | Non-owning view of a row range; throws `std::out_of_range` |
| `copy_from(const MatrixView &source)` | `void` | Replaces the contents with the view's |
| `resize(size_t rows, size_t cols)` | `void` | Resizes and discards the previous contents |
| `copy_rows_from(const Matrix &source, size_t first_row, size_t row_count)` | `void` | Copies a row range out of `source` |

#### Transformations

| Method | Returns | Description |
|---|---|---|
| `T()` | `Matrix` | The transpose |
| `sq()` | `Matrix` | The element-wise square |
| `display(int truncateX = -1, int truncateY = -1)` | `void` | Prints the matrix; `-1` prints everything |

#### Operators

| Family | Operators | Semantics |
|---|---|---|
| Matrix ⊕ matrix | `+` `-` `*` `/` `%` | Element-wise, with broadcasting. `%` is `fmod`. |
| Matrix ⊕ scalar | `+` `-` `*` `/` `%` | Element-wise against a scalar |
| Scalar ⊕ matrix | `+` `-` `*` `/` `%` | Element-wise against a scalar |
| Matrix ⋚ matrix | `==` `!=` `<` `>` `<=` `>=` | Element-wise, returning `1.0` or `0.0` |
| Matrix ⋚ scalar | `==` `!=` `<` `>` `<=` `>=` | Element-wise, returning `1.0` or `0.0` |
| Scalar ⋚ matrix | `==` `!=` `<` `>` `<=` `>=` | Element-wise, returning `1.0` or `0.0` |
| Matrix ⊕= matrix | `+=` `-=` `*=` `/=` `%=` | In-place element-wise |
| Matrix ⊕= scalar | `+=` `-=` `*=` `/=` `%=` | In-place element-wise |

---

### Core: `MatrixView`

A non-owning view over a contiguous range of rows of a `Matrix`. It lets the
training loop process a batch without copying the data.

| Member | Description |
|---|---|
| `MatrixView()` | Default constructor, an empty view |
| `MatrixView(const double *data, size_t rows, size_t cols)` | Wraps an existing buffer |
| `data()` | Pointer to the first element |
| `rows()` | Number of rows |
| `cols()` | Number of columns |
| `size()` | Total number of elements |

---

### Core: `NumCpp`

numpy-style operations on `Matrix`. All operations are static and return new
matrices, except the `_into` variants which write into a caller-provided buffer
to avoid allocations in hot loops.

#### Creation

| Method | Returns | Description |
|---|---|---|
| `zeroes(tuple<size_t, size_t> shape)` | `Matrix` | A zero-filled matrix |
| `ones(tuple<int, int> shape)` | `Matrix` | A matrix filled with ones |
| `diagflat(const Matrix &a)` | `Matrix` | A diagonal matrix built from a flat vector |
| `randn(tuple<size_t, size_t> shape)` | `Matrix` | Samples from the standard normal distribution |
| `binomial(int n, double p, tuple<size_t, size_t> shape)` | `Matrix` | Samples from the binomial distribution; validates `n ≥ 0` and `0 ≤ p ≤ 1` |
| `binomial_into(int n, double p, tuple<size_t, size_t> shape, Matrix &result)` | `void` | Same, written into `result` |
| `linspace(double start, double stop, size_t num)` | `Matrix` | `num` evenly spaced samples over the closed interval |

#### Matrix products

| Method | Returns | Description |
|---|---|---|
| `dot(const Matrix &A, const Matrix &B)` | `Matrix` | `A * B` |
| `dot_into(const Matrix &A, const Matrix &B, Matrix &result)` | `void` | `A * B` written into `result` |
| `dot_into(const MatrixView &A, const Matrix &B, Matrix &result)` | `void` | `A * B` with `A` read as a view |
| `dot_transpose_left(const Matrix &A, const Matrix &B)` | `Matrix` | `Aᵀ * B` without materialising the transpose |
| `dot_transpose_left_into(const Matrix &A, const Matrix &B, Matrix &result)` | `void` | Same, written into `result` |
| `dot_transpose_left_into(const MatrixView &A, const Matrix &B, Matrix &result)` | `void` | Same, with `A` read as a view |
| `dot_transpose_right(const Matrix &A, const Matrix &B)` | `Matrix` | `A * Bᵀ` without materialising the transpose |
| `dot_transpose_right_into(const Matrix &A, const Matrix &B, Matrix &result)` | `void` | Same, written into `result` |

All of them throw `ShapeInvalidForOperation` when the inner dimensions do not
match.

#### Reductions

| Method | Returns | Description |
|---|---|---|
| `sum(const Matrix &A)` | `double` | Sum of all elements |
| `sum(const Matrix &A, int axis)` | `Matrix` | Sum along axis 0 (rows) or 1 (columns) |
| `sum_into(const Matrix &A, int axis, Matrix &result)` | `void` | Sum along an axis into `result`; currently axis 0 only |
| `max(const Matrix &a)` | `double` | Largest element; throws `std::out_of_range` when empty |
| `max(const Matrix &a, int axis)` | `Matrix` | Maximum along axis 0 or 1 |
| `argmax(const Matrix &a)` | `std::tuple<size_t, size_t>` | Position of the largest element |
| `argmax(const Matrix &a, int axis)` | `Matrix` | Index of the maximum along axis 0 or 1 |
| `mean(const Matrix &a)` | `double` | Arithmetic mean; an empty matrix averages to zero |
| `unique_count(const Matrix &a)` | `size_t` | Number of distinct values |

#### Element-wise math

| Method | Returns | Description |
|---|---|---|
| `exp(const Matrix &a)` | `Matrix` | Element-wise exponential |
| `log(const Matrix &a)` | `Matrix` | Element-wise natural logarithm |
| `sqrt(const Matrix &a)` | `Matrix` | Element-wise square root |
| `abs(const Matrix &a)` | `Matrix` | Element-wise absolute value |
| `sin(const Matrix &a)` | `Matrix` | Element-wise sine |
| `cos(const Matrix &a)` | `Matrix` | Element-wise cosine |
| `maximum(double operand, const Matrix &a)` | `Matrix` | Element-wise maximum against a scalar |
| `clip(const Matrix &a, double min_val, double max_val)` | `Matrix` | Clamps to the closed interval |
| `where(const Matrix &condition, double ifTrue, double ifFalse)` | `Matrix` | Selects between two values element-wise |

#### Encoding

| Method | Returns | Description |
|---|---|---|
| `one_hot(const Matrix &y, size_t n_classes)` | `Matrix` | Converts class indices into a one-hot matrix |
| `outer(const Matrix &A, const Matrix &B)` | `Matrix` | The outer product `A * Bᵀ` of two flat matrices; throws on empty operands |

---

### Layers: `Layer_Dense`

A fully connected layer: `output = inputs * weights + biases`. The layer owns
its weights, biases and gradient buffers; `backward()` accumulates into
`dweights` and `dbiases`, which the optimizer then consumes.

#### Public members

| Member | Type | Description |
|---|---|---|
| `weights` | `Matrix` | Weights of shape `(n_inputs, n_neurons)` |
| `biases` | `Matrix` | Biases of shape `(1, n_neurons)` |
| `inputs` | `Matrix` | Cached inputs of the last forward pass |
| `output` | `Matrix` | Output of the last forward pass |
| `dweights` | `Matrix` | Gradient with respect to the weights |
| `dinputs` | `Matrix` | Gradient with respect to the inputs |
| `dbiases` | `Matrix` | Gradient with respect to the biases |
| `weight_momentums`, `bias_momentums` | `Matrix` | Optimizer state: first moment |
| `weight_cache`, `bias_cache` | `Matrix` | Optimizer state: accumulated squared gradients |
| `weight_regularizer_l1`, `weight_regularizer_l2` | `double` | Penalties applied to the weights |
| `bias_regularizer_l1`, `bias_regularizer_l2` | `double` | Penalties applied to the biases |

#### Methods

| Method | Description |
|---|---|
| `Layer_Dense(size_t n_inputs, size_t n_neurons, double weight_regularizer_l1 = 0, double weight_regularizer_l2 = 0, double bias_regularizer_l1 = 0, double bias_regularizer_l2 = 0)` | Constructs the layer and initialises the weights with `randn * WEIGHT_SCALAR` |
| `forward(const Matrix &inputs)` | Computes the forward pass and caches the inputs |
| `backward(const Matrix &dvalues, bool apply_regularization = true)` | Computes the parameter and input gradients; adds the L1/L2 penalties when requested |

---

### Layers: `Layer_Dropout`

Inverted dropout. During training each activation is zeroed with probability
`rate` and the survivors are scaled by `1 / (1 - rate)`, so no rescaling is
needed at inference time.

| Member | Type | Description |
|---|---|---|
| `rate` | `double` | Probability of dropping an activation, stored as `1 - rate` |
| `inputs` | `Matrix` | Cached inputs |
| `output` | `Matrix` | Output of the last forward pass |
| `binary_mask` | `Matrix` | The dropout mask |
| `dinputs` | `Matrix` | Gradient with respect to the inputs |

| Method | Description |
|---|---|
| `Layer_Dropout(double rate)` | Constructs the stage |
| `forward(const Matrix &inputs, bool cache_inputs = true)` | Applies the dropout mask |
| `backward(const Matrix &dvalues)` | Propagates the gradient through the mask |

---

### Layers: `Layer_Wrapper`

Bundles a dense layer, an optional dropout stage and an activation. A `Model`
owns a sequence of wrappers; the wrapper exposes a single forward/backward pair
to the training loop.

| Method | Returns | Description |
|---|---|---|
| `Layer_Wrapper(HyperParameters_Layer_Wrapper params)` | — | Constructs the layer from its hyper-parameters |
| `output(bool is_training = true)` | `const Matrix &` | The layer output; dropout is bypassed when `is_training` is false |
| `dinputs()` | `const Matrix &` | Gradient with respect to the layer input |
| `dense()` | `Layer_Dense &` | The owned dense layer (const and non-const overloads) |
| `activation_type()` | `ActivationType` | The activation applied by the layer |
| `has_dropout()` | `bool` | Whether the layer applies dropout |
| `forward(const Matrix &inputs, bool is_training = true)` | `void` | Runs the layer forward |
| `backward(const Matrix &dvalues, bool apply_regularization = true)` | `void` | Computes the gradient with respect to the layer input |

---

### Activations

#### `Activation`

The base class. Owns the `output`, `inputs` and `dinputs` buffers.

| Method | Description |
|---|---|
| `virtual forward(const Matrix &inputs, bool cache_inputs = true) = 0` | Runs the activation forward |
| `virtual backward(const Matrix &dvalues) = 0` | Computes the gradient with respect to the inputs |
| `Activation()` | Default constructor |
| `Activation(Matrix output, Matrix inputs, Matrix dinputs)` | Constructs with pre-allocated buffers |

#### `Activation_Linear`

Identity activation, `f(x) = x`.

#### `Activation_ReLU`

Rectified linear unit, `f(x) = max(0, x)`. The backward pass zeroes the
gradient of the negative entries.

#### `Activation_Softmax`

Softmax normalisation across the feature axis. The forward pass subtracts the
row maximum before exponentiating, which keeps it numerically stable. The
backward pass uses the identity
`dinputs = s * (d - sum(s * d))`, which avoids forming the Jacobian.

---

### Losses

#### `Loss`

The base class. Owns the `dinputs` gradient buffer.

| Method | Returns | Description |
|---|---|---|
| `virtual forward(const Matrix &y_pred, const Matrix &y_true) = 0` | `Matrix` | The per-sample losses |
| `virtual backward(const Matrix &dvalues, const Matrix &y_true) = 0` | `void` | Gradient of the loss with respect to the prediction |
| `calculate(const Matrix &output, const Matrix &y)` | `double` | Mean of the per-sample losses |
| `static accuracy(const Matrix &output, const Matrix &y)` | `double` | Fraction of correctly classified samples |
| `static regularization_loss(const Layer_Dense &layer)` | `double` | Total L1 + L2 penalty of a layer |

#### `Loss_MeanSquaredError`

Mean squared error, for regression. `forward()` returns the mean squared error
per sample; `backward()` scales the gradient by `-2 / (outputs * samples)`.

#### `Loss_CategoricalCrossEntropy`

Categorical cross-entropy, for classification. Accepts either one-hot targets
or a column of class indices. Probabilities are clipped to `[1e-7, 1 - 1e-7]`
before taking the logarithm.

#### `Activation_Softmax_Loss_CategoricalCrossentropy`

The fused softmax activation and cross-entropy loss. Combining the two folds
the gradient of the softmax into the loss gradient, which avoids materialising
an intermediate Jacobian and is measurably faster than running the two steps
separately.

| Member | Type | Description |
|---|---|---|
| `activation` | `Activation_Softmax` | The softmax activation |
| `loss` | `Loss_CategoricalCrossEntropy` | The loss |
| `output` | `Matrix` | Output of the softmax |
| `dinputs` | `Matrix` | Gradient with respect to the layer input |

| Method | Returns | Description |
|---|---|---|
| `forward(const Matrix &inputs, const Matrix &y_true)` | `double` | Runs the softmax forward and scores the result |
| `calculate(const Matrix &inputs, const Matrix &y_true)` | `double` | Alias of `forward()` that only returns the loss |
| `backward(const Matrix &y_true)` | `void` | Computes the fused gradient |

---

### Optimizers

#### `Optimizer`

The base class. Owns the learning-rate schedule and knows how to apply a
gradient step to a `Layer_Dense`.

| Method | Returns | Description |
|---|---|---|
| `explicit Optimizer(double learning_rate = 0.01, double decay = 0, double momentum = 0, double epsilon = 1e-7, double rho = 0.9, double beta_1 = 0.9, double beta_2 = 0.999)` | — | Constructs the optimizer |
| `get_current_learning_rate()` | `double` | The learning rate used by the next update |
| `virtual pre_update_params()` | `void` | Applies the learning-rate decay; called before each update |
| `post_update_params()` | `void` | Advances the iteration counter |
| `virtual update_params(Layer_Dense &layer) const = 0` | `void` | Applies a gradient step to a layer |
| `virtual update_params(Layer_Dense &layer, bool include_regularization) const` | `void` | Same, optionally folding the regularization penalty into the gradient |
| `static weight_gradient(const Layer_Dense &layer, size_t index, double gradient, bool include_regularization)` | `double` | The weight gradient with the penalty applied |
| `static bias_gradient(const Layer_Dense &layer, size_t index, double gradient, bool include_regularization)` | `double` | The bias gradient with the penalty applied |

#### `Optimizer_SGD`

Stochastic gradient descent with optional momentum.

`Optimizer_SGD(double learning_rate = 0.01, double decay = 0, double momentum = 0)`

#### `Optimizer_Adagrad`

Scales each parameter by the square root of its accumulated squared gradient.

`Optimizer_Adagrad(double learning_rate = 0.01, double decay = 0, double epsilon = 1e-7)`

#### `Optimizer_RMSprop`

Scales each parameter by the square root of an exponentially averaged squared
gradient.

`Optimizer_RMSprop(double learning_rate = 0.001, double decay = 0, double epsilon = 1e-7, double rho = 0.9)`

#### `Optimizer_Adam`

Combines momentum with per-parameter adaptive learning rates and applies bias
correction.

`Optimizer_Adam(double learning_rate = 0.001, double decay = 0, double epsilon = 1e-7, double beta_1 = 0.9, double beta_2 = 0.999)`

---

### Model

`Model` owns the layer sequence, the loss function and the optimizer, and
implements the forward, backward and parameter-update passes. It is the entry
point of the library for most users.

#### Constructor

| Signature | Description |
|---|---|
| `explicit Model(ModelType model_type = ModelType::Regression)` | Builds a model with the default loss for the type: categorical cross-entropy for classification, mean squared error for regression. The default optimizer is SGD. |

#### Configuration

| Method | Returns | Description |
|---|---|---|
| `set_model_type(ModelType)` | `void` | Switches the type and resets the loss accordingly |
| `model_type()` | `ModelType` | The current model type |
| `set_loss(std::unique_ptr<Loss>)` | `void` | Replaces the loss; throws `std::invalid_argument` on a null pointer |
| `loss()` | `Loss &` | The loss owned by the model |
| `add_layer(HyperParameters_Layer_Wrapper)` | `void` | Appends a layer; layers run in the order they are added |
| `set_optimizer(HyperParameters_Optimizer)` | `void` | Replaces the optimizer |

#### Inference

| Method | Returns | Description |
|---|---|---|
| `predict(const Matrix &X)` | `Matrix` | Forward pass without updating parameters; throws `std::runtime_error` when the model has no layers and `std::invalid_argument` when the feature count does not match |
| `predict_classes(const Matrix &X)` | `Matrix` | The class index of every sample for classification models, the raw output for regression models |
| `evaluate(const Matrix &X, const Matrix &y, bool verbose = true)` | `EvaluationResult` | Scores the model and optionally prints the metrics |

#### `Model::EvaluationResult`

| Member | Type | Description |
|---|---|---|
| `loss` | `double` | Loss value; the mean squared error for regression models |
| `accuracy` | `double` | Fraction of correctly classified samples; zero for regression models |
| `rmse` | `double` | Root mean squared error; zero for classification models |
| `display(ModelType type)` | `void` | Prints the metrics that suit the model type |

#### Training

```cpp
void train(const Matrix &X_train,
           const Matrix &y_train,
           size_t n_epochs,
           const Matrix &X_val = Matrix(),
           const Matrix &y_val = Matrix(),
           size_t print_every = 1,
           size_t batch_size = 0,
           size_t validation_every = 1);
```

| Parameter | Meaning |
|---|---|
| `X_train`, `y_train` | The training data. Both must have the same non-zero row count. |
| `n_epochs` | The number of epochs to run. |
| `X_val`, `y_val` | An optional validation set. Leave them empty to skip validation. |
| `print_every` | Print the progress every *n* epochs. `0` is rejected. |
| `batch_size` | The mini-batch size. `0` trains on the full dataset in one batch. |
| `validation_every` | Evaluate the validation set every *n* epochs. `0` is rejected. |

Throws `std::runtime_error` when the model has no layers, no loss or no
optimizer, and `std::invalid_argument` when the training data shapes do not
match or an interval is zero.

When the last layer uses a softmax activation and the loss is categorical
cross-entropy, the fused gradient kernel is used instead of the generic path.

#### Static helper

| Method | Description |
|---|---|
| `static displayEpoch(size_t n_epochs, size_t epoch, double train_loss, double val_loss, double train_acc, double val_acc, double lr, bool has_val, ModelType model_type)` | Prints the progress of a single epoch to stdout |

---

### Datasets

`Datasets` loads the bundled CSV files and splits them into train/test/validation
sets. Every loader accepts a `filename` argument and resolves it through
`cpptorch::Resources`.

| Method | Returns | Description |
|---|---|---|
| `spiral_data(size_t samples, size_t classes)` | `tuple<Matrix, Matrix>` | A synthetic spiral dataset with `samples` points per class. Returns inputs of shape `(samples * classes, 2)` and a column of class indices. |
| `sine_data(size_t samples = 1000)` | `tuple<Matrix, Matrix>` | A synthetic sine wave: inputs uniformly spaced over `[0, 1]`, targets `sin(2πx)`. |
| `california_housing(double train_ratio = 0.7, double test_ratio = 0.15, double val_ratio = 0.15, bool shuffle = true, unsigned int seed = 42, bool one_hot_categorical = true, const std::string &filename = "housing_data.csv")` | `tuple<Matrix, Matrix, Matrix, Matrix, Matrix, Matrix>` | The California housing dataset, split into train/test/validation. Missing `total_bedrooms` values are imputed with the median. `one_hot_categorical` one-hot encodes the `ocean_proximity` column instead of ordinal-encoding it. Returns `(X_train, y_train, X_test, y_test, X_val, y_val)`. |
| `california_housing(const std::string &filename, double train_ratio = ..., ...)` | same | Overload taking the filename first |
| `california_housing_data(...)` | same | Alias of `california_housing` |
| `housing_data(...)` | same | Alias of `california_housing` |
| `gender_audio_data(const std::string &filename = "gender_audio_features_kaggle.csv", double train_ratio = 0.7, double test_ratio = 0.15, double val_ratio = 0.15, bool shuffle = true, unsigned int seed = 42)` | `tuple<Matrix, Matrix, Matrix, Matrix, Matrix, Matrix>` | The Kaggle gender audio dataset: 43 acoustic features and a binary label. Returns `(X_train, y_train, X_test, y_test, X_val, y_val)`. |

All loaders throw `std::runtime_error` when the file cannot be opened or
contains no usable rows, and `std::invalid_argument` when the split ratios sum
to zero.

---

### Utilities

#### `StandardScaler`

Standardises features to zero mean and unit variance. Fit it on the training
set and then apply it unchanged to the validation and test sets.

| Method | Returns | Description |
|---|---|---|
| `fit(const Matrix &X)` | `void` | Computes the per-feature mean and standard deviation |
| `fit_transform(Matrix &X)` | `void` | Fits on `X` and standardises it in place |
| `transform(Matrix &X)` | `void` | Standardises `X` in place using the fitted statistics |

#### `Resources`

Resolves the directory that holds the bundled CSV datasets. See
[Locating the bundled datasets](#locating-the-bundled-datasets).

| Method | Returns | Description |
|---|---|---|
| `set_dir(const std::string &dir)` | `void` | Overrides the resources directory; an empty string clears the override |
| `dir()` | `const std::string &` | The resources directory, always terminated with `/` |
| `path(const std::string &filename)` | `std::string` | The full path of a file inside the resources directory |

#### `ShapeInvalidForOperation`

Thrown when an operation receives operands with incompatible shapes. It derives
from `std::exception` and provides a `what()` message naming the operation and
both shapes.

#### Free functions — `utils/utils.h`

| Function | Returns | Description |
|---|---|---|
| `randomNormalValue()` | `double` | A sample from the standard normal distribution. The generator is thread-local, so it is safe inside OpenMP regions. |
| `randomBinomialValue(int n, double p)` | `double` | A sample from the binomial distribution. Thread-local generator. |
| `readCSV(const std::string &filename)` | `std::vector<std::vector<double>>` | Reads a numeric CSV from the resources directory, skipping the header row. |

---

### Extension points

Three factories map an enum value to a concrete implementation. They are the
only places the core needs to know about a component.

| Function | Returns | Description |
|---|---|---|
| `make_activation(ActivationType type)` | `std::unique_ptr<Activation>` | Creates the activation registered for `type` |
| `make_loss(LossType type)` | `std::unique_ptr<Loss>` | Creates the loss registered for `type` |
| `make_optimizer(OptimizerType type, const HyperParameters_Optimizer &params)` | `std::unique_ptr<Optimizer>` | Creates the optimizer registered for `type` |

They live in `activation_registry.h`, `loss_registry.h` and
`optimizer_registry.h` respectively, and throw `std::invalid_argument` when the
type is unknown.

---

## Extending the library

CppTorch separates the **core** (the maths and the training loop) from the
**components** (activations, losses, optimizers). The core never needs to
change when a component is added — this is the property that keeps the library
stable.

| Component | Base class | Enum | Factory |
|---|---|---|---|
| Activation | `cpptorch::Activation` | `ActivationType` | `make_activation` |
| Loss | `cpptorch::Loss` | `LossType` | `make_loss` |
| Optimizer | `cpptorch::Optimizer` | `OptimizerType` | `make_optimizer` |
| Dataset | — | — | a static function on `Datasets` |

### Example: adding a Leaky ReLU activation

**1. Create the header** `include/cpptorch/layers/activations/leaky_relu.h`:

```cpp
#pragma once

#include "cpptorch/layers/activations/activation.h"

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

**2. Add the enum value** in `activation.h`:

```cpp
enum class ActivationType : std::uint8_t
{
    Linear = 0,
    ReLU = 1,
    Softmax = 2,
    LeakyReLU = 3,   // <- new
};
```

**3. Register it** in `activation_registry.h`:

```cpp
case ActivationType::LeakyReLU:
    return std::make_unique<Activation_LeakyReLU>();
```

**4. Export it** by adding the header to `include/cpptorch/cpptorch.h`.

**5. Test it** — add `tests/full_pass_with_leaky_relu.cpp`, register it in
`tests/CMakeLists.txt` and in the dispatcher in `tests/main.cpp`.

`model.h` was not touched. The same recipe applies to losses and optimizers;
the full walkthrough is in [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Building and testing

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
cd build && ctest --output-on-failure
```

The test suite is a single binary with a dispatcher, so you can also run a
single test by hand:

```bash
./build/tests/cpptorch_tests full_pass_with_optimizer_adam
```

The examples are plain executables:

```bash
./build/examples/cpptorch_example_quickstart
./build/examples/cpptorch_example_gender_audio_classification
```

The API documentation is generated with Doxygen:

```bash
doxygen Doxyfile   # writes docs/html
```

---

## Contributing

Contributions are welcome — see **[CONTRIBUTING.md](CONTRIBUTING.md)** for the
full guide. The short version:

### 1. Set up

```bash
git clone https://github.com/TaherMustansir1929/CppTorch.git
cd CppTorch
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
cd build && ctest --output-on-failure
```

### 2. Add a feature without breaking the core

The single most important rule: **a new component must not require changing
the training loop in `model.h`.**

To achieve that, always go through the extension points:

1. **Create one self-contained header** for the component. It derives from the
   family base class (`Activation`, `Loss` or `Optimizer`) and includes only
   what it needs. It must not include `model.h`.
2. **Add an enum value** to the family enum (`ActivationType`, `LossType`,
   `OptimizerType`). The enum is the only identifier the core needs, so adding
   a value is backwards compatible.
3. **Register the implementation** in the family factory
   (`activation_registry.h`, `loss_registry.h`, `optimizer_registry.h`).
4. **Export it** by adding the header to `include/cpptorch/cpptorch.h`.
5. **Test it.** Add a test file, register it in `tests/CMakeLists.txt`, and add
   its name to the dispatcher in `tests/main.cpp`.

What you must **not** touch:

- `include/cpptorch/model.h` — the training loop. If a feature genuinely
  requires changing it, open an issue first and explain why.
- `include/cpptorch/core/` — the maths. Changes here affect everything and
  need a strong numerical justification.
- The public API of existing classes. Adding is fine; changing or removing a
  signature is a breaking change and needs a major version bump.

### 3. Follow the conventions

- Everything public lives in `namespace cpptorch`.
- Headers use `#pragma once` and are self-contained.
- Public classes and methods carry Doxygen comments (`@brief`, `@param`,
  `@return`, `@throws`).
- Run `clang-format` with the repository's `.clang-format` before committing.
- Prefer `*_into` variants in hot paths to avoid allocations.
- New public API must be added to the API reference in this README and to
  `CHANGELOG.md`.

### 4. Open the pull request

Use the pull-request template and make sure the checklist is complete:

- [ ] The build is clean and `ctest` passes.
- [ ] New public API is documented and mentioned in the README.
- [ ] `model.h` and `core/` are unchanged, unless that is the point of the PR.
- [ ] `clang-format` has been run.
- [ ] `CHANGELOG.md` has an entry under *Unreleased*.

CI runs the build and the test suite on Linux, macOS and Windows, and verifies
that the installed package can be consumed through `find_package()`.

Please also read the [Code of Conduct](CODE_OF_CONDUCT.md).

---

## License

CppTorch is released under the [MIT License](LICENSE).
