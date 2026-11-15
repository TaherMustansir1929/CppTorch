# Changelog

All notable changes to CppTorch are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Nothing yet.

## [0.1.0] - 2026-11-15

The first release of CppTorch as a standalone, distributable library.

### Added

- Header-only `cpptorch` library exposed through the CMake target
  `CppTorch::cpptorch`.
- `cpptorch::Matrix` and `cpptorch::MatrixView` with numpy-style broadcasting,
  stored in a single flat buffer and handed to BLAS without copies.
- `cpptorch::NumCpp` with the numpy-style array operations: `dot`, `sum`,
  `max`, `argmax`, `mean`, `clip`, `one_hot`, `where`, `outer`, `linspace`,
  `randn`, `binomial`, `exp`, `log`, `sqrt`, `abs`, `sin`, `cos`.
- `cpptorch::Activation` with `Activation_Linear`, `Activation_ReLU` and
  `Activation_Softmax`, plus the fused
  `Activation_Softmax_Loss_CategoricalCrossentropy`.
- `cpptorch::Loss` with `Loss_MeanSquaredError` and
  `Loss_CategoricalCrossEntropy`.
- `cpptorch::Optimizer` with `Optimizer_SGD`, `Optimizer_Adagrad`,
  `Optimizer_RMSprop` and `Optimizer_Adam`.
- `cpptorch::Layer_Dense`, `cpptorch::Layer_Dropout` and
  `cpptorch::Layer_Wrapper`.
- `cpptorch::Model`, the high-level training API with `train()`, `predict()`,
  `predict_classes()` and `evaluate()`.
- `cpptorch::Datasets` with `spiral_data`, `sine_data`, `california_housing`
  and `gender_audio_data`.
- `cpptorch::StandardScaler` for feature standardisation.
- `cpptorch::Resources` for runtime resolution of the dataset directory.
- Extension points: `make_activation()`, `make_loss()` and `make_optimizer()`
  let new components be plugged in without touching the training loop.
- CMake package export (`cpptorchConfig.cmake`), a vcpkg manifest and a Conan
  recipe.
- A CTest suite with 15 tests and two runnable examples.

### Removed

- The emotion-recognition dataset, its feature extractor script, its notebook
  and the emotion classification entry point.

[Unreleased]: https://github.com/TaherMustansir1929/CppTorch/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/TaherMustansir1929/CppTorch/releases/tag/v0.1.0
