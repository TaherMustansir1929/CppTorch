/**
 * @file quickstart.cpp
 * @brief Minimal end-to-end usage of the CppTorch library.
 *
 * The spiral dataset is generated in memory, so this example needs no data files:
 * @code
 *   ./examples/quickstart
 * @endcode
 */

#include <cpptorch/cpptorch.h>

#include <iostream>

int main()
{
    using namespace cpptorch;

    // 1. Generate a 3-class spiral dataset (100 samples per class).
    auto [X, y] = Datasets::spiral_data(100, 3);

    // 2. Build a classifier: 2 -> 64 (ReLU) -> 32 (ReLU) -> 3 (Softmax).
    Model model(ModelType::Classification);
    model.add_layer({.n_inputs = X.cols(), .n_neurons = 64, .activation_type = ActivationType::ReLU});
    model.add_layer({.n_inputs = 64, .n_neurons = 32, .activation_type = ActivationType::ReLU});
    model.add_layer({.n_inputs = 32, .n_neurons = 3, .activation_type = ActivationType::Softmax});

    // 3. Train with the Adam optimizer.
    model.set_optimizer({.optmzr_type = OptimizerType::Adam, .learning_rate = 0.01, .decay = 1e-3});
    model.train(X, y, /*n_epochs=*/50, /*X_val=*/Matrix(), /*y_val=*/Matrix(), /*print_every=*/10);

    // 4. Inspect the predictions.
    Matrix predictions = model.predict_classes(X);
    std::cout << "Predicted class of the first 5 samples:";
    for (size_t i = 0; i < 5; ++i)
        std::cout << " " << static_cast<int>(predictions(i, 0));
    std::cout << "\nActual class of the first 5 samples: ";
    for (size_t i = 0; i < 5; ++i)
        std::cout << " " << static_cast<int>(y(i, 0));
    std::cout << "\n";

    return 0;
}
