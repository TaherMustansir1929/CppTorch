// Smoke test for the installed CppTorch package.
//
// It exercises the public API through the exported CppTorch::cpptorch target
// only, which proves that the installed headers, the BLAS dependency and the
// OpenMP dependency are all wired up correctly.

#include <cpptorch/cpptorch.h>

#include <cmath>
#include <iostream>

int main()
{
    using namespace cpptorch;

    // 1. Array primitives.
    Matrix a(2, 2, {1.0, 2.0, 3.0, 4.0});
    Matrix b = a + a * 2.0;
    if (std::abs(b(0, 0) - 3.0) > 1e-12 || std::abs(b(1, 1) - 12.0) > 1e-12)
    {
        std::cerr << "FAIL: matrix arithmetic" << std::endl;
        return 1;
    }

    // 2. Dataset generation.
    auto [X, y] = Datasets::spiral_data(10, 3);
    if (X.rows() != 30 || X.cols() != 2)
    {
        std::cerr << "FAIL: unexpected dataset shape" << std::endl;
        return 1;
    }

    // 3. A full training cycle.
    Model model(ModelType::Classification);
    model.add_layer({.n_inputs = X.cols(), .n_neurons = 8, .activation_type = ActivationType::ReLU});
    model.add_layer({.n_inputs = 8, .n_neurons = 3, .activation_type = ActivationType::Softmax});
    model.set_optimizer({.optmzr_type = OptimizerType::Adam, .learning_rate = 0.01});
    model.train(X, y, 3, Matrix(), Matrix(), 3);

    auto result = model.evaluate(X, y, false);
    if (result.accuracy < 0.0 || result.accuracy > 1.0)
    {
        std::cerr << "FAIL: accuracy out of range" << std::endl;
        return 1;
    }

    std::cout << "CppTorch consumer OK - accuracy: " << result.accuracy * 100.0 << "%" << std::endl;
    return 0;
}
