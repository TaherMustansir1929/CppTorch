#include "tests.h"

void Tests::test_softmax_backward()
{
    std::cout << "--- Testing Standalone Activation_Softmax Backward vs Combined Softmax+CCE ---\n";
    auto [X, y] = Datasets::spiral_data(5, 3);
    Matrix y_one_hot = NumCpp::one_hot(y, 3);

    Layer_Dense dense1(2, 3);
    dense1.forward(X);

    Activation_Softmax softmax;
    softmax.forward(dense1.output);

    Loss_CategoricalCrossEntropy loss_cce;
    double loss = loss_cce.calculate(softmax.output, y_one_hot);
    std::cout << "Loss: " << loss << "\n";

    // Standalone backward pass:
    // 1. Loss backward
    loss_cce.backward(softmax.output, y_one_hot);
    // 2. Softmax backward
    softmax.backward(loss_cce.dinputs);

    // Combined backward pass:
    Activation_Softmax_Loss_CategoricalCrossentropy combined;
    [[maybe_unused]] double comb_loss = combined.forward(dense1.output, y_one_hot);
    combined.backward(y_one_hot);

    std::cout << "Standalone Softmax dinputs:\n";
    softmax.dinputs.display();
    std::cout << "Combined Softmax+CCE dinputs:\n";
    combined.dinputs.display();

    double diff = NumCpp::sum(NumCpp::abs(softmax.dinputs - combined.dinputs));
    std::cout << "Absolute difference between standalone and combined: " << diff << "\n";
    if (diff < 1e-7)
    {
        std::cout << "SUCCESS: Standalone Softmax backward matches Combined Softmax+CCE perfectly!\n";
    }
    else
    {
        std::cerr << "FAILURE: Gradients do not match!\n";
    }
}
