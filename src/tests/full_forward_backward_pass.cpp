#include "tests.h"

void Tests::full_forward_backward_pass()
{
    auto [X, y] = Datasets::spiral_data(100, 3);
    Matrix y_one_hot = NumCpp::one_hot(y, 3);

    Layer_Dense dense1(2, 3);
    Activation_ReLU activation1;
    Layer_Dense dense2(3, 3);
    Activation_Softmax_Loss_CategoricalCrossentropy loss_activation;

    // Forward pass
    dense1.forward(X);
    activation1.forward(dense1.output);
    dense2.forward(activation1.output);
    double loss = loss_activation.forward(dense2.output, y_one_hot);

    // outputs of forward pass
    std::cout << "Outputs from forward pass: \n";
    loss_activation.output.display(5);
    std::cout << "Loss = " << loss << '\n';

    // calculate accuracy
    double acc = Loss::accuracy(loss_activation.output, y_one_hot);
    std::cout << "Accuracy = " << acc << '\n';

    // Backward pass
    loss_activation.backward(y_one_hot);
    dense2.backward(loss_activation.dinputs);
    activation1.backward(dense2.dinputs);
    dense1.backward(activation1.dinputs);

    // Print gradients
    std::cout << "\nDense1:\n";
    dense1.dweights.display();
    dense1.dbiases.display();
    std::cout << "\nDense2:\n";
    dense2.dweights.display();
    dense2.dbiases.display();
}
