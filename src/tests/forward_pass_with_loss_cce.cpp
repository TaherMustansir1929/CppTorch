#include "tests.h"

void Tests::forwardPass_with_Loss_CCE()
{
    auto [X, y] = Datasets::spiral_data(100, 3);

    // X.display(5);

    Layer_Dense dense1(2, 3);
    dense1.forward(X);

    Activation_ReLU relu;
    relu.forward(dense1.output);

    Layer_Dense dense2(dense1.output.cols(), 3);
    dense2.forward(relu.output);

    Activation_Softmax softmax;
    softmax.forward(dense2.output);

    softmax.output.display(5);

    // NumCpp::sum(softmax.output, 1).display(5);

    Loss_CategoricalCrossEntropy loss_function;
    Matrix y_one_hot_encoded = NumCpp::one_hot(y, 3);
    double loss = loss_function.calculate(softmax.output, y_one_hot_encoded);

    std::cout << "Loss: " << loss << "\n";

    std::cout << "Accuracy: " << Loss_CategoricalCrossEntropy::accuracy(softmax.output, y_one_hot_encoded) << "\n";
}
