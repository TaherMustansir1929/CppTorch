#include "tests.h"

void Tests::full_pass_with_optimizer_vgd()
{
    auto [X, y] = Datasets::spiral_data(100, 3);
    Matrix y_one_hot = NumCpp::one_hot(y, 3);

    Layer_Dense dense1(2, 64);
    Activation_ReLU activation1;
    Layer_Dense dense2(64, 3);
    Activation_Softmax_Loss_CategoricalCrossentropy loss_activation;
    Optimizer_SGD optimizer(1, 1e-3, 0.9);

    for (int i = 1; i <= 10000; i++)
    {
        // Forward pass
        dense1.forward(X);
        activation1.forward(dense1.output);
        dense2.forward(activation1.output);
        double loss = loss_activation.forward(dense2.output, y_one_hot);
        double accuracy = Loss::accuracy(loss_activation.output, y_one_hot);

        if ((i % 100) == 0)
        {
            printf("epoch: %d, accuracy: %.3f, loss: %.3f, lr: %.3f\n",
                   i,
                   accuracy,
                   loss,
                   optimizer.get_current_learning_rate());
        }

        // Backward pass
        loss_activation.backward(y_one_hot);
        dense2.backward(loss_activation.dinputs);
        activation1.backward(dense2.dinputs);
        dense1.backward(activation1.dinputs);

        // Update weights and biases
        optimizer.pre_update_params();
        optimizer.update_params(dense1);
        optimizer.update_params(dense2);
        optimizer.post_update_params();
    }
}
