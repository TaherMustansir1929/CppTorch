#include "tests.h"

void Tests::full_pass_with_regularization()
{
    auto [X, y] = Datasets::spiral_data(1000, 3);
    Matrix y_one_hot = NumCpp::one_hot(y, 3);

    Layer_Dense dense1(2, 64, 0, 5e-4, 0, 5e-4);
    Activation_ReLU activation1;
    Layer_Dense dense2(64, 3);
    Activation_Softmax_Loss_CategoricalCrossentropy loss_activation;
    Optimizer_Adam optimizer(0.02, 5e-7);

    // Training loop
    for (int epoch = 0; epoch <= 10000; epoch++)
    {
        // forward pass
        dense1.forward(X);
        activation1.forward(dense1.output);
        dense2.forward(activation1.output);
        double data_loss = loss_activation.forward(dense2.output, y_one_hot);
        double regularization_loss = Loss::regularization_loss(dense1) + Loss::regularization_loss(dense2);

        double loss = data_loss + regularization_loss;
        double accuracy = Loss::accuracy(loss_activation.output, y_one_hot);

        if ((epoch % 100) == 0)
        {
            printf("epoch: %d, accuracy: %.3f, loss: %.3f, (data_loss: %.3f, reg_loss: %.3f), lr: %f\n",
                   epoch,
                   accuracy,
                   loss,
                   data_loss,
                   regularization_loss,
                   optimizer.get_current_learning_rate());
        }

        loss_activation.backward(y_one_hot);
        dense2.backward(loss_activation.dinputs);
        activation1.backward(dense2.dinputs);
        dense1.backward(activation1.dinputs);

        optimizer.pre_update_params();
        optimizer.update_params(dense1);
        optimizer.update_params(dense2);
        optimizer.post_update_params();
    }

    // Validate the model
    auto [X_test, y_test] = Datasets::spiral_data(100, 3);
    Matrix y_test_one_hot = NumCpp::one_hot(y_test, 3);

    // forward pass
    dense1.forward(X_test);
    activation1.forward(dense1.output);
    dense2.forward(activation1.output);
    double loss = loss_activation.forward(dense2.output, y_test_one_hot);
    double accuracy = Loss::accuracy(loss_activation.output, y_test_one_hot);
    printf("Validation, acc: %.3f, loss: %.3f\n", accuracy, loss);
}
