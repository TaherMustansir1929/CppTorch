#include "tests.h"

void Tests::full_pass_with_mean_squared_error()
{
    // 1. Create a synthetic regression dataset (sine wave)
    auto [X, y] = Datasets::sine_data(100);

    // 2. Build the model: 1 -> 64 -> 64 -> 1
    Layer_Dense dense1(1, 64);
    Activation_ReLU activation1;
    Layer_Dense dense2(64, 64);
    Activation_ReLU activation2;
    Layer_Dense dense3(64, 1);
    Activation_Linear activation3;

    // 3. Loss and Optimizer
    Loss_MeanSquaredError loss_function;
    Optimizer_Adam optimizer(0.01, 1e-3);

    std::cout << "Starting training with Mean Squared Error loss...\n";

    // 4. Epoch loop
    for (int epoch = 1; epoch <= 1000; epoch++)
    {
        // --- Forward pass ---
        dense1.forward(X);
        activation1.forward(dense1.output);
        dense2.forward(activation1.output);
        activation2.forward(dense2.output);
        dense3.forward(activation2.output);
        activation3.forward(dense3.output);

        double loss = loss_function.calculate(activation3.output, y);

        if ((epoch % 100) == 0 || epoch == 1)
        {
            printf("epoch: %4d, loss (MSE): %.6f, lr: %f\n", epoch, loss, optimizer.get_current_learning_rate());
        }

        // --- Backward pass ---
        loss_function.backward(activation3.output, y);
        activation3.backward(loss_function.dinputs);
        dense3.backward(activation3.dinputs);
        activation2.backward(dense3.dinputs);
        dense2.backward(activation2.dinputs);
        activation1.backward(dense2.dinputs);
        dense1.backward(activation1.dinputs);

        // --- Optimization / Parameter update ---
        optimizer.pre_update_params();
        optimizer.update_params(dense1);
        optimizer.update_params(dense2);
        optimizer.update_params(dense3);
        optimizer.post_update_params();
    }

    // 5. Test inference / validation on unseen samples
    auto [X_test, y_test] = Datasets::sine_data(20);
    dense1.forward(X_test);
    activation1.forward(dense1.output);
    dense2.forward(activation1.output);
    activation2.forward(dense2.output);
    dense3.forward(activation2.output);
    activation3.forward(dense3.output);

    double test_loss = loss_function.calculate(activation3.output, y_test);
    printf("\nTest set MSE loss: %.6f\n", test_loss);
}
