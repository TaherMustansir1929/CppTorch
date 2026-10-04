#include "tests.h"

void Tests::train_california_housing()
{
    std::cout << "Loading California Housing dataset...\n";
    auto [X_train, y_train, X_test, y_test, X_val, y_val] = Datasets::california_housing(0.7, 0.15, 0.15);

    // Standardize features (Z-score normalization based on training distribution)
    StandardScaler scaler;
    scaler.fit_transform(X_train);
    scaler.transform(X_val);
    scaler.transform(X_test);

    // Scale targets to units of $100k (e.g. $150,000 -> 1.5) for numerical stability
    const double target_scale = 100000.0;
    y_train /= target_scale;
    y_val /= target_scale;
    y_test /= target_scale;

    // Multi-layer perceptron architecture: 13 -> 64 (ReLU) -> 32 (ReLU) -> 1 (Linear)
    Layer_Dense dense1(X_train.cols(), 64);
    Activation_ReLU activation1;
    Layer_Dense dense2(64, 32);
    Activation_ReLU activation2;
    Layer_Dense dense3(32, 1);
    Activation_Linear activation3;

    Loss_MeanSquaredError loss_function;
    Optimizer_Adam optimizer(0.01, 1e-3);

    std::cout << "\nStarting training on California Housing dataset (20 epochs)...\n";
    std::cout << "Training samples: " << X_train.rows() << ", Features: " << X_train.cols() << "\n\n";

    for (int epoch = 1; epoch <= 20; epoch++)
    {
        // --- Forward Pass (Training) ---
        dense1.forward(X_train);
        activation1.forward(dense1.output);
        dense2.forward(activation1.output);
        activation2.forward(dense2.output);
        dense3.forward(activation2.output);
        activation3.forward(dense3.output);

        double train_loss = loss_function.calculate(activation3.output, y_train);

        // --- Backward Pass ---
        loss_function.backward(activation3.output, y_train);
        activation3.backward(loss_function.dinputs);
        dense3.backward(activation3.dinputs);
        activation2.backward(dense3.dinputs);
        dense2.backward(activation2.dinputs);
        activation1.backward(dense2.dinputs);
        dense1.backward(activation1.dinputs);

        // --- Parameter Update ---
        optimizer.pre_update_params();
        optimizer.update_params(dense1);
        optimizer.update_params(dense2);
        optimizer.update_params(dense3);
        optimizer.post_update_params();

        // --- Validation Evaluation ---
        dense1.forward(X_val);
        activation1.forward(dense1.output);
        dense2.forward(activation1.output);
        activation2.forward(dense2.output);
        dense3.forward(activation2.output);
        activation3.forward(dense3.output);

        double val_loss = loss_function.calculate(activation3.output, y_val);

        printf("Epoch %2d/20 - Train Loss (MSE): %.4f - Val Loss (MSE): %.4f - lr: %.6f\n",
               epoch,
               train_loss,
               val_loss,
               optimizer.get_current_learning_rate());
    }

    // --- Final Test Set Evaluation ---
    dense1.forward(X_test);
    activation1.forward(dense1.output);
    dense2.forward(activation1.output);
    activation2.forward(dense2.output);
    dense3.forward(activation2.output);
    activation3.forward(dense3.output);

    double test_loss = loss_function.calculate(activation3.output, y_test);
    printf("\nFinal Test Set MSE Loss: %.4f (in $100k units^2)\n", test_loss);
    printf("Final Test Root Mean Squared Error (RMSE): $%.2f\n", std::sqrt(test_loss) * target_scale);
}
