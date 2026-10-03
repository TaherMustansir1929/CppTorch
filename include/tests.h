#pragma once

#include "activation.h"
#include "datasets.h"
#include "layer_dense.h"
#include "layer_dropout.h"
#include "loss.h"
#include "matrix.h"
#include "numcpp.h"
#include "optimizer.h"

#include <cstdio>
#include <iostream>

Matrix relu(const Matrix &x);
Matrix derivative_relu(const Matrix &x);

class Tests
{
public:
    static void forwardPass_with_Loss_CCE()
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

    static void backwardPass_on_one_Layer()
    {
        Matrix inputs(4, 1, {1, 2, 3, 4});

        Matrix weights(3, 4, {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2});

        Matrix biases(3, 1, {0.1, 0.2, 0.3});

        double learning_rate = 0.001;

        for (int iter = 0; iter < 200; iter++)
        {
            Matrix z = NumCpp::dot(weights, inputs) + biases;
            Matrix a = relu(z);
            double y = NumCpp::sum(a);

            double loss = std::pow(y, 2);

            // Backward pass
            // Gradient of loss with respect to output y
            double dL_dy = 2 * y;

            // Gradient of y with respect to a (creates a matrix of 1s matching 'a')
            Matrix dy_da = NumCpp::ones(a.shape());

            // Gradient of loss with respect to a
            Matrix dL_da = dL_dy * dy_da;

            // Gradient of a with respect to z (ReLU derivative)
            Matrix da_dz = derivative_relu(z);

            // Gradient of loss with respect to z (element-wise multiplication)
            Matrix dL_dz = dL_da * da_dz;

            // Gradient of loss with respect to weights and biases
            // Fixed outer product argument order to yield a 3x4 matrix
            // Matrix dL_dW = NumCpp::outer(dL_dz, inputs);
            Matrix dL_dW = NumCpp::dot(dL_dz, inputs.T());
            Matrix dL_db = dL_dz;

            // Update weights and biases
            weights -= learning_rate * dL_dW;
            biases -= learning_rate * dL_db;

            // Print the loss every 20 iterations
            if (iter % 20 == 0)
            {
                std::cout << "Iteration: " << iter << ",\tLoss: " << loss << "\n";
            }

            // Final loss
            if (iter == 199)
            {
                std::cout << "Final Loss: " << loss << "\n";
            }
        }

        // Final weights and biases
        std::cout << "Final weights:\n";
        weights.display();
        std::cout << "Final biases:\n";
        biases.display();
    }

    static void full_forward_backward_pass()
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

    static void full_pass_with_optimizer_vgd()
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
                       optimizer.current_learning_rate);
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

    static void full_pass_with_optimizer_adagrad()
    {
        auto [X, y] = Datasets::spiral_data(100, 3);
        Matrix y_one_hot = NumCpp::one_hot(y, 3);

        Layer_Dense dense1(2, 64);
        Activation_ReLU activation1;
        Layer_Dense dense2(64, 3);
        Activation_Softmax_Loss_CategoricalCrossentropy loss_activation;
        Optimizer_Adagrad optimizer(1, 1e-4);

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
                printf("epoch: %d, accuracy: %.3f, loss: %.3f, lr: %f\n",
                       i,
                       accuracy,
                       loss,
                       optimizer.current_learning_rate);
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

    static void full_pass_with_optimizer_rmsprop()
    {
        auto [X, y] = Datasets::spiral_data(100, 3);
        Matrix y_one_hot = NumCpp::one_hot(y, 3);

        Layer_Dense dense1(2, 64);
        Activation_ReLU activation1;
        Layer_Dense dense2(64, 3);
        Activation_Softmax_Loss_CategoricalCrossentropy loss_activation;
        Optimizer_RMSprop optimizer(0.02, 1e-5, 1e-7, 0.999);

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
                printf("epoch: %d, accuracy: %.3f, loss: %.3f, lr: %f\n",
                       i,
                       accuracy,
                       loss,
                       optimizer.current_learning_rate);
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

    static void full_pass_with_optimizer_adam()
    {
        auto [X, y] = Datasets::spiral_data(100, 3);
        Matrix y_one_hot = NumCpp::one_hot(y, 3);

        Layer_Dense dense1(2, 64);
        Activation_ReLU activation1;
        Layer_Dense dense2(64, 3);
        Activation_Softmax_Loss_CategoricalCrossentropy loss_activation;
        Optimizer_Adam optimizer(0.02, 1e-5);

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
                printf("epoch: %d, accuracy: %.3f, loss: %.3f, lr: %f\n",
                       i,
                       accuracy,
                       loss,
                       optimizer.current_learning_rate);
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

    static void full_pass_with_regularization()
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
                       optimizer.current_learning_rate);
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

    static void full_pass_with_dropout_layer()
    {
        auto [X, y] = Datasets::spiral_data(1000, 3);
        Matrix y_one_hot = NumCpp::one_hot(y, 3);

        Layer_Dense dense1(2, 64, 0, 5e-4, 0, 5e-4);
        Activation_ReLU activation1;
        Layer_Dropout dropout1(0.1);
        Layer_Dense dense2(64, 3);
        Activation_Softmax_Loss_CategoricalCrossentropy loss_activation;
        Optimizer_Adam optimizer(0.05, 5e-5);

        // Training loop
        for (int epoch = 0; epoch <= 10000; epoch++)
        {
            // forward pass
            dense1.forward(X);
            activation1.forward(dense1.output);
            dropout1.forward(activation1.output);
            dense2.forward(dropout1.output);
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
                       optimizer.current_learning_rate);
            }

            loss_activation.backward(y_one_hot);
            dense2.backward(loss_activation.dinputs);
            dropout1.backward(dense2.dinputs);
            activation1.backward(dropout1.dinputs);
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

    static void full_pass_with_mean_squared_error()
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
                printf("epoch: %4d, loss (MSE): %.6f, lr: %f\n", epoch, loss, optimizer.current_learning_rate);
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

    static void california_housing_dataset_test()
    {
        std::cout << "Loading California Housing dataset...\n";
        auto [X_train, y_train, X_test, y_test, X_val, y_val] = Datasets::california_housing(0.7, 0.15, 0.15);

        std::cout << "Successfully loaded and split California Housing dataset:\n";
        std::cout << "  X_train: " << X_train.rows() << " x " << X_train.cols() << "\n";
        std::cout << "  y_train: " << y_train.rows() << " x " << y_train.cols() << "\n";
        std::cout << "  X_test:  " << X_test.rows() << " x " << X_test.cols() << "\n";
        std::cout << "  y_test:  " << y_test.rows() << " x " << y_test.cols() << "\n";
        std::cout << "  X_val:   " << X_val.rows() << " x " << X_val.cols() << "\n";
        std::cout << "  y_val:   " << y_val.rows() << " x " << y_val.cols() << "\n";

        std::cout << "\nSample X_train (first 2 rows):\n";
        X_train.display(2);

        std::cout << "\nSample y_train (first 2 rows):\n";
        y_train.display(2);
    }

    static void train_california_housing()
    {
        std::cout << "Loading California Housing dataset...\n";
        auto [X_train, y_train, X_test, y_test, X_val, y_val] = Datasets::california_housing(0.7, 0.15, 0.15);

        // Standardize features (Z-score normalization based on training distribution)
        Matrix mean = NumCpp::sum(X_train, 0) / static_cast<double>(X_train.rows());
        Matrix diff = X_train - mean;
        Matrix variance = NumCpp::sum(diff * diff, 0) / static_cast<double>(X_train.rows());
        Matrix std_dev = NumCpp::sqrt(variance) + 1e-7;

        X_train = (X_train - mean) / std_dev;
        X_val = (X_val - mean) / std_dev;
        X_test = (X_test - mean) / std_dev;

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
                   optimizer.current_learning_rate);
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
};


Matrix relu(const Matrix &x)
{
    return NumCpp::maximum(0.0, x);
}

Matrix derivative_relu(const Matrix &x)
{
    return NumCpp::where(x > 0.0, 1.0, 0.0);
}