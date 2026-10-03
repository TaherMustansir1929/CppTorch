#pragma once

#include "activation.h"
#include "datasets.h"
#include "layer_dense.h"
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
            Matrix dy_da = NumCpp::ones_like(a.shape());

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
};


Matrix relu(const Matrix &x)
{
    return NumCpp::maximum(0.0, x);
}

Matrix derivative_relu(const Matrix &x)
{
    return NumCpp::where(x > 0.0, 1.0, 0.0);
}