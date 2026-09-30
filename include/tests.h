#pragma once

#include "activation.h"
#include "datasets.h"
#include "layer_dense.h"
#include "loss.h"
#include "matrix.h"
#include "numcpp.h"

#include <iostream>

Matrix relu(const Matrix &x);
Matrix derivative_relu(const Matrix &x);

class Tests
{
public:
    static void forwardPass_with_Loss_CCE()
    {
        auto [X, y] = spiral_dataset();

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
        auto [X, y] = spiral_dataset();
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
};


Matrix relu(const Matrix &x)
{
    return NumCpp::maximum(0.0, x);
}

Matrix derivative_relu(const Matrix &x)
{
    return NumCpp::where(x > 0.0, 1.0, 0.0);
}