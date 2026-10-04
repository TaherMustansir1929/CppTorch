#include "datasets.h"
#include "model.h"
#include "utils/scaler.h"

#include <cmath>
#include <iomanip>
#include <iostream>

int main()
{
    std::cout << "============================================================\n";
    std::cout << "   California Housing Prices Prediction with CppTorch Model \n";
    std::cout << "============================================================\n\n";

    // 1. Load dataset from resources/housing_data.csv (70% train, 15% test, 15% validation)
    std::cout << "[1/5] Loading California Housing dataset...\n";
    auto [X_train, y_train, X_test, y_test, X_val, y_val] = Datasets::california_housing(0.7, 0.15, 0.15);

    std::cout << "      Dataset loaded successfully:\n";
    std::cout << "      - Training samples:   " << X_train.rows() << " (Features: " << X_train.cols() << ")\n";
    std::cout << "      - Validation samples: " << X_val.rows() << "\n";
    std::cout << "      - Test samples:       " << X_test.rows() << "\n\n";

    // 2. Standardize features using StandardScaler (fit on train, transform all)
    std::cout << "[2/5] Standardizing features using StandardScaler...\n";
    StandardScaler scaler;
    scaler.fit_transform(X_train);
    scaler.transform(X_val);
    scaler.transform(X_test);

    // Scale target house prices to units of $100k (e.g., $150,000 -> 1.5) for numerical stability
    const double target_scale = 100000.0;
    y_train /= target_scale;
    y_val /= target_scale;
    y_test /= target_scale;
    std::cout << "      Features standardized (zero mean, unit variance).\n";
    std::cout << "      Target scaled by $" << static_cast<long long>(target_scale) << " for training stability.\n\n";

    // 3. Build Neural Network Architecture: 13 -> 64 (ReLU) -> 32 (ReLU) -> 1 (Linear)
    std::cout << "[3/5] Initializing Regression Model...\n";
    Model model(ModelType::Regression);

    model.add_layer({
        .n_inputs = X_train.cols(),
        .n_neurons = 64,
        .activation_type = ActivationType::ReLU
    });

    model.add_layer({
        .n_inputs = 64,
        .n_neurons = 32,
        .activation_type = ActivationType::ReLU
    });

    model.add_layer({
        .n_inputs = 32,
        .n_neurons = 1,
        .activation_type = ActivationType::Linear
    });

    // Configure Adam optimizer with initial learning rate and decay
    model.set_optimizer({
        .optmzr_type = OptimizerType::Adam,
        .learning_rate = 0.01,
        .decay = 1e-3
    });
    std::cout << "      Model architecture created with Adam optimizer (lr=0.01, decay=0.001).\n\n";

    // 4. Train the Model
    const size_t epochs = 20;
    std::cout << "[4/5] Training model for " << epochs << " epochs...\n";
    model.train(X_train, y_train, epochs, X_val, y_val, /*print_every=*/1);
    std::cout << "\n";

    // 5. Evaluate on Unseen Test Dataset
    std::cout << "[5/5] Evaluating model on unseen test set (" << X_test.rows() << " samples)...\n";
    auto test_res = model.evaluate(X_test, y_test, /*verbose=*/true);

    double test_rmse_dollars = test_res.rmse * target_scale;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "      Test MSE Loss: " << test_res.loss << " (in $100k^2)\n";
    std::cout << "      Test RMSE:     $" << test_rmse_dollars << "\n\n";

    // Preview sample predictions against actual values
    std::cout << "Sample Predictions (First 5 Test Houses):\n";
    std::cout << "----------------------------------------------------------------------\n";
    std::cout << std::left << std::setw(10) << "Sample"
              << std::setw(20) << "Predicted Price"
              << std::setw(20) << "Actual Price"
              << std::setw(15) << "Absolute Error" << "\n";
    std::cout << "----------------------------------------------------------------------\n";

    Matrix test_preds = model.predict(X_test);
    for (size_t i = 0; i < 5; ++i)
    {
        double pred_price = test_preds(i, 0) * target_scale;
        double actual_price = y_test(i, 0) * target_scale;
        double abs_error = std::abs(pred_price - actual_price);

        std::cout << std::left << std::setw(10) << (i + 1)
                  << "$" << std::setw(19) << pred_price
                  << "$" << std::setw(19) << actual_price
                  << "$" << std::setw(14) << abs_error << "\n";
    }
    std::cout << "----------------------------------------------------------------------\n";

    return 0;
}
