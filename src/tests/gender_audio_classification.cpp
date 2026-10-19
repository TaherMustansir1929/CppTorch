#include "tests.h"
#include <iomanip>

void Tests::gender_audio_classification()
{
    std::cout << "============================================================\n";
    std::cout << "     Kaggle Gender Audio Classification Training \n";
    std::cout << "============================================================\n\n";

    // 1. Load dataset (70% train, 15% test, 15% validation)
    std::cout << "[1/5] Loading Kaggle Gender Classification dataset...\n";
    auto [X_train, y_train, X_test, y_test, X_val, y_val] = Datasets::gender_audio_data();

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

    // 3. Build Neural Network Architecture: 43 -> 64 (ReLU) -> 32 (ReLU) -> 2 (Softmax)
    std::cout << "[3/5] Initializing Classification Model...\n";
    Model model(ModelType::Classification);

    model.add_layer({.n_inputs = X_train.cols(), .n_neurons = 64, .activation_type = ActivationType::ReLU});

    model.add_layer({.n_inputs = 64, .n_neurons = 32, .activation_type = ActivationType::ReLU});

    // Classification requires 2 neurons for binary categorical outputs and a Softmax Activation
    model.add_layer({.n_inputs = 32, .n_neurons = 2, .activation_type = ActivationType::Softmax});

    // Configure Adam optimizer with initial learning rate and decay
    model.set_optimizer({.optmzr_type = OptimizerType::Adam, .learning_rate = 0.01, .decay = 1e-3});
    std::cout << "      Model architecture created with Adam optimizer (lr=0.01, decay=0.001).\n\n";

    // 4. Train the Model
    const size_t epochs = 100;
    std::cout << "[4/5] Training model for " << epochs << " epochs...\n";
    model.train(X_train, y_train, epochs, X_val, y_val, /*print_every=*/1);
    std::cout << "\n";

    // 5. Evaluate on Unseen Test Dataset
    std::cout << "[5/5] Evaluating model on unseen test set (" << X_test.rows() << " samples)...\n";
    auto test_res = model.evaluate(X_test, y_test, /*verbose=*/true);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "      Test Cross-Entropy Loss: " << test_res.loss << "\n";
    std::cout << "      Test Accuracy:           " << (test_res.accuracy * 100.0) << "%\n\n";

    // Preview sample predictions against actual values
    std::cout << "Sample Predictions:\n";
    std::cout << "----------------------------------------------------------------------\n";
    std::cout << std::left << std::setw(10) << "Sample" << std::setw(20) << "Predicted Gender" << std::setw(20)
              << "Actual Gender" << std::setw(15) << "Result" << "\n";
    std::cout << "----------------------------------------------------------------------\n";

    // Fetch the 1D predicted class indices for the test set
    Matrix test_class_preds = model.predict_classes(X_test);

    for (size_t i = 0; i < 10; ++i)
    {
        double pred_class = test_class_preds(i, 0);
        double actual_class = y_test(i, 0);

        std::string pred_str = (pred_class == 1.0) ? "Female" : "Male";
        std::string actual_str = (actual_class == 1.0) ? "Female" : "Male";
        std::string result = (pred_class == actual_class) ? "CORRECT" : "INCORRECT";

        std::cout << std::left << std::setw(10) << (i + 1) << std::setw(20) << pred_str << std::setw(20) << actual_str
                  << std::setw(15) << result << "\n";
    }
    std::cout << "----------------------------------------------------------------------\n";
}