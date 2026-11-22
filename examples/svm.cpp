#include <cpptorch/cpptorch.h>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

struct ClassificationMetrics
{
    double accuracy = 0.0;
    double precision = 0.0;
    double recall = 0.0;
    double f1 = 0.0;
    double mse = 0.0;
    double rmse = 0.0;
    size_t tp = 0;
    size_t tn = 0;
    size_t fp = 0;
    size_t fn = 0;
};

[[nodiscard]] double parse_positive_double(const char *text, const char *name)
{
    const std::string value(text);
    const double result = std::stod(value);
    if (result <= 0.0)
        throw std::runtime_error(std::string(name) + " must be greater than zero.");
    return result;
}

[[nodiscard]] size_t parse_positive_size(const char *text, const char *name)
{
    const std::string value(text);
    size_t result = 0;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || result == 0)
        throw std::runtime_error(std::string(name) + " must be a positive integer.");
    return result;
}

// Splits shuffled indices into training and test sets.
[[nodiscard]] std::pair<std::vector<size_t>, std::vector<size_t>> split_indices(
    size_t total_samples, double train_ratio, unsigned int seed)
{
    std::vector<size_t> indices(total_samples);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 gen(seed);
    std::shuffle(indices.begin(), indices.end(), gen);

    const size_t train_size = static_cast<size_t>(std::round(total_samples * train_ratio));
    std::vector<size_t> train_idx(indices.begin(), indices.begin() + train_size);
    std::vector<size_t> test_idx(indices.begin() + train_size, indices.end());
    return {train_idx, test_idx};
}

// Builds train/test matrices from the full dataset using the given index split.
[[nodiscard]] std::tuple<cpptorch::Matrix, cpptorch::Matrix, cpptorch::Matrix, cpptorch::Matrix>
build_split(const cpptorch::Matrix &X,
            const cpptorch::Matrix &y,
            const std::vector<size_t> &train_idx,
            const std::vector<size_t> &test_idx)
{
    const size_t n_features = X.cols();

    cpptorch::Matrix X_train(train_idx.size(), n_features);
    cpptorch::Matrix y_train(train_idx.size(), 1);
    for (size_t i = 0; i < train_idx.size(); ++i)
    {
        const size_t src = train_idx[i];
        for (size_t j = 0; j < n_features; ++j)
            X_train(i, j) = X(src, j);
        y_train(i, 0) = y(src, 0);
    }

    cpptorch::Matrix X_test(test_idx.size(), n_features);
    cpptorch::Matrix y_test(test_idx.size(), 1);
    for (size_t i = 0; i < test_idx.size(); ++i)
    {
        const size_t src = test_idx[i];
        for (size_t j = 0; j < n_features; ++j)
            X_test(i, j) = X(src, j);
        y_test(i, 0) = y(src, 0);
    }

    return {X_train, y_train, X_test, y_test};
}

// Computes classification and regression metrics on the test set.
[[nodiscard]] ClassificationMetrics compute_test_metrics(const cpptorch::Matrix &outputs,
                                                         const cpptorch::Matrix &targets)
{
    ClassificationMetrics metrics;

    const size_t n = outputs.rows();
    double squared_error_sum = 0.0;

    for (size_t i = 0; i < n; ++i)
    {
        const double prediction = outputs(i, 0);
        const double target = targets(i, 0);
        const double difference = prediction - target;
        squared_error_sum += difference * difference;

        const bool pred_positive = prediction > 0.0;
        const bool target_positive = target > 0.0;

        if (pred_positive && target_positive)
            ++metrics.tp;
        else if (!pred_positive && !target_positive)
            ++metrics.tn;
        else if (pred_positive && !target_positive)
            ++metrics.fp;
        else
            ++metrics.fn;
    }

    metrics.accuracy = static_cast<double>(metrics.tp + metrics.tn) / static_cast<double>(n);
    metrics.mse = squared_error_sum / static_cast<double>(n);
    metrics.rmse = std::sqrt(metrics.mse);

    const double precision_denom = static_cast<double>(metrics.tp + metrics.fp);
    metrics.precision = precision_denom > 0.0
                            ? static_cast<double>(metrics.tp) / precision_denom
                            : 0.0;

    const double recall_denom = static_cast<double>(metrics.tp + metrics.fn);
    metrics.recall = recall_denom > 0.0
                         ? static_cast<double>(metrics.tp) / recall_denom
                         : 0.0;

    const double f1_denom = metrics.precision + metrics.recall;
    metrics.f1 = f1_denom > 0.0 ? 2.0 * metrics.precision * metrics.recall / f1_denom : 0.0;

    return metrics;
}

void print_confusion_matrix(const ClassificationMetrics &m)
{
    std::cout << "\nConfusion matrix (rows = true, columns = predicted):\n";
    std::cout << "              pred=-1   pred=+1\n";
    std::cout << "true=-1  " << std::setw(9) << m.tn << "  " << std::setw(9) << m.fp << "\n";
    std::cout << "true=+1  " << std::setw(9) << m.fn << "  " << std::setw(9) << m.tp << "\n";
}

void print_classification_report(const ClassificationMetrics &m)
{
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "\nTest-set classification metrics:\n";
    std::cout << "  Accuracy : " << m.accuracy << " (" << m.accuracy * 100.0 << "%)\n";
    std::cout << "  Precision: " << m.precision << "\n";
    std::cout << "  Recall   : " << m.recall << "\n";
    std::cout << "  F1-score : " << m.f1 << "\n";
    std::cout << "  MSE      : " << m.mse << "\n";
    std::cout << "  RMSE     : " << m.rmse << "\n";
    std::cout << "  TP=" << m.tp << "  TN=" << m.tn << "  FP=" << m.fp << "  FN=" << m.fn << "\n";
}

} // namespace

int main(int argc, char **argv)
{
    try
    {
#ifndef CPPTORCH_DEFAULT_RESOURCES_PATH
#define CPPTORCH_DEFAULT_RESOURCES_PATH "resources"
#endif
        if (argc > 4)
            throw std::runtime_error("Usage: svm [dataset.csv] [epochs] [learning_rate]");

        const std::string dataset_filename = argc > 1 ? argv[1] : "svm_dataset.csv";
        const size_t epochs = argc > 2 ? parse_positive_size(argv[2], "epochs") : 200;
        const double learning_rate = argc > 3 ? parse_positive_double(argv[3], "learning_rate") : 0.01;

        // 1. Data: load the SVM benchmark dataset (x1, x2, label +/- 1).
        auto [X, y] = cpptorch::Datasets::svm_data(dataset_filename);
        const size_t n_samples = X.rows();
        std::cout << "Dataset: " << dataset_filename << "\n"
                  << "Samples: " << n_samples << ", features: " << X.cols() << "\n";

        // 2. Train/test split (80/20).
        const auto [train_idx, test_idx] = split_indices(n_samples, 0.8, 42);
        auto [X_train, y_train, X_test, y_test] = build_split(X, y, train_idx, test_idx);
        std::cout << "Train samples: " << X_train.rows() << ", test samples: " << X_test.rows() << "\n";

        // 3. Standardise features (fit on train, apply to test).
        cpptorch::StandardScaler feature_scaler;
        feature_scaler.fit_transform(X_train);
        feature_scaler.transform(X_test);

        // 4. Model: a neural network approximating the SVM decision function.
        //    2 -> 32 (ReLU) -> 16 (ReLU) -> 1 (Linear), trained with MSE on +/- 1 targets.
        cpptorch::Model model(cpptorch::ModelType::Regression);
        model.add_layer({.n_inputs = 2,
                         .n_neurons = 32,
                         .activation_type = cpptorch::ActivationType::ReLU});
        model.add_layer({.n_inputs = 32,
                         .n_neurons = 16,
                         .activation_type = cpptorch::ActivationType::ReLU});
        model.add_layer({.n_inputs = 16,
                         .n_neurons = 1,
                         .activation_type = cpptorch::ActivationType::Linear});
        model.set_optimizer({.optmzr_type = cpptorch::OptimizerType::Adam,
                             .learning_rate = learning_rate,
                             .decay = 1e-4});

        // 5. Train: training loss is logged every epoch by Model::train.
        std::cout << "\nTraining neural SVM...\n";
        model.train(X_train, y_train, epochs, cpptorch::Matrix(), cpptorch::Matrix(), 10, 0, 10);

        // 6. Evaluate on the held-out test set.
        const cpptorch::Matrix test_outputs = model.predict(X_test);
        const ClassificationMetrics metrics = compute_test_metrics(test_outputs, y_test);

        cpptorch::Model::EvaluationResult eval_result = model.evaluate(X_test, y_test, true);
        (void)eval_result;

        print_classification_report(metrics);
        print_confusion_matrix(metrics);

        // 7. Sanity check: training-set accuracy.
        const cpptorch::Matrix train_outputs = model.predict(X_train);
        const ClassificationMetrics train_metrics = compute_test_metrics(train_outputs, y_train);
        std::cout << "\nTraining-set classification metrics:\n";
        std::cout << "  Accuracy : " << train_metrics.accuracy << " (" << train_metrics.accuracy * 100.0 << "%)\n";
        std::cout << "  MSE      : " << train_metrics.mse << "\n";
        std::cout << "  RMSE     : " << train_metrics.rmse << "\n";

        std::cout << "\nNeural SVM training complete.\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "SVM example failed: " << error.what() << '\n';
        return 1;
    }
}
