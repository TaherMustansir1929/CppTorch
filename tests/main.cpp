#include "tests.h"

#include <cstring>
#include <iostream>
#include <map>
#include <string>

namespace
{

const std::map<std::string, void (*)()> &registry()
{
    static const std::map<std::string, void (*)()> tests = {
        {"backward_pass_on_one_layer", &cpptorch::tests::backward_pass_on_one_layer},
        {"forward_pass_with_loss_cce", &cpptorch::tests::forward_pass_with_loss_cce},
        {"full_forward_backward_pass", &cpptorch::tests::full_forward_backward_pass},
        {"full_pass_with_optimizer_sgd", &cpptorch::tests::full_pass_with_optimizer_sgd},
        {"full_pass_with_optimizer_adagrad", &cpptorch::tests::full_pass_with_optimizer_adagrad},
        {"full_pass_with_optimizer_rmsprop", &cpptorch::tests::full_pass_with_optimizer_rmsprop},
        {"full_pass_with_optimizer_adam", &cpptorch::tests::full_pass_with_optimizer_adam},
        {"full_pass_with_regularization", &cpptorch::tests::full_pass_with_regularization},
        {"full_pass_with_dropout_layer", &cpptorch::tests::full_pass_with_dropout_layer},
        {"full_pass_with_mean_squared_error", &cpptorch::tests::full_pass_with_mean_squared_error},
        {"california_housing_dataset_test", &cpptorch::tests::california_housing_dataset_test},
        {"train_california_housing", &cpptorch::tests::train_california_housing},
        {"california_housing_with_cpptorch", &cpptorch::tests::california_housing_with_cpptorch},
        {"test_softmax_backward", &cpptorch::tests::test_softmax_backward},
        {"gender_audio_classification", &cpptorch::tests::gender_audio_classification},
    };
    return tests;
}

} // namespace

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <test-name>\n\nAvailable tests:\n";
        for (const auto &[name, _] : registry())
            std::cerr << "  " << name << "\n";
        return 1;
    }

    const auto &tests = registry();
    const auto it = tests.find(argv[1]);
    if (it == tests.end())
    {
        std::cerr << "Unknown test: " << argv[1] << "\n";
        return 1;
    }

    it->second();
    return 0;
}
