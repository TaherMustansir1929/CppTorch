#include "tests.h"

void Tests::california_housing_dataset_test()
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
