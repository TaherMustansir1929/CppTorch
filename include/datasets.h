#pragma once

#include "matrix.h"
#include "utils.h"
#include <tuple>
#include <vector>

std::tuple<Matrix, std::vector<int>> spiral_dataset()
{
    std::vector<std::vector<double>> fileData = readCSV("spiral_data.csv");

    // Pre-allocate a 1D vector to hold the entire dataset continuously
    std::vector<double> flat_data;
    flat_data.reserve(fileData.size() * 2);

    std::vector<int> y;
    y.reserve(fileData.size());

    for (const auto &vec : fileData)
    {
        flat_data.push_back(vec[0]);
        flat_data.push_back(vec[1]);
        y.push_back(static_cast<int>(vec[2]));
    }

    // Initialize Matrix with the flat contiguous dataset
    Matrix X(fileData.size(), 2, flat_data);

    return {X, y};
}