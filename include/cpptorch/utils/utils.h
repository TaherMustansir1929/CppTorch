#pragma once

#include "cpptorch/utils/resources.h"

#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace cpptorch
{

/**
 * @brief Draws a sample from the standard normal distribution.
 *
 * The generator is thread-local, so the function is safe to call from
 * OpenMP parallel regions.
 *
 * @return A normally distributed value with mean 0 and standard deviation 1.
 */
inline double randomNormalValue()
{
    static thread_local std::random_device rd;
    static thread_local std::mt19937 gen(rd());
    static thread_local std::normal_distribution<double> dist(0.0, 1.0);

    return dist(gen);
}

/**
 * @brief Draws a sample from the binomial distribution.
 *
 * The generator is thread-local, so the function is safe to call from
 * OpenMP parallel regions.
 *
 * @param n Number of trials.
 * @param p Probability of success, between 0 and 1.
 * @return The number of successes.
 */
inline double randomBinomialValue(int n, double p)
{
    static thread_local std::random_device rd;
    static thread_local std::mt19937 gen(rd());
    static thread_local std::binomial_distribution<int> dist(n, p);
    dist.param(std::binomial_distribution<int>::param_type(n, p));

    return static_cast<double>(dist(gen));
}

/**
 * @brief Reads a numeric CSV file into a vector of rows.
 *
 * The file is looked up in the resources directory, see
 * @ref cpptorch::Resources. The header line is skipped and every remaining
 * row is parsed as comma-separated doubles.
 *
 * @param filename Name of the file inside the resources directory.
 * @return One vector of doubles per data row.
 */
inline std::vector<std::vector<double>> readCSV(const std::string &filename)
{
    std::vector<std::vector<double>> data;
    std::ifstream file(Resources::path(filename));

    std::string line;

    // Skip header
    getline(file, line);

    while (getline(file, line))
    {
        std::stringstream ss(line);
        std::string value;
        std::vector<double> row;

        while (getline(ss, value, ','))
        {
            row.push_back(stod(value));
        }
        data.push_back(row);
    }

    return data;
}

} // namespace cpptorch
