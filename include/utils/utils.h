#pragma once

#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>


inline double randomNormalValue()
{
    static thread_local std::random_device rd;
    static thread_local std::mt19937 gen(rd());
    static thread_local std::normal_distribution<double> dist(0.0, 1.0);

    return dist(gen);
}

inline double randomBinomialValue(int n, double p)
{
    static thread_local std::random_device rd;
    static thread_local std::mt19937 gen(rd());
    static thread_local std::binomial_distribution<int> dist(n, p);
    dist.param(std::binomial_distribution<int>::param_type(n, p));

    return static_cast<double>(dist(gen));
}


static std::vector<std::vector<double>> readCSV(const std::string &filename)
{
    std::vector<std::vector<double>> data;
    std::ifstream file(RESOURCES_PATH + filename);

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
