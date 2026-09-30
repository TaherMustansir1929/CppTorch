#pragma once

#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>


double randomValue(double min = -1, double max = 1)
{
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(min, max);

    return dist(gen);
}

double randomNormalValue()
{
    static std::random_device rd;
    static std::mt19937 gen(rd());
    // Mean of 0.0, Standard Deviation of 1.0
    std::normal_distribution<double> dist(0.0, 1.0);

    return dist(gen);
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