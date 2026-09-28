#pragma once

#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

template <typename T>
int minColSize(vector<vector<T>> arr)
{
    int minSize = arr[0].size();
    for (vector<T> &vec : arr)
    {
        if (minSize > vec.size())
        {
            minSize = vec.size();
        }
    }

    return minSize;
}


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


static vector<vector<double>> readCSV(const string &filename)
{
    vector<vector<double>> data;
    ifstream file(RESOURCES_PATH + filename);

    string line;

    // Skip header
    getline(file, line);

    while (getline(file, line))
    {
        stringstream ss(line);
        string value;
        vector<double> row;

        while (getline(ss, value, ','))
        {
            row.push_back(stod(value));
        }
        data.push_back(row);
    }

    return data;
}