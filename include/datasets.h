#pragma once

#include "core/matrix.h"
#include "core/numcpp.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

class Datasets
{
public:
    static std::tuple<Matrix, Matrix> spiral_data(size_t samples, size_t classes)
    {
        // X = np.zeros((samples*classes, 2))
        Matrix X(samples * classes, 2);

        // y = np.zeros(samples*classes, dtype='uint8')
        // We represent y as a column vector of doubles to match your Matrix structure
        Matrix y(samples * classes, 1);

        for (size_t class_number = 0; class_number < classes; ++class_number)
        {
            // Starting index for the current class slice
            size_t start_ix = samples * class_number;

            // r = np.linspace(0.0, 1, samples)
            Matrix r = NumCpp::linspace(0.0, 1.0, samples);

            // t = np.linspace(class_number*4, (class_number+1)*4, samples) + np.random.randn(samples)*0.2
            Matrix t = NumCpp::linspace(static_cast<double>(class_number) * 4.0,
                                        static_cast<double>((class_number + 1)) * 4.0,
                                        samples) +
                       (NumCpp::randn({samples, 1}) * 0.2);

            // Calculate coordinates using element-wise multiplication
            Matrix sin_t = NumCpp::sin(t * 2.5);
            Matrix cos_t = NumCpp::cos(t * 2.5);

            Matrix r_sin = r * sin_t;
            Matrix r_cos = r * cos_t;

            // X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
            // y[ix] = class_number
            for (size_t i = 0; i < samples; ++i)
            {
                size_t ix = start_ix + i;

                X(ix, 0) = r_sin[i];
                X(ix, 1) = r_cos[i];
                y(ix, 0) = static_cast<double>(class_number);
            }
        }

        return {X, y};
    }

    static std::tuple<Matrix, Matrix> sine_data(size_t samples = 1000)
    {
        Matrix X = NumCpp::linspace(0.0, 1.0, samples);
        Matrix y = NumCpp::sin(2.0 * 3.14159265358979323846 * X);
        return {X, y};
    }

    /**
     * @brief Generates California Housing prices dataset from housing_data.csv.
     *
     * Features (X):
     *   - 0: longitude
     *   - 1: latitude
     *   - 2: housing_median_age
     *   - 3: total_rooms
     *   - 4: total_bedrooms (missing values imputed with median)
     *   - 5: population
     *   - 6: households
     *   - 7: median_income
     *   - 8-12: ocean_proximity one-hot encoded (<1H OCEAN, INLAND, ISLAND, NEAR BAY, NEAR OCEAN)
     *           (or single column 8 if one_hot_categorical = false)
     *
     * Target (y):
     *   - median_house_value (column vector)
     *
     * @return std::tuple<Matrix, Matrix, Matrix, Matrix, Matrix, Matrix>
     *         Tuple of {X_train, y_train, X_test, y_test, X_val, y_val}
     */
    static std::tuple<Matrix, Matrix, Matrix, Matrix, Matrix, Matrix> california_housing(
        double train_ratio = 0.7,
        double test_ratio = 0.15,
        double val_ratio = 0.15,
        bool shuffle = true,
        unsigned int seed = 42,
        bool one_hot_categorical = true,
        const std::string &filename = "housing_data.csv")
    {
        std::ifstream file;
#ifdef RESOURCES_PATH
        file.open(std::string(RESOURCES_PATH) + filename);
#endif
        if (!file.is_open())
        {
            file.open("./resources/" + filename);
        }
        if (!file.is_open())
        {
            file.open(filename);
        }
        if (!file.is_open())
        {
            throw std::runtime_error("[Datasets::california_housing] Failed to open CSV file: " + filename);
        }

        std::string line;
        // Skip header
        if (!std::getline(file, line))
        {
            throw std::runtime_error("[Datasets::california_housing] Empty CSV file: " + filename);
        }

        struct RawRow
        {
            double longitude = 0.0;
            double latitude = 0.0;
            double housing_median_age = 0.0;
            double total_rooms = 0.0;
            double total_bedrooms = 0.0;
            bool bedrooms_missing = false;
            double population = 0.0;
            double households = 0.0;
            double median_income = 0.0;
            double median_house_value = 0.0;
            int ocean_cat = 0;
        };

        std::vector<RawRow> rows;
        std::vector<double> valid_bedrooms;

        auto trim = [](std::string &s) {
            size_t first = s.find_first_not_of(" \t\r\n");
            if (first == std::string::npos)
            {
                s.clear();
                return;
            }
            size_t last = s.find_last_not_of(" \t\r\n");
            s = s.substr(first, (last - first + 1));
        };

        while (std::getline(file, line))
        {
            if (!line.empty() && line.back() == '\r')
            {
                line.pop_back();
            }
            if (line.empty())
            {
                continue;
            }

            std::stringstream ss(line);
            std::string token;
            std::vector<std::string> tokens;
            while (std::getline(ss, token, ','))
            {
                trim(token);
                tokens.push_back(token);
            }

            if (tokens.size() < 10)
            {
                continue;
            }

            RawRow r;
            r.longitude = std::stod(tokens[0]);
            r.latitude = std::stod(tokens[1]);
            r.housing_median_age = std::stod(tokens[2]);
            r.total_rooms = std::stod(tokens[3]);

            if (tokens[4].empty())
            {
                r.bedrooms_missing = true;
            }
            else
            {
                r.total_bedrooms = std::stod(tokens[4]);
                valid_bedrooms.push_back(r.total_bedrooms);
            }

            r.population = std::stod(tokens[5]);
            r.households = std::stod(tokens[6]);
            r.median_income = std::stod(tokens[7]);
            r.median_house_value = std::stod(tokens[8]);

            // Categorical: ocean_proximity
            const std::string &ocean = tokens[9];
            if (ocean == "<1H OCEAN")
            {
                r.ocean_cat = 0;
            }
            else if (ocean == "INLAND")
            {
                r.ocean_cat = 1;
            }
            else if (ocean == "ISLAND")
            {
                r.ocean_cat = 2;
            }
            else if (ocean == "NEAR BAY")
            {
                r.ocean_cat = 3;
            }
            else if (ocean == "NEAR OCEAN")
            {
                r.ocean_cat = 4;
            }
            else
            {
                r.ocean_cat = 0;
            }

            rows.push_back(r);
        }

        if (rows.empty())
        {
            throw std::runtime_error("[Datasets::california_housing] No valid rows parsed from " + filename);
        }

        // Impute missing total_bedrooms with dataset median
        double median_bedrooms = 0.0;
        if (!valid_bedrooms.empty())
        {
            std::sort(valid_bedrooms.begin(), valid_bedrooms.end());
            size_t mid = valid_bedrooms.size() / 2;
            if (valid_bedrooms.size() % 2 == 0)
            {
                median_bedrooms = (valid_bedrooms[mid - 1] + valid_bedrooms[mid]) / 2.0;
            }
            else
            {
                median_bedrooms = valid_bedrooms[mid];
            }
        }

        for (auto &r : rows)
        {
            if (r.bedrooms_missing)
            {
                r.total_bedrooms = median_bedrooms;
            }
        }

        size_t total_samples = rows.size();
        size_t num_features = one_hot_categorical ? 13 : 9;

        // Shuffle rows
        std::vector<size_t> indices(total_samples);
        std::iota(indices.begin(), indices.end(), 0);
        if (shuffle)
        {
            std::mt19937 gen(seed);
            std::shuffle(indices.begin(), indices.end(), gen);
        }

        // Calculate split partition sizes
        double total_ratio = train_ratio + test_ratio + val_ratio;
        if (total_ratio <= 0.0)
        {
            throw std::invalid_argument("[Datasets::california_housing] Split ratios must sum to > 0");
        }

        size_t train_size = static_cast<size_t>(std::round(total_samples * (train_ratio / total_ratio)));
        size_t test_size = static_cast<size_t>(std::round(total_samples * (test_ratio / total_ratio)));
        if (train_size + test_size > total_samples)
        {
            test_size = total_samples - train_size;
        }
        size_t val_size = total_samples - train_size - test_size;

        // Helper to construct X and y matrices for an index slice
        auto build_matrices = [&](size_t start_idx, size_t count) -> std::pair<Matrix, Matrix> {
            std::vector<double> X_data(count * num_features, 0.0);
            std::vector<double> y_data(count, 0.0);

            for (size_t i = 0; i < count; ++i)
            {
                size_t src_idx = indices[start_idx + i];
                const auto &r = rows[src_idx];

                size_t x_offset = i * num_features;
                X_data[x_offset + 0] = r.longitude;
                X_data[x_offset + 1] = r.latitude;
                X_data[x_offset + 2] = r.housing_median_age;
                X_data[x_offset + 3] = r.total_rooms;
                X_data[x_offset + 4] = r.total_bedrooms;
                X_data[x_offset + 5] = r.population;
                X_data[x_offset + 6] = r.households;
                X_data[x_offset + 7] = r.median_income;

                if (one_hot_categorical)
                {
                    for (int c = 0; c < 5; ++c)
                    {
                        X_data[x_offset + 8 + c] = (r.ocean_cat == c) ? 1.0 : 0.0;
                    }
                }
                else
                {
                    X_data[x_offset + 8] = static_cast<double>(r.ocean_cat);
                }

                y_data[i] = r.median_house_value;
            }

            return {Matrix(count, num_features, X_data), Matrix(count, 1, y_data)};
        };

        auto [X_train, y_train] = build_matrices(0, train_size);
        auto [X_test, y_test] = build_matrices(train_size, test_size);
        auto [X_val, y_val] = build_matrices(train_size + test_size, val_size);

        return {X_train, y_train, X_test, y_test, X_val, y_val};
    }

    static std::tuple<Matrix, Matrix, Matrix, Matrix, Matrix, Matrix> california_housing(
        const std::string &filename,
        double train_ratio = 0.7,
        double test_ratio = 0.15,
        double val_ratio = 0.15,
        bool shuffle = true,
        unsigned int seed = 42,
        bool one_hot_categorical = true)
    {
        return california_housing(train_ratio, test_ratio, val_ratio, shuffle, seed, one_hot_categorical, filename);
    }

    static auto california_housing_data(double train_ratio = 0.7,
                                        double test_ratio = 0.15,
                                        double val_ratio = 0.15,
                                        bool shuffle = true,
                                        unsigned int seed = 42,
                                        bool one_hot_categorical = true,
                                        const std::string &filename = "housing_data.csv")
    {
        return california_housing(train_ratio, test_ratio, val_ratio, shuffle, seed, one_hot_categorical, filename);
    }

    static auto housing_data(double train_ratio = 0.7,
                             double test_ratio = 0.15,
                             double val_ratio = 0.15,
                             bool shuffle = true,
                             unsigned int seed = 42,
                             bool one_hot_categorical = true,
                             const std::string &filename = "housing_data.csv")
    {
        return california_housing(train_ratio, test_ratio, val_ratio, shuffle, seed, one_hot_categorical, filename);
    }
};
