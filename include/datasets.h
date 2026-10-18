#pragma once

#include "core/matrix.h"
#include "core/numcpp.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>
#include <omp.h>
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
        Matrix X(samples * classes, 2);
        Matrix y(samples * classes, 1);

        for (size_t class_number = 0; class_number < classes; ++class_number)
        {
            size_t start_ix = samples * class_number;
            double target_class = static_cast<double>(class_number);

            // Hoist static math calculations out of array declarations
            double t_start = target_class * 4.0;
            double t_end = (target_class + 1.0) * 4.0;

            Matrix r = NumCpp::linspace(0.0, 1.0, samples);
            Matrix t = NumCpp::linspace(t_start, t_end, samples) + (NumCpp::randn({samples, 1}) * 0.2);

            Matrix sin_t = NumCpp::sin(t * 2.5);
            Matrix cos_t = NumCpp::cos(t * 2.5);

            Matrix r_sin = r * sin_t;
            Matrix r_cos = r * cos_t;

#pragma omp parallel for simd if (samples > 1024)
            for (size_t i = 0; i < samples; ++i)
            {
                size_t ix = start_ix + i;
                X(ix, 0) = r_sin[i];
                X(ix, 1) = r_cos[i];
                y(ix, 0) = target_class;
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
            file.open("./resources/" + filename);
        if (!file.is_open())
            file.open(filename);
        if (!file.is_open())
            throw std::runtime_error("[Datasets::california_housing] Failed to open CSV file: " + filename);

        std::string line;
        if (!std::getline(file, line))
            throw std::runtime_error("[Datasets::california_housing] Empty CSV file: " + filename);

        // Pre-allocate large continuous buffers for matrix creation to avoid struct-to-matrix mapping penalty
        std::vector<double> raw_X;
        std::vector<double> raw_y;
        std::vector<double> valid_bedrooms;
        std::vector<size_t> missing_bedroom_indices;

        raw_X.reserve(20000 * (one_hot_categorical ? 13 : 9));
        raw_y.reserve(20000);

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

        size_t row_idx = 0;
        size_t num_features = one_hot_categorical ? 13 : 9;

        while (std::getline(file, line))
        {
            if (!line.empty() && line.back() == '\r')
                line.pop_back();
            if (line.empty())
                continue;

            std::stringstream ss(line);
            std::string token;
            std::vector<std::string> tokens;
            while (std::getline(ss, token, ','))
            {
                trim(token);
                tokens.push_back(token);
            }

            if (tokens.size() < 10)
                continue;

            // Direct buffer insertion
            raw_X.push_back(std::stod(tokens[0]));
            raw_X.push_back(std::stod(tokens[1]));
            raw_X.push_back(std::stod(tokens[2]));
            raw_X.push_back(std::stod(tokens[3]));

            if (tokens[4].empty())
            {
                missing_bedroom_indices.push_back(row_idx);
                raw_X.push_back(0.0); // Placeholder
            }
            else
            {
                double beds = std::stod(tokens[4]);
                raw_X.push_back(beds);
                valid_bedrooms.push_back(beds);
            }

            raw_X.push_back(std::stod(tokens[5]));
            raw_X.push_back(std::stod(tokens[6]));
            raw_X.push_back(std::stod(tokens[7]));
            raw_y.push_back(std::stod(tokens[8]));

            int ocean_cat = 0;
            const std::string &ocean = tokens[9];
            if (ocean == "INLAND")
                ocean_cat = 1;
            else if (ocean == "ISLAND")
                ocean_cat = 2;
            else if (ocean == "NEAR BAY")
                ocean_cat = 3;
            else if (ocean == "NEAR OCEAN")
                ocean_cat = 4;

            if (one_hot_categorical)
            {
                for (int c = 0; c < 5; ++c)
                    raw_X.push_back((ocean_cat == c) ? 1.0 : 0.0);
            }
            else
            {
                raw_X.push_back(static_cast<double>(ocean_cat));
            }

            row_idx++;
        }

        if (raw_y.empty())
            throw std::runtime_error("[Datasets::california_housing] No valid rows parsed from " + filename);

        // Impute missing total_bedrooms with dataset median
        double median_bedrooms = 0.0;
        if (!valid_bedrooms.empty())
        {
            std::sort(valid_bedrooms.begin(), valid_bedrooms.end());
            size_t mid = valid_bedrooms.size() / 2;
            median_bedrooms = (valid_bedrooms.size() % 2 == 0) ? (valid_bedrooms[mid - 1] + valid_bedrooms[mid]) / 2.0
                                                               : valid_bedrooms[mid];
        }

        for (size_t idx : missing_bedroom_indices)
        {
            raw_X[idx * num_features + 4] = median_bedrooms;
        }

        size_t total_samples = raw_y.size();

        std::vector<size_t> indices(total_samples);
        std::iota(indices.begin(), indices.end(), 0);
        if (shuffle)
        {
            std::mt19937 gen(seed);
            std::shuffle(indices.begin(), indices.end(), gen);
        }

        double total_ratio = train_ratio + test_ratio + val_ratio;
        if (total_ratio <= 0.0)
            throw std::invalid_argument("[Datasets::california_housing] Split ratios must sum to > 0");

        size_t train_size = static_cast<size_t>(std::round(total_samples * (train_ratio / total_ratio)));
        size_t test_size = static_cast<size_t>(std::round(total_samples * (test_ratio / total_ratio)));
        if (train_size + test_size > total_samples)
            test_size = total_samples - train_size;
        size_t val_size = total_samples - train_size - test_size;

        auto build_matrices = [&](size_t start_idx, size_t count) -> std::pair<Matrix, Matrix> {
            std::vector<double> X_split(count * num_features);
            std::vector<double> y_split(count);

            for (size_t i = 0; i < count; ++i)
            {
                size_t src_idx = indices[start_idx + i];
                size_t src_x_offset = src_idx * num_features;
                size_t dest_x_offset = i * num_features;

                // Fast flat memory pointer redirection using std::copy
                std::copy(raw_X.begin() + src_x_offset,
                          raw_X.begin() + src_x_offset + num_features,
                          X_split.begin() + dest_x_offset);
                y_split[i] = raw_y[src_idx];
            }

            return {Matrix(count, num_features, X_split), Matrix(count, 1, y_split)};
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


    static std::tuple<Matrix, Matrix, Matrix, Matrix, Matrix, Matrix> gender_audio_data(
        const std::string &filename = "gender_audio_features_kaggle.csv",
        double train_ratio = 0.7,
        double test_ratio = 0.15,
        double val_ratio = 0.15,
        bool shuffle = true,
        unsigned int seed = 42)
    {
        std::ifstream file;
#ifdef RESOURCES_PATH
        file.open(std::string(RESOURCES_PATH) + filename);
#endif
        if (!file.is_open())
            file.open("./resources/" + filename);
        if (!file.is_open())
            file.open(filename);
        if (!file.is_open())
            throw std::runtime_error("[Datasets::gender_audio_data] Failed to open CSV file: " + filename);

        std::string line;
        // Skip header
        if (!std::getline(file, line))
            throw std::runtime_error("[Datasets::gender_audio_data] Empty CSV file: " + filename);

        // Pre-allocate large continuous buffers for matrix creation to avoid struct-to-matrix mapping penalty
        std::vector<double> raw_X;
        std::vector<double> raw_y;

        // The Kaggle dataset has 43 features + 1 label
        const size_t num_features = 43;
        raw_X.reserve(17000 * num_features);
        raw_y.reserve(17000);

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
                line.pop_back();
            if (line.empty())
                continue;

            std::stringstream ss(line);
            std::string token;
            std::vector<std::string> tokens;
            while (std::getline(ss, token, ','))
            {
                trim(token);
                tokens.push_back(token);
            }

            // Index 0-42: numeric audio features
            // Index 43: label (target 0 or 1)
            if (tokens.size() < 44)
                continue;

            // Direct buffer insertion for the 43 features
            for (size_t i = 0; i < 43; ++i)
            {
                raw_X.push_back(std::stod(tokens[i]));
            }

            // Insert target label at index 43
            raw_y.push_back(std::stod(tokens[43]));
        }

        if (raw_y.empty())
            throw std::runtime_error("[Datasets::gender_audio_data] No valid rows parsed from " + filename);

        size_t total_samples = raw_y.size();

        std::vector<size_t> indices(total_samples);
        std::iota(indices.begin(), indices.end(), 0);
        if (shuffle)
        {
            std::mt19937 gen(seed);
            std::shuffle(indices.begin(), indices.end(), gen);
        }

        double total_ratio = train_ratio + test_ratio + val_ratio;
        if (total_ratio <= 0.0)
            throw std::invalid_argument("[Datasets::gender_audio_data] Split ratios must sum to > 0");

        size_t train_size = static_cast<size_t>(std::round(total_samples * (train_ratio / total_ratio)));
        size_t test_size = static_cast<size_t>(std::round(total_samples * (test_ratio / total_ratio)));
        if (train_size + test_size > total_samples)
            test_size = total_samples - train_size;
        size_t val_size = total_samples - train_size - test_size;

        auto build_matrices = [&](size_t start_idx, size_t count) -> std::pair<Matrix, Matrix> {
            std::vector<double> X_split(count * num_features);
            std::vector<double> y_split(count);

            for (size_t i = 0; i < count; ++i)
            {
                size_t src_idx = indices[start_idx + i];
                size_t src_x_offset = src_idx * num_features;
                size_t dest_x_offset = i * num_features;

                // Fast flat memory pointer redirection using std::copy
                std::copy(raw_X.begin() + src_x_offset,
                          raw_X.begin() + src_x_offset + num_features,
                          X_split.begin() + dest_x_offset);
                y_split[i] = raw_y[src_idx];
            }

            return {Matrix(count, num_features, X_split), Matrix(count, 1, y_split)};
        };

        auto [X_train, y_train] = build_matrices(0, train_size);
        auto [X_test, y_test] = build_matrices(train_size, test_size);
        auto [X_val, y_val] = build_matrices(train_size + test_size, val_size);

        return {X_train, y_train, X_test, y_test, X_val, y_val};
    }


    static std::tuple<Matrix, Matrix, Matrix, Matrix, Matrix, Matrix> emotion_audio_data(
        double train_ratio = 0.7,
        double test_ratio = 0.15,
        double val_ratio = 0.15,
        bool shuffle = true,
        unsigned int seed = 42,
        const std::string &filename = "emotion_audio_features.csv")
    {
        std::ifstream file;
#ifdef RESOURCES_PATH
        file.open(std::string(RESOURCES_PATH) + filename);
#endif
        if (!file.is_open())
            file.open("./resources/" + filename);
        if (!file.is_open())
            file.open(filename);
        if (!file.is_open())
            throw std::runtime_error("[Datasets::emotion_audio_data] Failed to open CSV file: " + filename);

        std::string line;
        // Skip header
        if (!std::getline(file, line))
            throw std::runtime_error("[Datasets::emotion_audio_data] Empty CSV file: " + filename);

        // Pre-allocate large continuous buffers for matrix creation
        std::vector<double> raw_X;
        std::vector<double> raw_y;

        // The Emotion dataset has 80 numeric features + 1 label
        const size_t num_features = 80;
        // The dataset is 4200 samples. Reserving a safe margin.
        raw_X.reserve(4500 * num_features);
        raw_y.reserve(4500);

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
                line.pop_back();
            if (line.empty())
                continue;

            std::stringstream ss(line);
            std::string token;
            std::vector<std::string> tokens;
            while (std::getline(ss, token, ','))
            {
                trim(token);
                tokens.push_back(token);
            }

            // The dataset contains 85 columns.
            // Index 0: sample_id (ignore)
            // Index 1: speaker_id (ignore)
            // Index 2: gender (ignore)
            // Index 3-82: numeric audio features
            // Index 83: emotion string (ignore)
            // Index 84: emotion_id (target 0 to 5)
            if (tokens.size() < 85)
                continue;

            // Direct buffer insertion for the 80 numeric features
            for (size_t i = 3; i <= 82; ++i)
            {
                raw_X.push_back(std::stod(tokens[i]));
            }

            // Insert target label_id at index 84
            raw_y.push_back(std::stod(tokens[84]));
        }

        if (raw_y.empty())
            throw std::runtime_error("[Datasets::emotion_audio_data] No valid rows parsed from " + filename);

        size_t total_samples = raw_y.size();

        std::vector<size_t> indices(total_samples);
        std::iota(indices.begin(), indices.end(), 0);
        if (shuffle)
        {
            std::mt19937 gen(seed);
            std::shuffle(indices.begin(), indices.end(), gen);
        }

        double total_ratio = train_ratio + test_ratio + val_ratio;
        if (total_ratio <= 0.0)
            throw std::invalid_argument("[Datasets::emotion_audio_data] Split ratios must sum to > 0");

        size_t train_size = static_cast<size_t>(std::round(total_samples * (train_ratio / total_ratio)));
        size_t test_size = static_cast<size_t>(std::round(total_samples * (test_ratio / total_ratio)));
        if (train_size + test_size > total_samples)
            test_size = total_samples - train_size;
        size_t val_size = total_samples - train_size - test_size;

        auto build_matrices = [&](size_t start_idx, size_t count) -> std::pair<Matrix, Matrix> {
            std::vector<double> X_split(count * num_features);
            std::vector<double> y_split(count);

            for (size_t i = 0; i < count; ++i)
            {
                size_t src_idx = indices[start_idx + i];
                size_t src_x_offset = src_idx * num_features;
                size_t dest_x_offset = i * num_features;

                // Fast flat memory pointer redirection using std::copy
                std::copy(raw_X.begin() + src_x_offset,
                          raw_X.begin() + src_x_offset + num_features,
                          X_split.begin() + dest_x_offset);
                y_split[i] = raw_y[src_idx];
            }

            return {Matrix(count, num_features, X_split), Matrix(count, 1, y_split)};
        };

        auto [X_train, y_train] = build_matrices(0, train_size);
        auto [X_test, y_test] = build_matrices(train_size, test_size);
        auto [X_val, y_val] = build_matrices(train_size + test_size, val_size);

        return {X_train, y_train, X_test, y_test, X_val, y_val};
    }
};