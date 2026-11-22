#pragma once

#include "cpptorch/core/matrix.h"
#include "cpptorch/core/numcpp.h"
#include "cpptorch/utils/resources.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numbers>
#include <numeric>
#include <omp.h>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cpptorch
{

/**
 * @brief A single flow-field sample from a 2D Navier-Stokes dataset.
 */
struct NavierStokesFlowField
{
    double x = 0.0;
    double y = 0.0;
    double omega = 0.0;
    double u = 0.0;
    double v = 0.0;
};

/**
 * @brief One parsed row of a Navier-Stokes CSV file.
 */
struct NavierStokesRecord
{
    int sample = 0;
    int frame = 0;
    int ix = 0;
    int iy = 0;
    double time = 0.0;
    NavierStokesFlowField field;
};

/**
 * @brief A parsed Navier-Stokes dataset.
 *
 * Holds the full spatiotemporal grid of flow fields together with the
 * coordinates and timestamps needed to reconstruct per-frame inputs and to
 * evaluate PDE residuals.
 */
struct NavierStokesDataset
{
    size_t sample_count = 0;
    size_t frame_count = 0;
    size_t nx = 0;
    size_t ny = 0;
    std::vector<double> times;
    std::vector<double> x_coordinates;
    std::vector<double> y_coordinates;
    std::vector<NavierStokesFlowField> fields;

    [[nodiscard]] size_t index(size_t sample, size_t frame, size_t cell) const
    {
        return (((sample * frame_count) + frame) * nx * ny) + cell;
    }
};

class Datasets
{
private:
    /**
     * Opens a dataset CSV, preferring the configured resources directory and
     * falling back to the current working directory.
     */
    static std::ifstream open_dataset(const std::string &filename)
    {
        std::ifstream file;
        file.open(Resources::path(filename));
        if (!file.is_open())
        {
            file.open("./resources/" + filename);
        }
        if (!file.is_open())
        {
            file.open(filename);
        }
        return file;
    }

    /**
     * Splits a CSV line into its comma-separated columns.
     */
    [[nodiscard]] static std::vector<std::string> split_csv_line(const std::string &line)
    {
        std::vector<std::string> columns;
        std::stringstream stream(line);
        std::string value;
        while (std::getline(stream, value, ','))
            columns.push_back(value);
        return columns;
    }

    /**
     * Parses an integer CSV cell, throwing on invalid input.
     */
    [[nodiscard]] static int parse_integer(const std::string &value, const char *column_name)
    {
        int result = 0;
        const char *begin = value.data();
        const char *end = begin + value.size();
        const auto parsed = std::from_chars(begin, end, result);
        if (parsed.ec != std::errc{} || parsed.ptr != end)
            throw std::runtime_error(std::string("Invalid integer in CSV column '") + column_name + "'.");
        return result;
    }

    /**
     * Parses a floating-point CSV cell, throwing on invalid or non-finite input.
     */
    [[nodiscard]] static double parse_number(const std::string &value, const char *column_name)
    {
        size_t parsed_length = 0;
        const double result = std::stod(value, &parsed_length);
        if (parsed_length != value.size() || !std::isfinite(result))
            throw std::runtime_error(std::string("Invalid number in CSV column '") + column_name + "'.");
        return result;
    }

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
        Matrix y = NumCpp::sin(2.0 * std::numbers::pi * X);
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
        std::ifstream file = open_dataset(filename);
        if (!file.is_open())
        {
            throw std::runtime_error("[Datasets::california_housing] Failed to open CSV file: " + filename);
        }

        std::string line;
        if (!std::getline(file, line))
        {
            throw std::runtime_error("[Datasets::california_housing] Empty CSV file: " + filename);
        }

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
            {
                ocean_cat = 1;
            }
            else if (ocean == "ISLAND")
            {
                ocean_cat = 2;
            }
            else if (ocean == "NEAR BAY")
            {
                ocean_cat = 3;
            }
            else if (ocean == "NEAR OCEAN")
            {
                ocean_cat = 4;
            }

            if (one_hot_categorical)
            {
                for (int c = 0; c < 5; ++c)
                {
                    raw_X.push_back((ocean_cat == c) ? 1.0 : 0.0);
                }
            }
            else
            {
                raw_X.push_back(static_cast<double>(ocean_cat));
            }

            row_idx++;
        }

        if (raw_y.empty())
        {
            throw std::runtime_error("[Datasets::california_housing] No valid rows parsed from " + filename);
        }

        // Impute missing total_bedrooms with dataset median
        double median_bedrooms = 0.0;
        if (!valid_bedrooms.empty())
        {
            std::ranges::sort(valid_bedrooms);
            size_t mid = valid_bedrooms.size() / 2;
            median_bedrooms = (valid_bedrooms.size() % 2 == 0) ? (valid_bedrooms[mid - 1] + valid_bedrooms[mid]) / 2.0
                                                               : valid_bedrooms[mid];
        }

        for (size_t idx : missing_bedroom_indices)
        {
            raw_X[(idx * num_features) + 4] = median_bedrooms;
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

        auto train_size = static_cast<size_t>(std::round(total_samples * (train_ratio / total_ratio)));
        auto test_size = static_cast<size_t>(std::round(total_samples * (test_ratio / total_ratio)));
        if (train_size + test_size > total_samples)
        {
            test_size = total_samples - train_size;
        }
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
        std::ifstream file = open_dataset(filename);
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

        auto train_size = static_cast<size_t>(std::round(total_samples * (train_ratio / total_ratio)));
        auto test_size = static_cast<size_t>(std::round(total_samples * (test_ratio / total_ratio)));
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

    /**
     * Loads a 2D Navier-Stokes flow-field dataset from CSV.
     *
     * The CSV must contain the columns: sample, frame, t, ix, iy, x, y, w, u, v.
     * The grid is validated to be contiguous and complete (one row per
     * sample/frame/cell combination). Timestamps and coordinates are checked for
     * consistency across the grid.
     *
     * @param filename Name of the file inside the resources directory.
     * @return The parsed dataset with flow fields, coordinates and timestamps.
     * @throws std::runtime_error if the file cannot be opened or is malformed.
     */
    [[nodiscard]] static NavierStokesDataset load_navier_stokes(const std::string &filename = "ns2d_subset.csv")
    {
        std::ifstream file = open_dataset(filename);
        if (!file.is_open())
        {
            throw std::runtime_error("[Datasets::load_navier_stokes] Failed to open CSV file: " + filename);
        }

        std::string line;
        if (!std::getline(file, line))
        {
            throw std::runtime_error("[Datasets::load_navier_stokes] Empty CSV file: " + filename);
        }

        const std::vector<std::string> header = split_csv_line(line);
        std::unordered_map<std::string, size_t> column_index;
        for (size_t i = 0; i < header.size(); ++i)
            column_index.emplace(header[i], i);

        const std::array<std::string, 10> required_columns =
            {"sample", "frame", "t", "ix", "iy", "x", "y", "w", "u", "v"};
        for (const std::string &name : required_columns)
            if (column_index.find(name) == column_index.end())
                throw std::runtime_error("[Datasets::load_navier_stokes] Dataset is missing required column '" + name +
                                         "'.");

        std::vector<NavierStokesRecord> records;
        std::set<int> sample_ids;
        std::set<int> frame_ids;
        std::set<int> x_indices;
        std::set<int> y_indices;

        while (std::getline(file, line))
        {
            if (line.empty())
                continue;
            const std::vector<std::string> values = split_csv_line(line);
            if (values.size() != header.size())
                throw std::runtime_error("[Datasets::load_navier_stokes] Dataset row has a different number of "
                                         "columns than its header.");

            const auto value_at = [&](const char *name) -> const std::string & {
                return values[column_index.at(name)];
            };

            NavierStokesRecord record;
            record.sample = parse_integer(value_at("sample"), "sample");
            record.frame = parse_integer(value_at("frame"), "frame");
            record.ix = parse_integer(value_at("ix"), "ix");
            record.iy = parse_integer(value_at("iy"), "iy");
            record.time = parse_number(value_at("t"), "t");
            record.field.x = parse_number(value_at("x"), "x");
            record.field.y = parse_number(value_at("y"), "y");
            record.field.omega = parse_number(value_at("w"), "w");
            record.field.u = parse_number(value_at("u"), "u");
            record.field.v = parse_number(value_at("v"), "v");

            if (record.sample < 0 || record.frame < 0 || record.ix < 0 || record.iy < 0)
                throw std::runtime_error("[Datasets::load_navier_stokes] Dataset indices must be non-negative.");

            sample_ids.insert(record.sample);
            frame_ids.insert(record.frame);
            x_indices.insert(record.ix);
            y_indices.insert(record.iy);
            records.push_back(record);
        }

        if (records.empty() || sample_ids.empty() || frame_ids.size() < 3 || x_indices.size() < 3 ||
            y_indices.size() < 3)
        {
            throw std::runtime_error("[Datasets::load_navier_stokes] Dataset must contain samples, at least three "
                                     "frames, and a 3x3 grid.");
        }

        const auto contiguous_indices = [](const std::set<int> &indices) {
            int expected = 0;
            for (const int index : indices)
                if (index != expected++)
                    return false;
            return true;
        };
        if (!contiguous_indices(x_indices) || !contiguous_indices(y_indices))
        {
            throw std::runtime_error("[Datasets::load_navier_stokes] Grid indices must start at zero and be "
                                     "contiguous.");
        }

        NavierStokesDataset dataset;
        dataset.sample_count = sample_ids.size();
        dataset.frame_count = frame_ids.size();
        dataset.nx = x_indices.size();
        dataset.ny = y_indices.size();
        const size_t grid_size = dataset.nx * dataset.ny;
        const size_t expected_rows = dataset.sample_count * dataset.frame_count * grid_size;
        if (records.size() != expected_rows)
        {
            throw std::runtime_error("[Datasets::load_navier_stokes] Dataset does not contain exactly one complete "
                                     "grid for each sample and frame.");
        }

        dataset.fields.resize(expected_rows);
        dataset.times.resize(dataset.frame_count, std::numeric_limits<double>::quiet_NaN());
        dataset.x_coordinates.resize(dataset.nx, std::numeric_limits<double>::quiet_NaN());
        dataset.y_coordinates.resize(dataset.ny, std::numeric_limits<double>::quiet_NaN());
        std::vector<bool> seen(expected_rows, false);

        const std::vector<int> ordered_samples(sample_ids.begin(), sample_ids.end());
        const std::vector<int> ordered_frames(frame_ids.begin(), frame_ids.end());
        for (const NavierStokesRecord &record : records)
        {
            const size_t sample =
                static_cast<size_t>(std::lower_bound(ordered_samples.begin(), ordered_samples.end(), record.sample) -
                                    ordered_samples.begin());
            const size_t frame = static_cast<size_t>(
                std::lower_bound(ordered_frames.begin(), ordered_frames.end(), record.frame) - ordered_frames.begin());
            const size_t ix = static_cast<size_t>(record.ix);
            const size_t iy = static_cast<size_t>(record.iy);
            const size_t cell = iy * dataset.nx + ix;
            const size_t point_index = dataset.index(sample, frame, cell);

            if (seen[point_index])
            {
                throw std::runtime_error("[Datasets::load_navier_stokes] Dataset contains a duplicate "
                                         "sample/frame/grid location.");
            }
            seen[point_index] = true;
            dataset.fields[point_index] = record.field;

            if (std::isnan(dataset.times[frame]))
                dataset.times[frame] = record.time;
            else if (std::abs(dataset.times[frame] - record.time) > 1e-10)
            {
                throw std::runtime_error("[Datasets::load_navier_stokes] Rows in the same frame have inconsistent "
                                         "timestamps.");
            }

            if (std::isnan(dataset.x_coordinates[ix]))
                dataset.x_coordinates[ix] = record.field.x;
            else if (std::abs(dataset.x_coordinates[ix] - record.field.x) > 1e-10)
            {
                throw std::runtime_error("[Datasets::load_navier_stokes] X coordinates are inconsistent across grid "
                                         "rows.");
            }

            if (std::isnan(dataset.y_coordinates[iy]))
                dataset.y_coordinates[iy] = record.field.y;
            else if (std::abs(dataset.y_coordinates[iy] - record.field.y) > 1e-10)
            {
                throw std::runtime_error("[Datasets::load_navier_stokes] Y coordinates are inconsistent across grid "
                                         "rows.");
            }
        }

        if (std::find(seen.begin(), seen.end(), false) != seen.end())
        {
            throw std::runtime_error("[Datasets::load_navier_stokes] Dataset is missing one or more "
                                     "sample/frame/grid locations.");
        }

        const auto grid_spacing = [](const std::vector<double> &coordinates, const char *axis) {
            const double spacing = coordinates[1] - coordinates[0];
            if (!(spacing > 0.0))
                throw std::runtime_error(std::string("[Datasets::load_navier_stokes] The ") + axis +
                                         " grid must be increasing.");
            for (size_t i = 1; i < coordinates.size(); ++i)
                if (std::abs((coordinates[i] - coordinates[i - 1]) - spacing) > 1e-8)
                    throw std::runtime_error(std::string("[Datasets::load_navier_stokes] The ") + axis +
                                             " grid must be uniformly spaced.");
            return spacing;
        };
        (void)grid_spacing(dataset.x_coordinates, "x");
        (void)grid_spacing(dataset.y_coordinates, "y");

        return dataset;
    }

    /**
     * Loads the SVM benchmark dataset from CSV.
     *
     * The CSV must contain the columns: x1, x2, label, where label is either
     * +1 or -1. Returns the full feature matrix of shape (n_samples, 2) and the
     * label column of shape (n_samples, 1).
     *
     * @param filename Name of the file inside the resources directory.
     * @return A tuple of (X, y) with X of shape (n_samples, 2) and y of shape
     *         (n_samples, 1) holding +/- 1 labels.
     * @throws std::runtime_error if the file cannot be opened or is malformed.
     */
    [[nodiscard]] static std::tuple<Matrix, Matrix> svm_data(const std::string &filename = "svm_dataset.csv")
    {
        std::ifstream file = open_dataset(filename);
        if (!file.is_open())
        {
            throw std::runtime_error("[Datasets::svm_data] Failed to open CSV file: " + filename);
        }

        std::string line;
        if (!std::getline(file, line))
        {
            throw std::runtime_error("[Datasets::svm_data] Empty CSV file: " + filename);
        }

        std::vector<double> raw_X;
        std::vector<double> raw_y;
        raw_X.reserve(8000 * 2);
        raw_y.reserve(8000);

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

            if (tokens.size() < 3)
                continue;

            raw_X.push_back(std::stod(tokens[0]));
            raw_X.push_back(std::stod(tokens[1]));
            raw_y.push_back(std::stod(tokens[2]));
        }

        if (raw_y.empty())
        {
            throw std::runtime_error("[Datasets::svm_data] No valid rows parsed from " + filename);
        }

        size_t total_samples = raw_y.size();
        const size_t num_features = 2;

        Matrix X(total_samples, num_features);
        Matrix y(total_samples, 1);

        for (size_t i = 0; i < total_samples; ++i)
        {
            X(i, 0) = raw_X[i * 2];
            X(i, 1) = raw_X[i * 2 + 1];
            y(i, 0) = raw_y[i];
        }

        return {X, y};
    }
};
} // namespace cpptorch
