#include <cpptorch/cpptorch.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

struct TargetScale
{
    std::array<double, 3> mean{};
    std::array<double, 3> standard_deviation{};

    void fit_transform(cpptorch::Matrix &targets)
    {
        for (size_t column = 0; column < mean.size(); ++column)
        {
            for (size_t row = 0; row < targets.rows(); ++row)
                mean[column] += targets(row, column);
            mean[column] /= static_cast<double>(targets.rows());

            double variance = 0.0;
            for (size_t row = 0; row < targets.rows(); ++row)
            {
                const double difference = targets(row, column) - mean[column];
                variance += difference * difference;
            }
            standard_deviation[column] =
                std::max(std::sqrt(variance / static_cast<double>(targets.rows())), 1e-12);

            for (size_t row = 0; row < targets.rows(); ++row)
                targets(row, column) = (targets(row, column) - mean[column]) / standard_deviation[column];
        }
    }

    void transform(cpptorch::Matrix &targets) const
    {
        for (size_t row = 0; row < targets.rows(); ++row)
            for (size_t column = 0; column < mean.size(); ++column)
                targets(row, column) = (targets(row, column) - mean[column]) / standard_deviation[column];
    }

    [[nodiscard]] double restore(double value, size_t column) const
    {
        return value * standard_deviation[column] + mean[column];
    }
};

struct ResidualMetrics
{
    double continuity_squared_sum = 0.0;
    double vorticity_transport_squared_sum = 0.0;
    size_t count = 0;
};

[[nodiscard]] double parse_positive_double(const char *text, const char *name)
{
    const std::string value(text);
    const double result = std::stod(value);
    if (result <= 0.0)
        throw std::runtime_error(std::string(name) + " must be greater than zero.");
    return result;
}

[[nodiscard]] size_t parse_positive_size(const char *text, const char *name)
{
    const std::string value(text);
    size_t result = 0;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || result == 0)
        throw std::runtime_error(std::string(name) + " must be a positive integer.");
    return result;
}

[[nodiscard]] ResidualMetrics calculate_residuals(const cpptorch::NavierStokesDataset &dataset,
                                                  const std::vector<cpptorch::NavierStokesFlowField> &fields,
                                                  size_t first_validation_frame,
                                                  double viscosity)
{
    const double dx = dataset.x_coordinates[1] - dataset.x_coordinates[0];
    const double dy = dataset.y_coordinates[1] - dataset.y_coordinates[0];
    const double inverse_2dx = 1.0 / (2.0 * dx);
    const double inverse_2dy = 1.0 / (2.0 * dy);
    const double inverse_dx_squared = 1.0 / (dx * dx);
    const double inverse_dy_squared = 1.0 / (dy * dy);
    ResidualMetrics metrics;

    for (size_t sample = 0; sample < dataset.sample_count; ++sample)
    {
        for (size_t frame = first_validation_frame; frame + 1 < dataset.frame_count; ++frame)
        {
            const double elapsed_time = dataset.times[frame + 1] - dataset.times[frame - 1];
            if (!(elapsed_time > 0.0))
                throw std::runtime_error("Frame timestamps must be strictly increasing.");

            for (size_t iy = 0; iy < dataset.ny; ++iy)
            {
                const size_t iy_before = (iy + dataset.ny - 1) % dataset.ny;
                const size_t iy_after = (iy + 1) % dataset.ny;
                for (size_t ix = 0; ix < dataset.nx; ++ix)
                {
                    const size_t ix_before = (ix + dataset.nx - 1) % dataset.nx;
                    const size_t ix_after = (ix + 1) % dataset.nx;
                    const size_t cell = iy * dataset.nx + ix;
                    const auto &center = fields[dataset.index(sample, frame, cell)];
                    const auto &left = fields[dataset.index(sample, frame, iy * dataset.nx + ix_before)];
                    const auto &right = fields[dataset.index(sample, frame, iy * dataset.nx + ix_after)];
                    const auto &below = fields[dataset.index(sample, frame, iy_before * dataset.nx + ix)];
                    const auto &above = fields[dataset.index(sample, frame, iy_after * dataset.nx + ix)];
                    const auto &previous =
                        fields[dataset.index(sample, frame - 1, cell)];
                    const auto &next =
                        fields[dataset.index(sample, frame + 1, cell)];

                    const double u_x = (right.u - left.u) * inverse_2dx;
                    const double v_y = (above.v - below.v) * inverse_2dy;
                    const double omega_x = (right.omega - left.omega) * inverse_2dx;
                    const double omega_y = (above.omega - below.omega) * inverse_2dy;
                    const double omega_t = (next.omega - previous.omega) / elapsed_time;
                    const double laplacian_omega =
                        (right.omega - 2.0 * center.omega + left.omega) * inverse_dx_squared +
                        (above.omega - 2.0 * center.omega + below.omega) * inverse_dy_squared;

                    const double continuity = u_x + v_y;
                    const double vorticity_transport =
                        omega_t + center.u * omega_x + center.v * omega_y - viscosity * laplacian_omega;
                    metrics.continuity_squared_sum += continuity * continuity;
                    metrics.vorticity_transport_squared_sum += vorticity_transport * vorticity_transport;
                    ++metrics.count;
                }
            }
        }
    }
    return metrics;
}

void print_residuals(const char *label, const ResidualMetrics &metrics)
{
    if (metrics.count == 0)
        throw std::runtime_error("No validation frames available for PDE residual diagnostics.");
    const double inverse_count = 1.0 / static_cast<double>(metrics.count);
    std::cout << label << " PDE residual RMS: continuity="
              << std::sqrt(metrics.continuity_squared_sum * inverse_count)
              << ", vorticity-transport="
              << std::sqrt(metrics.vorticity_transport_squared_sum * inverse_count) << '\n';
}

} // namespace

int main(int argc, char **argv)
{
    try
    {
#ifndef CPPTORCH_DEFAULT_RESOURCES_PATH
#define CPPTORCH_DEFAULT_RESOURCES_PATH "resources"
#endif
        if (argc > 4)
            throw std::runtime_error("Usage: Navier-Stokes [dataset.csv] [epochs] [kinematic_viscosity]");

        const std::string dataset_filename = argc > 1 ? argv[1] : "ns2d_subset.csv";
        const size_t epochs = argc > 2 ? parse_positive_size(argv[2], "epochs") : 30;
        const double viscosity = argc > 3 ? parse_positive_double(argv[3], "kinematic_viscosity") : 0.01;

        const cpptorch::NavierStokesDataset dataset =
            cpptorch::Datasets::load_navier_stokes(dataset_filename);
        const size_t grid_size = dataset.nx * dataset.ny;
        const size_t training_frame_count = std::max<size_t>(2, (dataset.frame_count * 4 + 2) / 5);
        if (training_frame_count >= dataset.frame_count)
            throw std::runtime_error("Dataset needs at least one held-out frame.");

        const size_t training_rows = dataset.sample_count * training_frame_count * grid_size;
        const size_t test_rows =
            dataset.sample_count * (dataset.frame_count - training_frame_count) * grid_size;
        std::vector<size_t> training_order(training_rows);
        std::iota(training_order.begin(), training_order.end(), 0);
        std::mt19937 generator(42);
        std::shuffle(training_order.begin(), training_order.end(), generator);

        std::vector<double> training_features(training_rows * 3);
        std::vector<double> training_targets(training_rows * 3);
        std::vector<double> test_features(test_rows * 3);
        std::vector<double> test_targets(test_rows * 3);

        const auto write_row = [](const cpptorch::NavierStokesFlowField &field,
                                  double time,
                                  std::vector<double> &features,
                                  std::vector<double> &targets,
                                  size_t row) {
            const size_t offset = row * 3;
            features[offset] = field.x;
            features[offset + 1] = field.y;
            features[offset + 2] = time;
            targets[offset] = field.omega;
            targets[offset + 1] = field.u;
            targets[offset + 2] = field.v;
        };

        for (size_t row = 0; row < training_rows; ++row)
        {
            const size_t shuffled_index = training_order[row];
            const size_t sample = shuffled_index / (training_frame_count * grid_size);
            const size_t within_sample = shuffled_index % (training_frame_count * grid_size);
            const size_t frame = within_sample / grid_size;
            const size_t cell = within_sample % grid_size;
            write_row(dataset.fields[dataset.index(sample, frame, cell)],
                      dataset.times[frame],
                      training_features,
                      training_targets,
                      row);
        }

        size_t test_row = 0;
        for (size_t sample = 0; sample < dataset.sample_count; ++sample)
            for (size_t frame = training_frame_count; frame < dataset.frame_count; ++frame)
                for (size_t cell = 0; cell < grid_size; ++cell)
                    write_row(dataset.fields[dataset.index(sample, frame, cell)],
                              dataset.times[frame],
                              test_features,
                              test_targets,
                              test_row++);

        cpptorch::Matrix x_train(training_rows, 3, training_features);
        cpptorch::Matrix y_train(training_rows, 3, training_targets);
        cpptorch::Matrix x_test(test_rows, 3, test_features);
        cpptorch::Matrix y_test(test_rows, 3, test_targets);

        cpptorch::StandardScaler feature_scaler;
        feature_scaler.fit_transform(x_train);
        feature_scaler.transform(x_test);
        TargetScale target_scaler;
        target_scaler.fit_transform(y_train);
        target_scaler.transform(y_test);

        cpptorch::Model model(cpptorch::ModelType::Regression);
        model.add_layer({.n_inputs = 3, .n_neurons = 64, .activation_type = cpptorch::ActivationType::ReLU});
        model.add_layer({.n_inputs = 64, .n_neurons = 64, .activation_type = cpptorch::ActivationType::ReLU});
        model.add_layer({.n_inputs = 64, .n_neurons = 3, .activation_type = cpptorch::ActivationType::Linear});
        model.set_optimizer(
            {.optmzr_type = cpptorch::OptimizerType::Adam, .learning_rate = 0.001, .decay = 1e-5});

        std::cout << "Dataset: " << dataset_filename << "\n"
                  << "Samples: " << dataset.sample_count << ", grid: " << dataset.nx << 'x' << dataset.ny
                  << ", frames: " << dataset.frame_count << '\n'
                  << "Time-ordered split: " << training_frame_count << '/' << dataset.frame_count
                  << " training frames (" << 100.0 * static_cast<double>(training_frame_count) /
                                                   static_cast<double>(dataset.frame_count)
                  << "%), " << (dataset.frame_count - training_frame_count) << '/'
                  << dataset.frame_count << " held-out test frames ("
                  << 100.0 * static_cast<double>(dataset.frame_count - training_frame_count) /
                         static_cast<double>(dataset.frame_count)
                  << "%).\n"
                  << "Using the 2D incompressible vorticity form with kinematic viscosity=" << viscosity
                  << " and periodic spatial boundaries.\n";
        model.train(x_train, y_train, epochs, cpptorch::Matrix(), cpptorch::Matrix(), 10, 2048, 10);

        const cpptorch::Matrix test_predictions = model.predict(x_test);
        std::array<double, 3> squared_error_sum{};
        std::array<double, 3> absolute_error_sum{};
        std::array<double, 3> target_sum{};
        std::array<double, 3> target_squared_sum{};
        for (size_t row = 0; row < test_rows; ++row)
            for (size_t column = 0; column < squared_error_sum.size(); ++column)
            {
                const double prediction = target_scaler.restore(test_predictions(row, column), column);
                const double target = target_scaler.restore(y_test(row, column), column);
                const double difference = prediction - target;
                squared_error_sum[column] += difference * difference;
                absolute_error_sum[column] += std::abs(difference);
                target_sum[column] += target;
                target_squared_sum[column] += target * target;
            }
        constexpr std::array<const char *, 3> field_names = {"vorticity", "u", "v"};
        const double test_sample_count = static_cast<double>(test_rows);
        std::cout << "Held-out test metrics (R^2 is an accuracy-like regression score; it can be negative):\n";
        for (size_t column = 0; column < squared_error_sum.size(); ++column)
        {
            const double target_mean = target_sum[column] / test_sample_count;
            const double total_squared_sum =
                std::max(0.0, target_squared_sum[column] - test_sample_count * target_mean * target_mean);
            const double r_squared = total_squared_sum > 0.0
                                         ? 1.0 - squared_error_sum[column] / total_squared_sum
                                         : std::numeric_limits<double>::quiet_NaN();
            std::cout << "  " << field_names[column] << ": R^2=" << r_squared
                      << " (" << r_squared * 100.0 << "%), RMSE="
                      << std::sqrt(squared_error_sum[column] / test_sample_count)
                      << ", MAE=" << absolute_error_sum[column] / test_sample_count << '\n';
        }

        std::vector<cpptorch::NavierStokesFlowField> predicted_fields(dataset.fields.size());
        for (size_t sample = 0; sample < dataset.sample_count; ++sample)
            for (size_t frame = 0; frame < dataset.frame_count; ++frame)
            {
                std::vector<double> feature_data(grid_size * 3);
                for (size_t iy = 0; iy < dataset.ny; ++iy)
                    for (size_t ix = 0; ix < dataset.nx; ++ix)
                    {
                        const size_t cell = iy * dataset.nx + ix;
                        const size_t offset = cell * 3;
                        feature_data[offset] = dataset.x_coordinates[ix];
                        feature_data[offset + 1] = dataset.y_coordinates[iy];
                        feature_data[offset + 2] = dataset.times[frame];
                    }

                cpptorch::Matrix frame_features(grid_size, 3, feature_data);
                feature_scaler.transform(frame_features);
                const cpptorch::Matrix frame_predictions = model.predict(frame_features);
                for (size_t cell = 0; cell < grid_size; ++cell)
                {
                    auto &prediction = predicted_fields[dataset.index(sample, frame, cell)];
                    prediction.x = dataset.fields[dataset.index(sample, frame, cell)].x;
                    prediction.y = dataset.fields[dataset.index(sample, frame, cell)].y;
                    prediction.omega = target_scaler.restore(frame_predictions(cell, 0), 0);
                    prediction.u = target_scaler.restore(frame_predictions(cell, 1), 1);
                    prediction.v = target_scaler.restore(frame_predictions(cell, 2), 2);
                }
            }

        const ResidualMetrics data_residuals =
            calculate_residuals(dataset, dataset.fields, training_frame_count, viscosity);
        const ResidualMetrics model_residuals =
            calculate_residuals(dataset, predicted_fields, training_frame_count, viscosity);
        print_residuals("Reference-field", data_residuals);
        print_residuals("Neural-network", model_residuals);
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "Navier-Stokes example failed: " << error.what() << '\n';
        return 1;
    }
}
