#pragma once

#include "cpptorch/core/matrix.h"
#include "cpptorch/core/numcpp.h"

namespace cpptorch
{

/**
 * @brief Inverted dropout.
 *
 * During training, each activation is zeroed with probability `rate` and the
 * surviving activations are scaled by `1 / (1 - rate)` so that no rescaling is
 * needed at inference time.
 */
class Layer_Dropout
{
public:
    /// Probability of dropping an activation, stored as `1 - rate`.
    double rate;
    /// Cached inputs, the dropout mask and the resulting output.
    Matrix inputs, output, binary_mask;
    /// Gradient of the loss with respect to the inputs.
    Matrix dinputs;

    /**
     * @brief Constructs a dropout stage.
     * @param rate The dropout probability, between 0 and 1.
     */
    explicit Layer_Dropout(double rate) : rate(1.0 - rate)
    {
        inputs = Matrix(0, 0);
        output = Matrix(0, 0);
        binary_mask = Matrix(0, 0);
        dinputs = Matrix(0, 0);
    }

    /**
     * @brief Applies the dropout mask to @p inputs.
     * @param inputs       Input matrix.
     * @param cache_inputs When true, the inputs are stored for the backward pass.
     */
    void forward(const Matrix &inputs, bool cache_inputs = true)
    {
        if (cache_inputs)
            this->inputs = inputs;
        NumCpp::binomial_into(1, rate, inputs.shape(), this->binary_mask);
        this->output.resize(inputs.rows(), inputs.cols());
#pragma omp parallel for simd if (inputs.size() > 16384)
        for (size_t i = 0; i < inputs.size(); ++i)
        {
            this->binary_mask[i] /= this->rate;
            this->output[i] = inputs[i] * this->binary_mask[i];
        }
    }

    /**
     * @brief Propagates the gradient through the dropout mask.
     * @param dvalues Gradient of the loss with respect to the output.
     */
    void backward(const Matrix &dvalues)
    {
        this->dinputs.resize(dvalues.rows(), dvalues.cols());
#pragma omp parallel for simd if (dvalues.size() > 16384)
        for (size_t i = 0; i < dvalues.size(); ++i)
            this->dinputs[i] = dvalues[i] * this->binary_mask[i];
    }
};

} // namespace cpptorch
