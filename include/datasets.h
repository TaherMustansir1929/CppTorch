#pragma once

#include "matrix.h"
#include "numcpp.h"
#include <tuple>

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
};