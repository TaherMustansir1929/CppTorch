#pragma once

#include <exception>
#include <string>
#include <tuple>

namespace cpptorch
{
/**
 * @brief Thrown when an operation receives operands with incompatible shapes.
 */
class ShapeInvalidForOperation : public std::exception
{
    std::string msg;

public:
    /**
     * @brief Constructs the exception.
     * @param optr   Name of the operation that failed.
     * @param shapeA Shape of the first operand.
     * @param shapeB Shape of the second operand.
     */
    ShapeInvalidForOperation(const std::string &optr, std::tuple<int, int> shapeA, std::tuple<int, int> shapeB)
    {
        auto [rowA, colA] = shapeA;
        auto [rowB, colB] = shapeB;

        msg = "[ShapeInvalidForOperation] Operation: \"" + optr + "\" cannot be performed on A=Matrix(" +
              std::to_string(rowA) + ", " + std::to_string(colA) + ") & B=Matrix(" + std::to_string(rowB) + ", " +
              std::to_string(colB) + ")\n";
    }

    [[nodiscard]] const char *what() const noexcept override
    {
        return msg.c_str();
    }
};

} // namespace cpptorch
