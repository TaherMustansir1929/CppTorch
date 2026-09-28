#pragma once

#include <exception>
#include <string>
#include <tuple>
class ShapeInvalidForOperation : public std::exception
{
    std::string msg;

public:
    ShapeInvalidForOperation(const std::string &optr, std::tuple<int, int> shapeA, std::tuple<int, int> shapeB)
    {
        auto [rowA, colA] = shapeA;
        auto [rowB, colB] = shapeB;

        msg = "[ShapeInvalidForOperation] Operation: \"" + optr + "\" cannot be performed on A=Matrix(" +
              std::to_string(rowA) + ", " + std::to_string(colA) + ") & B=Matrix(" + std::to_string(rowA) + ", " +
              std::to_string(colA) + ")\n";
    }

    [[nodiscard]] const char *what() const noexcept override
    {
        return msg.c_str();
    }
};