#pragma once
#include <vector>
#include <cstddef>
#include <cstdint>

namespace sv::gmm
{
    struct GmmModel
    {
        std::size_t numGaussians = 0;
        std::size_t dim = 0;

        std::vector<double> weights; // K
        std::vector<std::vector<double>> means; // K x D
        std::vector<std::vector<double>> vars; // K x D (diagonal variances)

        [[nodiscard]] bool empty() const { return numGaussians == 0 || dim == 0; }
    };
}
