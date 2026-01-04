#pragma once
#include <vector>
#include <cstddef>

namespace sv::gmm
{
    struct BwStats
    {
        std::size_t K = 0;
        std::size_t D = 0;

        std::vector<double> N; // K
        std::vector<std::vector<double>> F; // K x D
        std::vector<std::vector<double>> S; // K x D

        double totalLogLikelihood = 0.0;
        std::size_t totalFrames = 0;

        BwStats() = default;
        BwStats(std::size_t k, std::size_t d) { reset(k, d); }

        void reset(std::size_t k, std::size_t d)
        {
            K = k;
            D = d;
            N.assign(K, 0.0);
            F.assign(K, std::vector<double>(D, 0.0));
            S.assign(K, std::vector<double>(D, 0.0));
            totalLogLikelihood = 0.0;
            totalFrames = 0;
        }

        void clearAccumulators()
        {
            std::fill(N.begin(), N.end(), 0.0);
            for (auto& v : F) std::fill(v.begin(), v.end(), 0.0);
            for (auto& v : S) std::fill(v.begin(), v.end(), 0.0);
            totalLogLikelihood = 0.0;
            totalFrames = 0;
        }
    };
}
