#pragma once

#include "sv/gmm/gmm_model.h"
#include "sv/gmm/bw_stats.h"

#include <vector>

#include "libvoicefeat/config.h"

namespace sv::gmm
{
    class GmmBwStatsAccumulator
    {
    public:
        struct Options
        {
            double minWeight = 1e-12;
        };

        GmmBwStatsAccumulator() : GmmBwStatsAccumulator(Options()) {}
        explicit GmmBwStatsAccumulator(Options opt);

        void accumulate(BwStats& stats, const GmmModel& model, const libvoicefeat::FeatureMatrix& m) const;

    private:
        Options _opt;

        static double logSumExp(const std::vector<double>& v);

        [[nodiscard]] double logGaussianDiag(const std::vector<float>& x,
                               const std::vector<double>& mean,
                               const std::vector<double>& var) const;
    };
}
