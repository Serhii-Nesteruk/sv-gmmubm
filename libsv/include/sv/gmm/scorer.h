#pragma once

#include <libvoicefeat/types.h>
#include <libvoicefeat/config.h>

#include "sv/gmm/gmm_model.h"

namespace sv::gmm
{
    class GmmLlrScorer
    {
    public:
        struct Options
        {
            double minWeight = 1e-12;
            bool normalizeByFrames = true;
        };

        GmmLlrScorer() : GmmLlrScorer(Options())
        {
        };
        explicit GmmLlrScorer(Options opt);

        [[nodiscard]] double score(const GmmModel& spk, const GmmModel& ubm,
                                   const libvoicefeat::FeatureMatrix& m) const;

        [[nodiscard]] double avgLogLikelihood(const GmmModel& model, const libvoicefeat::FeatureMatrix& m) const;

    private:
        Options _opt;

        static double logSumExp(const std::vector<double>& v);

        [[nodiscard]] static double logGaussianDiag(const std::vector<float>& x,
                                             const std::vector<double>& mean,
                                             const std::vector<double>& var) ;

        [[nodiscard]] double sumLogLikelihood(const GmmModel& model, const libvoicefeat::FeatureMatrix& m) const;
    };
}
