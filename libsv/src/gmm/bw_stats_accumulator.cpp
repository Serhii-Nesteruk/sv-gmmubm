#include "sv/gmm/bw_stats_accumulator.h"

#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace sv::gmm
{

GmmBwStatsAccumulator::GmmBwStatsAccumulator(Options opt) : _opt(opt) {}

double GmmBwStatsAccumulator::logSumExp(const std::vector<double>& v)
{
    double m = *std::max_element(v.begin(), v.end());
    double s = 0.0;
    for (double x : v) s += std::exp(x - m);
    return m + std::log(s);
}

double GmmBwStatsAccumulator::logGaussianDiag(const std::vector<float>& x,
                                             const std::vector<double>& mean,
                                             const std::vector<double>& var) const
{
    const std::size_t D = x.size();
    double logDet = 0.0;
    double quad = 0.0;

    for (std::size_t d = 0; d < D; ++d) {
        const double vd = var[d];
        const double diff = static_cast<double>(x[d]) - mean[d];
        logDet += std::log(vd);
        quad += (diff * diff) / vd;
    }

    const double logNorm = -0.5 * (static_cast<double>(D) * std::log(2.0 * M_PI) + logDet);
    return logNorm - 0.5 * quad;
}

void GmmBwStatsAccumulator::accumulate(BwStats& stats, const GmmModel& model, const libvoicefeat::FeatureMatrix& m) const
{
    const std::size_t K = model.numGaussians;
    const std::size_t D = model.dim;

    if (stats.K != K || stats.D != D) stats.reset(K, D);

    std::vector<double> logp(K);

    for (const auto& x : m)
    {
        if (x.size() != D)
            throw std::runtime_error("BW accumulate: feature dim mismatch");

        for (std::size_t k = 0; k < K; ++k) {
            const double w = std::max(model.weights[k], _opt.minWeight);
            logp[k] = std::log(w) + logGaussianDiag(x, model.means[k], model.vars[k]);
        }

        const double logDen = logSumExp(logp);
        stats.totalLogLikelihood += logDen;
        stats.totalFrames++;

        for (std::size_t k = 0; k < K; ++k) {
            const double gamma = std::exp(logp[k] - logDen);
            stats.N[k] += gamma;

            auto& Fk = stats.F[k];
            auto& Sk = stats.S[k];

            for (std::size_t d = 0; d < D; ++d) {
                const auto xd = static_cast<double>(x[d]);
                Fk[d] += gamma * xd;
                Sk[d] += gamma * xd * xd;
            }
        }
    }
}

}
