#include "sv/gmm/scorer.h"

#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace sv::gmm
{
    GmmLlrScorer::GmmLlrScorer(Options opt) : _opt(opt)
    {
    }

    double GmmLlrScorer::logSumExp(const std::vector<double>& v)
    {
        double m = *std::max_element(v.begin(), v.end());
        double s = 0.0;
        for (double x : v) s += std::exp(x - m);
        return m + std::log(s);
    }

    double GmmLlrScorer::logGaussianDiag(const std::vector<float>& x,
                                         const std::vector<double>& mean,
                                         const std::vector<double>& var)
    {
        const std::size_t D = x.size();
        double logDet = 0.0;
        double quad = 0.0;

        for (std::size_t d = 0; d < D; ++d)
        {
            const double vd = var[d];
            const double diff = static_cast<double>(x[d]) - mean[d];
            logDet += std::log(vd);
            quad += (diff * diff) / vd;
        }

        const double logNorm = -0.5 * (static_cast<double>(D) * std::log(2.0 * M_PI) + logDet);
        return logNorm - 0.5 * quad;
    }

    double GmmLlrScorer::sumLogLikelihood(const GmmModel& model, const libvoicefeat::FeatureMatrix& m) const
    {
        if (model.empty()) throw std::runtime_error("LLR: model is empty");
        const std::size_t K = model.numGaussians;
        const std::size_t D = model.dim;

        std::vector<double> logp(K);
        double sum = 0.0;

        for (const auto& x : m)
        {
            if (x.size() != D)
                throw std::runtime_error("LLR: feature dim mismatch");

            for (std::size_t k = 0; k < K; ++k)
            {
                const double w = std::max(model.weights[k], _opt.minWeight);
                logp[k] = std::log(w) + logGaussianDiag(x, model.means[k], model.vars[k]);
            }
            sum += logSumExp(logp);
        }

        return sum;
    }

    double GmmLlrScorer::avgLogLikelihood(const GmmModel& model, const libvoicefeat::FeatureMatrix& m) const
    {
        if (m.empty()) return 0.0;
        const double ll = sumLogLikelihood(model, m);
        return ll / static_cast<double>(m.size());
    }

    double GmmLlrScorer::score(const GmmModel& spk, const GmmModel& ubm, const libvoicefeat::FeatureMatrix& m) const
    {
        if (m.empty()) return 0.0;

        const double llSpk = sumLogLikelihood(spk, m);
        const double llUbm = sumLogLikelihood(ubm, m);

        if (_opt.normalizeByFrames)
        {
            const auto T = static_cast<double>(m.size());
            return (llSpk - llUbm) / T;
        }
        return (llSpk - llUbm);
    }
}
