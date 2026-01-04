#include "sv/gmm/map_adaptor.h"

#include <stdexcept>
#include <algorithm>

namespace sv::gmm
{
    GmmMapAdaptor::GmmMapAdaptor(Options opt) : _opt(opt)
    {
    }

    GmmModel GmmMapAdaptor::adaptMeansOnly(const GmmModel& ubm, const BwStats& s) const
    {
        if (ubm.empty()) throw std::runtime_error("MAP: UBM is empty");
        if (s.K != ubm.numGaussians || s.D != ubm.dim)
            throw std::runtime_error("MAP: stats shape mismatch");

        GmmModel out = ubm; 

        const std::size_t K = ubm.numGaussians;
        const std::size_t D = ubm.dim;
        const double r = _opt.relevanceFactor;

        for (std::size_t k = 0; k < K; ++k)
        {
            const double Nk = s.N[k];
            if (Nk <= _opt.minOcc)
            {
                out.means[k] = ubm.means[k];
                continue;
            }

            const double alpha = Nk / (Nk + r);
            for (std::size_t d = 0; d < D; ++d)
            {
                const double mlMean = s.F[k][d] / Nk;
                out.means[k][d] = alpha * mlMean + (1.0 - alpha) * ubm.means[k][d];
            }
        }

        return out;
    }
} 
