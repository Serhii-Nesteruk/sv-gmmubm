#pragma once

#include "sv/gmm/gmm_model.h"
#include "sv/gmm/bw_stats.h"

namespace sv::gmm
{

    class GmmMapAdaptor
    {
    public:
        struct Options
        {
            double relevanceFactor = 16.0;
            double minOcc = 1e-3;
        };

        GmmMapAdaptor() : GmmMapAdaptor(Options()) {}
        explicit GmmMapAdaptor(Options opt);


        [[nodiscard]] GmmModel adaptMeansOnly(const GmmModel& ubm, const BwStats& spkStats) const;

    private:
        Options _opt;
    };

}
