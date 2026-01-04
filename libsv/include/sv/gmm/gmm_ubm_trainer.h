#pragma once

#include "sv/gmm/gmm_model.h"
#include "sv/gmm/bw_stats.h"

#include <filesystem>
#include <vector>
#include <random>

#include <libvoicefeat/features/feature.h>
#include "sv/io/feature_serdes.h"

namespace fs = std::filesystem;

namespace sv::gmm
{
    class GmmUbmTrainer
    {
    public:
        struct Options
        {
            std::size_t numGaussians = 64;
            std::size_t maxIterations = 10;

            double varianceFloor = 1e-2;
            double minComponentOcc = 10.0;
            double minWeight = 1e-8;

            uint32_t seed = 777;
            bool verbose = true;
        };

        GmmUbmTrainer() : GmmUbmTrainer(Options{}) {}
        explicit GmmUbmTrainer(Options opt);

        [[nodiscard]] GmmModel train(const std::vector<libvoicefeat::features::Feature>& feats);

        [[nodiscard]] GmmModel trainFromLfv(const std::vector<fs::path>& lvfFiles, const sv::io::FeatureSerdes& serdes);

    private:
        using FeatureMatrix = libvoicefeat::FeatureMatrix;
        using Feature = libvoicefeat::features::Feature;

        struct GlobalStats
        {
            std::size_t D = 0;
            std::vector<double> mean; // D
            std::vector<double> var; // D
            std::size_t frames = 0;
        };

        Options _opt;
        std::mt19937 _rng;

        GlobalStats computeGlobalStats(const std::vector<Feature>& feats);
        GlobalStats computeGlobalStatsFromLfv(const std::vector<fs::path>& lvfFiles,
                                              const sv::io::FeatureSerdes& serdes);

        void initModel(GmmModel& model,
                       const GlobalStats& gs,
                       const std::vector<Feature>& feats);

        void initModelFromLfv(GmmModel& model,
                              const GlobalStats& gs,
                              const std::vector<fs::path>& lvfFiles,
                              const sv::io::FeatureSerdes& serdes);

        void accumulateBwStats(BwStats& stats, const GmmModel& model, const FeatureMatrix& m);
        void maximize(GmmModel& model, const BwStats& stats, const GlobalStats& gs);

        [[nodiscard]] double logGaussianDiag(const std::vector<float>& x,
                               const std::vector<double>& mean,
                               const std::vector<double>& var) const;

        static double logSumExp(const std::vector<double>& v);

        void reinitComponent(GmmModel& model, std::size_t k, const GlobalStats& gs);
    };
}
