#include <algorithm>
#include <iostream>

#include <libvoicefeat/libvoicefeat.h>
#include <filesystem>

#include "sv/gmm/bw_stats_accumulator.h"
#include "sv/gmm/gmm_model.h"
#include "sv/gmm/gmm_model_serdes.h"
#include "sv/gmm/map_adaptor.h"
#include "sv/io/feature_serdes.h"

using namespace libvoicefeat;
using namespace sv::gmm;

using ListOfPaths = std::vector<fs::path>;

ListOfPaths getAllLvfFilesFromDir(const fs::path& rootDir) // TODO: DRY
{
    if (!fs::exists(rootDir) || !fs::is_directory(rootDir)) {
        throw std::runtime_error("Invalid directory: " + rootDir.string());
    }

    ListOfPaths paths;
    for (const auto& entry : fs::recursive_directory_iterator(
             rootDir, fs::directory_options::skip_permission_denied))
    {
        if (!entry.is_regular_file()) continue;

        const fs::path& p = entry.path();
        if (p.extension() == ".lvf") {
            paths.push_back(p);
        }
    }

    std::sort(paths.begin(), paths.end());
    return paths;
}

int main()
{
    auto spkLfvFiles = getAllLvfFilesFromDir("../../../data/features/TEST/DR1/FAKS0");
    GmmModelSerdes ubmSerdes;
    sv::io::FeatureSerdes featureSerdes;
    GmmModel ubm = ubmSerdes.load("../../../data/models/ubm.bin");
    GmmBwStatsAccumulator acc;
    BwStats stats(ubm.numGaussians, ubm.dim);

    for (auto& lvf : spkLfvFiles) {
        auto feat = featureSerdes.load(lvf);
        acc.accumulate(stats, ubm, feat.getComputedMatrix());
    }

    GmmMapAdaptor adaptor({ .relevanceFactor = 16.0 });
    GmmModel spkModel = adaptor.adaptMeansOnly(ubm, stats);

    GmmModelSerdes spkSerdes;

    spkSerdes.save("../../../data/models/spk_FAKS0.bin", spkModel);

    return 0;
}