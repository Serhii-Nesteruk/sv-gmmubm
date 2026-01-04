#include <algorithm>
#include <iostream>

#include "sv/gmm/gmm_ubm_trainer.h"
#include "sv/gmm/gmm_model_serdes.h"

#include <cmath>
#include <limits>

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
     fs::path featuresRoot    = "../../../data/features";
     fs::path trainRoot    = featuresRoot / "TRAIN";
     const auto lvfFiles = getAllLvfFilesFromDir(trainRoot);

     sv::io::FeatureSerdes featSerdes;

     sv::gmm::GmmUbmTrainer::Options options;
     options.numGaussians = 128;
     options.maxIterations = 15;
     options.verbose = true;

     sv::gmm::GmmUbmTrainer trainer(options);

     auto ubm = trainer.trainFromLfv(lvfFiles, featSerdes);

     sv::gmm::GmmModelSerdes modelSerdes;
     modelSerdes.save("../../../data/models/ubm.bin", ubm);

    return 0;
}