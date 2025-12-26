#include "sv/io/feature_serdes.h"

#include <iostream>
#include <filesystem>
#include <vector>
#include <algorithm>

#include <libvoicefeat/libvoicefeat.h>

using namespace libvoicefeat;
using namespace libvoicefeat::features;

namespace fs = std::filesystem;

using ListOfPaths = std::vector<fs::path>;

ListOfPaths getAllWavFilesFromDir(const fs::path& rootDir)
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
        if (p.extension() == ".wav") {
            paths.push_back(p);
        }
    }

    std::sort(paths.begin(), paths.end());
    return paths;
}

fs::path makeFeaturePath(const fs::path& wavPath,
                         const fs::path& timitRoot,
                         const fs::path& featuresRoot)
{
    fs::path rel = fs::relative(wavPath, timitRoot); // e.g. TRAIN/DR1/FAKS0/SX403.WAV
    fs::path out = featuresRoot / rel;               // data/features/...
    out.replace_extension(".lvf");
    return out;
}

int main()
{
    try
    {
        fs::path timitRoot    = "../../../data/timit";
        fs::path trainRoot    = timitRoot / "TRAIN";
        fs::path featuresRoot = "../../../data/features";

        ListOfPaths wavFiles = getAllWavFilesFromDir(trainRoot);
        CepstralConfig config;
        config.type = CepstralType::MFCC;
        config.delta.useDeltas = true;
        config.delta.useDeltaDeltas = true;

        CepstralExtractor extractor(config);

        sv::io::FeatureSerdes serdes;

        size_t loaded = 0;
        size_t computed = 0;

        for (const auto& wavPath : wavFiles)
        {
            fs::path featPath = makeFeaturePath(wavPath, timitRoot, featuresRoot);

            Feature feat;

            if (fs::exists(featPath))
            {
                feat = serdes.load(featPath);
                ++loaded;
            }
            else
            {
                feat = extractor.extractFromFile(wavPath);
                serdes.save(featPath, feat);
                ++computed;
            }
        }

        std::cout << "Done. Loaded: " << loaded << ", computed+saved: " << computed << "\n";
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
