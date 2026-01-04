#include <algorithm>
#include <filesystem>
#include <iostream>
#include <list>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>
#include <numeric>

#include "sv/gmm/gmm_model.h"
#include "sv/gmm/gmm_model_serdes.h"
#include "sv/gmm/bw_stats_accumulator.h"
#include "sv/gmm/map_adaptor.h"
#include "sv/gmm/scorer.h"
#include "sv/io/feature_serdes.h"

namespace fs = std::filesystem;

using namespace sv::gmm;
using namespace sv::io;

using Paths = std::vector<fs::path>;
using SpeakerModelsMap = std::unordered_map<std::string, GmmModel>;
using diff_t = std::vector<fs::path>::difference_type;

struct Scores
{
    std::vector<double> genuineScores{};
    std::vector<double> impostorScores{};
} scores;

struct SpeakerData
{
    std::string id{};
    fs::path dir{};
    Paths lvfFiles{};
    Paths enroll{};
    Paths test{};
};

static bool hasLvfFiles(const fs::path& dir)
{
    for (const auto& e : fs::directory_iterator(dir))
    {
        if (!e.is_regular_file()) continue;
        if (e.path().extension() == ".lvf") return true;
    }
    return false;
}

static Paths listLvfFilesSorted(const fs::path& dir)
{
    Paths out;
    for (const auto& e : fs::directory_iterator(dir))
    {
        if (!e.is_regular_file()) continue;
        auto& p = e.path();
        if (p.extension() == ".lvf") out.push_back(p);
    }
    std::sort(out.begin(), out.end(),
              [](const fs::path& a, const fs::path& b)
              {
                  return a.filename().string() < b.filename().string();
              });
    return out;
}

static std::vector<SpeakerData> collectSpeakers(const fs::path& root)
{
    std::vector<SpeakerData> speakers;

    if (!fs::exists(root) || !fs::is_directory(root))
    {
        throw std::runtime_error("Invalid root: " + root.string());
    }

    for (const auto& e : fs::recursive_directory_iterator(root))
    {
        if (!e.is_directory()) continue;

        const fs::path& dir = e.path();
        if (!hasLvfFiles(dir)) continue;

        SpeakerData s;
        s.id = dir.filename().string();
        s.dir = dir;
        s.lvfFiles = listLvfFilesSorted(dir);

        if (!s.lvfFiles.empty())
            speakers.push_back(std::move(s));
    }

   std::sort(speakers.begin(), speakers.end(),
              [](const SpeakerData& a, const SpeakerData& b) { return a.dir < b.dir; });

    return speakers;
}

static double median(std::vector<double> v)
{
    if (v.empty()) return 0.0;
    const size_t n = v.size();
    std::nth_element(v.begin(), v.begin() + n / 2, v.end());
    double med = v[n / 2];
    if (n % 2 == 0)
    {
        auto it = std::max_element(v.begin(), v.begin() + n / 2);
        med = (*it + med) * 0.5;
    }
    return med;
}

static double mean(const std::vector<double>& v)
{
    if (v.empty()) return 0.0;
    return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
}

void removeSpeakersWithBadAmountOfFiles(std::vector<SpeakerData>& speakers, int minAmountOfFiles)
{
    speakers.erase(std::remove_if(speakers.begin(), speakers.end(),
                                  [&](const SpeakerData& s)
                                  {
                                      return s.lvfFiles.size() < (minAmountOfFiles);
                                  }),
                   speakers.end());
}

void splitEnrollTest(std::vector<SpeakerData>& speakers, const size_t enrollN, const size_t testM)
{
    for (auto& s : speakers)
    {
        s.enroll.assign(s.lvfFiles.begin(), std::next(s.lvfFiles.begin(), static_cast<diff_t>(enrollN)));
        s.test.assign(std::prev(s.lvfFiles.end(), static_cast<diff_t>(testM)), s.lvfFiles.end());
    }
}

SpeakerModelsMap buildSpeakerModels(std::vector<SpeakerData>& speakers, GmmBwStatsAccumulator& acc,
    FeatureSerdes& featureSerdes, GmmModel& ubm, GmmMapAdaptor& adaptor)
{
    SpeakerModelsMap spkModels;
    spkModels.reserve(speakers.size());

    for (const auto& s : speakers)
    {
        BwStats stats(ubm.numGaussians, ubm.dim);

        for (const auto& f : s.enroll)
        {
            auto feat = featureSerdes.load(f);
            acc.accumulate(stats, ubm, feat.getComputedMatrix());
        }

        auto spkModel = adaptor.adaptMeansOnly(ubm, stats);
        spkModels.emplace(s.id, std::move(spkModel));
    }

    return spkModels;
}

int main()
{
    try
    {
        const fs::path root = "../../../data/features/TEST";
        constexpr size_t targetSpeakers = 30;
        constexpr size_t enrollN = 5;
        constexpr size_t testM = 2;
        constexpr size_t impostorPerSpeaker = 5;

        const uint32_t seed = 777;

        GmmModelSerdes modelSerdes;
        GmmModel ubm = modelSerdes.load("../../../data/models/ubm.bin");

        FeatureSerdes featureSerdes;
        GmmBwStatsAccumulator acc;
        GmmMapAdaptor adaptor({.relevanceFactor = 16.0, .minOcc = 1e-3});
        GmmLlrScorer scorer;

        auto speakers = collectSpeakers(root);

        removeSpeakersWithBadAmountOfFiles(speakers, enrollN + testM);

        if (speakers.size() < targetSpeakers)
        {
            std::cout << "[WARN] Available speakers=" << speakers.size()
                << " < targetSpeakers=" << targetSpeakers << std::endl;
        }

        std::mt19937 rng(seed);
        std::shuffle(speakers.begin(), speakers.end(), rng);
        if (speakers.size() > targetSpeakers) speakers.resize(targetSpeakers);

        splitEnrollTest(speakers, enrollN, testM);

        SpeakerModelsMap spkModels = buildSpeakerModels(speakers, acc, featureSerdes, ubm, adaptor);

        // Genuine
        for (const auto& s : speakers)
        {
            const auto& model = spkModels.at(s.id);

            for (const auto& testFile : s.test)
            {
                auto feat = featureSerdes.load(testFile);
                const double sc = scorer.score(model, ubm, feat.getComputedMatrix());

                scores.genuineScores.push_back(sc);

                std::cout << "Genuine " << s.id << " VS " << s.id
                    << " (" << testFile.filename().string() << "): "
                    << sc << "\n";
            }
        }

        // Impostor
        std::uniform_int_distribution<size_t> spkDist(0, speakers.size() - 1);

        for (const auto& s : speakers)
        {
            const auto& model = spkModels.at(s.id);

            size_t added = 0;
            while (added < impostorPerSpeaker)
            {
                size_t j = spkDist(rng);
                if (speakers[j].id == s.id) continue;

                const auto& other = speakers[j];
                std::uniform_int_distribution<size_t> tfDist(0, other.test.size() - 1);
                const auto& testFile = other.test[tfDist(rng)];

                auto feat = featureSerdes.load(testFile);
                const double sc = scorer.score(model, ubm, feat.getComputedMatrix());

                scores.impostorScores.push_back(sc);

                std::cout << "Impostor " << s.id << " VS " << other.id
                    << " (" << testFile.filename().string() << "): "
                    << sc << "\n";

                ++added;
            }
        }

        // threshold
        const double medG = median(scores.genuineScores);
        const double medI = median(scores.impostorScores);
        const double thr = 0.5 * (medG + medI);

        std::cout << "\n=== Summary ===\n";
        std::cout << "speakers=" << speakers.size()
            << " enrollN=" << enrollN
            << " testM=" << testM
            << " impostorPerSpeaker=" << impostorPerSpeaker << "\n";

        std::cout << "genuine:  n=" << scores.genuineScores.size()
            << " mean=" << mean(scores.genuineScores)
            << " median=" << medG << "\n";

        std::cout << "impostor: n=" << scores.impostorScores.size()
            << " mean=" << mean(scores.impostorScores)
            << " median=" << medI << "\n";

        std::cout << "thr (midpoint of medians) = " << thr << "\n";

        size_t FR = 0;
        for (double s : scores.genuineScores) if (s < thr) ++FR;

        size_t FA = 0;
        for (double s : scores.impostorScores) if (s >= thr) ++FA;

        const double FRR = scores.genuineScores.empty()
                               ? 0.0
                               : static_cast<double>(FR) / static_cast<double>(scores.genuineScores.size());
        const double FAR = scores.impostorScores.empty()
                               ? 0.0
                               : static_cast<double>(FA) / static_cast<double>(scores.impostorScores.size());

        std::cout << "FRR (genuine rejected) = " << FRR << "\n";
        std::cout << "FAR (impostor accepted) = " << FAR << "\n";

        // sanity
        if (!speakers.empty())
        {
            auto feat = featureSerdes.load(speakers.front().test.front());
            double ubmVsUbm = scorer.score(ubm, ubm, feat.getComputedMatrix());
            std::cout << "UBM vs UBM (sanity) = " << ubmVsUbm << "\n";
        }

        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
