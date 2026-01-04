#pragma once

#include <filesystem>
#include <cstdint>
#include <array>
#include <fstream>

#include "sv/gmm/gmm_model.h"

namespace fs = std::filesystem;

namespace sv::gmm
{

    class GmmModelSerdes
    {
    public:
        GmmModelSerdes() = default;

        void save(const fs::path& file, const GmmModel& model) const;
        [[nodiscard]] GmmModel load(const fs::path& file) const;

    private:
        static constexpr uint32_t kVersion = 1;
        static constexpr std::array<char, 8> kMagic = {'S','V','G','M','M','\0','\0','\0'};

        static void writeU32(std::ofstream& out, uint32_t v);
        static void readU32(std::ifstream& in, uint32_t& v);

        static void writeU64(std::ofstream& out, uint64_t v);
        static void readU64(std::ifstream& in, uint64_t& v);

        static void writeF64(std::ofstream& out, double v);
        static void readF64(std::ifstream& in, double& v);

        static void ensureReadable(std::ifstream& in, const fs::path& file);
        static void ensureWritable(std::ofstream& out, const fs::path& file);

        static void validateModel(const GmmModel& model);
    };

}