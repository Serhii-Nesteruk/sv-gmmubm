    #pragma once

    #include <filesystem>
    #include <cstdint>
    #include <array>
    #include <vector>
    #include <fstream>

    #include <libvoicefeat/libvoicefeat.h>
    #include <libvoicefeat/features/feature.h>

    namespace fs = std::filesystem;

    namespace sv::io
    {
        class FeatureSerdes
        {
        public:
            FeatureSerdes() = default;

            void save(const fs::path& file, const libvoicefeat::features::Feature& feat) const;
            [[nodiscard]] libvoicefeat::features::Feature load(const fs::path& file) const;

        private:
            static constexpr uint32_t kVersion = 1;
            static constexpr std::array<char, 8> kMagic = {'L', 'V', 'F', 'E', 'A', 'T', '\0', '\0'};

            static void writeU32(std::ofstream& out, uint32_t v);
            static void readU32(std::ifstream& in, uint32_t& v);

            static void writeI32(std::ofstream& out, int32_t v);
            static void readI32(std::ifstream& in, int32_t& v);

            static void writeF64(std::ofstream& out, double v);
            static void readF64(std::ifstream& in, double& v);

            static void writeU8(std::ofstream& out, uint8_t v);
            static void readU8(std::ifstream& in, uint8_t& v);

            static void writeFeatureOptions(std::ofstream& out, const libvoicefeat::FeatureOptions& o);
            static void readFeatureOptions(std::ifstream& in, libvoicefeat::FeatureOptions& o);

            static bool checkRectangular(const libvoicefeat::FeatureMatrix& m);
        };
    }
