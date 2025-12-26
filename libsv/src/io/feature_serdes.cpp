#include "sv/io/feature_serdes.h"

#include <stdexcept>

using libvoicefeat::features::Feature;
using libvoicefeat::FeatureOptions;
using libvoicefeat::FeatureMatrix;
using libvoicefeat::VADFlags;
using libvoicefeat::VADState;
using libvoicefeat::CepstralType;

namespace sv::io
{
    void FeatureSerdes::writeU32(std::ofstream& out, uint32_t v)
    {
        out.write(reinterpret_cast<const char*>(&v), sizeof(v));
    }

    void FeatureSerdes::readU32(std::ifstream& in, uint32_t& v)
    {
        in.read(reinterpret_cast<char*>(&v), sizeof(v));
    }

    void FeatureSerdes::writeI32(std::ofstream& out, int32_t v)
    {
        out.write(reinterpret_cast<const char*>(&v), sizeof(v));
    }

    void FeatureSerdes::readI32(std::ifstream& in, int32_t& v)
    {
        in.read(reinterpret_cast<char*>(&v), sizeof(v));
    }

    void FeatureSerdes::writeF64(std::ofstream& out, double v)
    {
        out.write(reinterpret_cast<const char*>(&v), sizeof(v));
    }

    void FeatureSerdes::readF64(std::ifstream& in, double& v)
    {
        in.read(reinterpret_cast<char*>(&v), sizeof(v));
    }

    void FeatureSerdes::writeU8(std::ofstream& out, uint8_t v)
    {
        out.write(reinterpret_cast<const char*>(&v), sizeof(v));
    }

    void FeatureSerdes::readU8(std::ifstream& in, uint8_t& v)
    {
        in.read(reinterpret_cast<char*>(&v), sizeof(v));
    }

    bool FeatureSerdes::checkRectangular(const FeatureMatrix& m)
    {
        if (m.empty()) return true;
        const size_t cols = m[0].size();
        for (const auto& row : m)
        {
            if (row.size() != cols) return false;
        }
        return true;
    }

    void FeatureSerdes::writeFeatureOptions(std::ofstream& out, const FeatureOptions& o)
    {
        writeI32(out, o.sampleRate);
        writeI32(out, o.numFilters);
        writeI32(out, o.numCoeffs);
        writeF64(out, o.minFreq);
        writeF64(out, o.maxFreq);
        writeU8(out, static_cast<uint8_t>(o.includeEnergy ? 1 : 0));
        writeU32(out, static_cast<uint32_t>(o.filterbank));
        writeU32(out, static_cast<uint32_t>(o.melScale));
        writeU32(out, static_cast<uint32_t>(o.compressionType));
    }

    void FeatureSerdes::readFeatureOptions(std::ifstream& in, FeatureOptions& o)
    {
        int32_t i32 = 0;
        double f64 = 0.0;
        uint8_t u8 = 0;
        uint32_t u32 = 0;

        readI32(in, i32);
        o.sampleRate = i32;
        readI32(in, i32);
        o.numFilters = i32;
        readI32(in, i32);
        o.numCoeffs = i32;

        readF64(in, f64);
        o.minFreq = f64;
        readF64(in, f64);
        o.maxFreq = f64;

        readU8(in, u8);
        o.includeEnergy = (u8 != 0);

        readU32(in, u32);
        o.filterbank = static_cast<libvoicefeat::FilterbankType>(u32);
        readU32(in, u32);
        o.melScale = static_cast<libvoicefeat::MelScale>(u32);
        readU32(in, u32);
        o.compressionType = static_cast<libvoicefeat::CompressionType>(u32);
    }

    void FeatureSerdes::save(const fs::path& file, const Feature& feat) const
    {
        fs::create_directories(file.parent_path());

        std::ofstream out(file, std::ios::binary);
        if (!out) throw std::runtime_error("Cannot open for write: " + file.string());

        // header
        out.write(kMagic.data(), (std::streamsize)kMagic.size());
        writeU32(out, kVersion);

        // cepstral type
        writeU32(out, static_cast<uint32_t>(feat.getCepstralType()));

        // options
        FeatureOptions opts = feat.getOptions();
        writeFeatureOptions(out, opts);

        // matrix
        const auto& M = const_cast<Feature&>(feat).getComputedMatrix();
        if (!checkRectangular(M))
        {
            throw std::runtime_error("Non-rectangular FeatureMatrix: " + file.string());
        }

        const uint32_t rows = static_cast<uint32_t>(M.size());
        const uint32_t cols = rows ? static_cast<uint32_t>(M[0].size()) : 0;
        writeU32(out, rows);
        writeU32(out, cols);

        for (uint32_t i = 0; i < rows; ++i)
        {
            for (uint32_t j = 0; j < cols; ++j)
            {
                float v = M[i][j];
                out.write(reinterpret_cast<const char*>(&v), sizeof(v));
            }
        }

        const VADFlags& flags = feat.getVADFlags();
        writeU32(out, static_cast<uint32_t>(flags.size()));
        for (auto st : flags)
        {
            writeU8(out, static_cast<uint8_t>(st));
        }

        if (!out) throw std::runtime_error("Write failed: " + file.string());
    }

    Feature FeatureSerdes::load(const fs::path& file) const
    {
        std::ifstream in(file, std::ios::binary);
        if (!in) throw std::runtime_error("Cannot open for read: " + file.string());

        // header
        std::array<char, 8> magic{};
        in.read(magic.data(), (std::streamsize)magic.size());
        if (magic != kMagic) throw std::runtime_error("Bad magic: " + file.string());

        uint32_t version = 0;
        readU32(in, version);
        if (version != kVersion) throw std::runtime_error("Unsupported version: " + file.string());

        // cepstral type
        uint32_t ct_u32 = 0;
        readU32(in, ct_u32);
        auto ct = static_cast<CepstralType>(ct_u32);

        // options
        FeatureOptions opts{};
        readFeatureOptions(in, opts);

        // matrix
        uint32_t rows = 0, cols = 0;
        readU32(in, rows);
        readU32(in, cols);

        FeatureMatrix M;
        M.resize(rows);
        for (uint32_t i = 0; i < rows; ++i)
        {
            M[i].resize(cols);
            for (uint32_t j = 0; j < cols; ++j)
            {
                float v = 0.f;
                in.read(reinterpret_cast<char*>(&v), sizeof(v));
                M[i][j] = v;
            }
        }

        // VAD flags
        uint32_t nFlags = 0;
        readU32(in, nFlags);

        VADFlags flags;
        flags.resize(nFlags);

        for (uint32_t i = 0; i < nFlags; ++i)
        {
            uint8_t b = 0;
            readU8(in, b);
            flags[i] = static_cast<VADState>(b);
        }

        if (!in) throw std::runtime_error("Read failed: " + file.string());

        // build Feature
        Feature feat;
        feat.setCepstralType(ct);
        feat.setOptions(opts);
        feat.getComputedMatrix() = std::move(M);
        feat.setVADFlags(flags);

        return feat;
    }
}
