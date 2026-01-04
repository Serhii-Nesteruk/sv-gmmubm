#include "sv/gmm/gmm_model_serdes.h"

#include <stdexcept>
#include <numeric>

namespace sv::gmm
{
    void GmmModelSerdes::writeU32(std::ofstream& out, uint32_t v)
    {
        out.write(reinterpret_cast<const char*>(&v), sizeof(v));
    }

    void GmmModelSerdes::readU32(std::ifstream& in, uint32_t& v)
    {
        in.read(reinterpret_cast<char*>(&v), sizeof(v));
    }

    void GmmModelSerdes::writeU64(std::ofstream& out, uint64_t v)
    {
        out.write(reinterpret_cast<const char*>(&v), sizeof(v));
    }

    void GmmModelSerdes::readU64(std::ifstream& in, uint64_t& v)
    {
        in.read(reinterpret_cast<char*>(&v), sizeof(v));
    }

    void GmmModelSerdes::writeF64(std::ofstream& out, double v)
    {
        out.write(reinterpret_cast<const char*>(&v), sizeof(v));
    }

    void GmmModelSerdes::readF64(std::ifstream& in, double& v)
    {
        in.read(reinterpret_cast<char*>(&v), sizeof(v));
    }

    void GmmModelSerdes::ensureWritable(std::ofstream& out, const fs::path& file)
    {
        if (!out) throw std::runtime_error("Cannot open for write: " + file.string());
    }

    void GmmModelSerdes::ensureReadable(std::ifstream& in, const fs::path& file)
    {
        if (!in) throw std::runtime_error("Cannot open for read: " + file.string());
    }

    void GmmModelSerdes::validateModel(const GmmModel& model)
    {
        if (model.numGaussians == 0 || model.dim == 0)
        {
            throw std::runtime_error("GmmModel is empty");
        }
        if (model.weights.size() != model.numGaussians)
        {
            throw std::runtime_error("GmmModel weights size mismatch");
        }
        if (model.means.size() != model.numGaussians || model.vars.size() != model.numGaussians)
        {
            throw std::runtime_error("GmmModel means/vars size mismatch");
        }
        for (size_t k = 0; k < model.numGaussians; ++k)
        {
            if (model.means[k].size() != model.dim || model.vars[k].size() != model.dim)
            {
                throw std::runtime_error("GmmModel component dim mismatch");
            }
        }
    }

    void GmmModelSerdes::save(const fs::path& file, const GmmModel& model) const
    {
        validateModel(model);
        fs::create_directories(file.parent_path());

        std::ofstream out(file, std::ios::binary);
        ensureWritable(out, file);

        out.write(kMagic.data(), (std::streamsize)kMagic.size());
        writeU32(out, kVersion);

        writeU64(out, static_cast<uint64_t>(model.numGaussians));
        writeU64(out, static_cast<uint64_t>(model.dim));

        // weights
        for (double w : model.weights) writeF64(out, w);

        // means
        for (size_t k = 0; k < model.numGaussians; ++k)
        {
            for (size_t d = 0; d < model.dim; ++d)
            {
                writeF64(out, model.means[k][d]);
            }
        }

        // vars
        for (size_t k = 0; k < model.numGaussians; ++k)
        {
            for (size_t d = 0; d < model.dim; ++d)
            {
                writeF64(out, model.vars[k][d]);
            }
        }

        if (!out) throw std::runtime_error("Write failed: " + file.string());
    }

    GmmModel GmmModelSerdes::load(const fs::path& file) const
    {
        std::ifstream in(file, std::ios::binary);
        ensureReadable(in, file);

        std::array<char, 8> magic{};
        in.read(magic.data(), (std::streamsize)magic.size());
        if (magic != kMagic)
        {
            throw std::runtime_error("Bad magic: " + file.string());
        }

        uint32_t version = 0;
        readU32(in, version);
        if (version != kVersion)
        {
            throw std::runtime_error("Unsupported version: " + file.string());
        }

        uint64_t K64 = 0, D64 = 0;
        readU64(in, K64);
        readU64(in, D64);

        const size_t K = static_cast<size_t>(K64);
        const size_t D = static_cast<size_t>(D64);

        if (K == 0 || D == 0)
        {
            throw std::runtime_error("Invalid model shape in file: " + file.string());
        }

        GmmModel model;
        model.numGaussians = K;
        model.dim = D;

        model.weights.resize(K);
        model.means.assign(K, std::vector<double>(D));
        model.vars.assign(K, std::vector<double>(D));

        for (size_t k = 0; k < K; ++k) readF64(in, model.weights[k]);

        for (size_t k = 0; k < K; ++k)
        {
            for (size_t d = 0; d < D; ++d)
            {
                readF64(in, model.means[k][d]);
            }
        }

        for (size_t k = 0; k < K; ++k)
        {
            for (size_t d = 0; d < D; ++d)
            {
                readF64(in, model.vars[k][d]);
            }
        }

        if (!in) throw std::runtime_error("Read failed: " + file.string());

        validateModel(model);
        return model;
    }
}
