#include "sv/gmm/gmm_ubm_trainer.h"

#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <iostream>

namespace sv::gmm
{

GmmUbmTrainer::GmmUbmTrainer(Options opt)
    : _opt(opt), _rng(opt.seed)
{
}

double GmmUbmTrainer::logSumExp(const std::vector<double>& v)
{
    double m = *std::max_element(v.begin(), v.end());
    double s = 0.0;
    for (double x : v) s += std::exp(x - m);
    return m + std::log(s);
}

double GmmUbmTrainer::logGaussianDiag(const std::vector<float>& x,
                                     const std::vector<double>& mean,
                                     const std::vector<double>& var) const
{
    const std::size_t D = x.size();

    double logDet = 0.0;
    double quad = 0.0;
    for (std::size_t d = 0; d < D; ++d) {
        const double vd = var[d];
        const double diff = static_cast<double>(x[d]) - mean[d];
        logDet += std::log(vd);
        quad += (diff * diff) / vd;
    }

    const double logNorm = -0.5 * (static_cast<double>(D) * std::log(2.0 * M_PI) + logDet);
    return logNorm - 0.5 * quad;
}

void GmmUbmTrainer::accumulateBwStats(BwStats& stats, const GmmModel& model, const FeatureMatrix& m)
{
    const std::size_t K = model.numGaussians;
    const std::size_t D = model.dim;

    std::vector<double> logp(K);

    for (const auto& x : m)
    {
        // TODO: VAD
        // Feature.getVADFlags()[frameIdx] == Speech

        if (x.size() != D) {
            throw std::runtime_error("Feature dim mismatch while accumulating BW stats");
        }

        for (std::size_t k = 0; k < K; ++k) {
            const double lw = std::log(std::max(model.weights[k], _opt.minWeight));
            logp[k] = lw + logGaussianDiag(x, model.means[k], model.vars[k]);
        }

        const double logDen = logSumExp(logp);
        stats.totalLogLikelihood += logDen;
        stats.totalFrames++;

        for (std::size_t k = 0; k < K; ++k) {
            const double gamma = std::exp(logp[k] - logDen);
            stats.N[k] += gamma;

            auto& Fk = stats.F[k];
            auto& Sk = stats.S[k];

            for (std::size_t d = 0; d < D; ++d) {
                const double xd = static_cast<double>(x[d]);
                Fk[d] += gamma * xd;
                Sk[d] += gamma * xd * xd;
            }
        }
    }
}

void GmmUbmTrainer::reinitComponent(GmmModel& model, std::size_t k, const GlobalStats& gs)
{
    std::normal_distribution<double> nd(0.0, 1.0);

    model.weights[k] = 1.0 / static_cast<double>(model.numGaussians);

    for (std::size_t d = 0; d < model.dim; ++d) {
        const double sigma = std::sqrt(std::max(gs.var[d], 1e-12));
        model.means[k][d] = gs.mean[d] + 0.1 * sigma * nd(_rng);
        model.vars[k][d] = std::max(gs.var[d], 1e-12);
    }
}

void GmmUbmTrainer::maximize(GmmModel& model, const BwStats& stats, const GlobalStats& gs)
{
    const std::size_t K = model.numGaussians;
    const std::size_t D = model.dim;

    const double T = static_cast<double>(stats.totalFrames);
    if (T <= 0.0) throw std::runtime_error("No frames in BW stats");

    // weights
    for (std::size_t k = 0; k < K; ++k) {
        model.weights[k] = std::max(stats.N[k] / T, _opt.minWeight);
    }

    // renormalize weights
    double wsum = 0.0;
    for (double w : model.weights) wsum += w;
    for (double& w : model.weights) w /= wsum;

    // means + variances
    for (std::size_t k = 0; k < K; ++k)
    {
        const double Nk = stats.N[k];

        if (Nk < _opt.minComponentOcc) {
            reinitComponent(model, k, gs);
            continue;
        }

        for (std::size_t d = 0; d < D; ++d)
        {
            const double mean = stats.F[k][d] / Nk;
            const double ex2  = stats.S[k][d] / Nk;
            double var = ex2 - mean * mean;

            const double floorVar = _opt.varianceFloor * std::max(gs.var[d], 1e-12);
            if (var < floorVar) var = floorVar;

            model.means[k][d] = mean;
            model.vars[k][d] = var;
        }
    }
}

GmmUbmTrainer::GlobalStats GmmUbmTrainer::computeGlobalStats(const std::vector<Feature>& feats)
{
    GlobalStats gs;

    bool dimSet = false;
    for (const auto& f : feats) {
        const auto& m = const_cast<Feature&>(f).getComputedMatrix();
        if (m.empty()) continue;
        gs.D = m[0].size();
        dimSet = true;
        break;
    }
    if (!dimSet) return gs;

    gs.mean.assign(gs.D, 0.0);
    gs.var.assign(gs.D, 0.0);

    // mean
    for (const auto& f : feats) {
        const auto& m = const_cast<Feature&>(f).getComputedMatrix();
        for (const auto& x : m) {
            gs.frames++;
            for (std::size_t d = 0; d < gs.D; ++d) gs.mean[d] += x[d];
        }
    }
    if (gs.frames == 0) return gs;

    for (double& v : gs.mean) v /= static_cast<double>(gs.frames);

    // var
    for (const auto& f : feats) {
        const auto& m = const_cast<Feature&>(f).getComputedMatrix();
        for (const auto& x : m) {
            for (std::size_t d = 0; d < gs.D; ++d) {
                const double diff = static_cast<double>(x[d]) - gs.mean[d];
                gs.var[d] += diff * diff;
            }
        }
    }
    for (double& v : gs.var) v /= static_cast<double>(gs.frames);

    return gs;
}

GmmUbmTrainer::GlobalStats GmmUbmTrainer::computeGlobalStatsFromLfv(const std::vector<fs::path>& lvfFiles,
                                                                   const sv::io::FeatureSerdes& serdes)
{
    GlobalStats gs;

    // find D
    for (const auto& p : lvfFiles) {
        auto f = serdes.load(p);
        const auto& m = f.getComputedMatrix();
        if (!m.empty()) { gs.D = m[0].size(); break; }
    }
    if (gs.D == 0) return gs;

    gs.mean.assign(gs.D, 0.0);
    gs.var.assign(gs.D, 0.0);

    // mean
    for (const auto& p : lvfFiles) {
        auto f = serdes.load(p);
        const auto& m = f.getComputedMatrix();
        for (const auto& x : m) {
            gs.frames++;
            for (std::size_t d = 0; d < gs.D; ++d) gs.mean[d] += x[d];
        }
    }
    if (gs.frames == 0) return gs;
    for (double& v : gs.mean) v /= static_cast<double>(gs.frames);

    // var
    for (const auto& p : lvfFiles) {
        auto f = serdes.load(p);
        const auto& m = f.getComputedMatrix();
        for (const auto& x : m) {
            for (std::size_t d = 0; d < gs.D; ++d) {
                const double diff = static_cast<double>(x[d]) - gs.mean[d];
                gs.var[d] += diff * diff;
            }
        }
    }
    for (double& v : gs.var) v /= static_cast<double>(gs.frames);

    return gs;
}

void GmmUbmTrainer::initModel(GmmModel& model, const GlobalStats& gs, const std::vector<Feature>& feats)
{
    model.numGaussians = _opt.numGaussians;
    model.dim = gs.D;

    const std::size_t K = model.numGaussians;
    const std::size_t D = model.dim;

    model.weights.assign(K, 1.0 / static_cast<double>(K));
    model.means.assign(K, std::vector<double>(D, 0.0));
    model.vars.assign(K, std::vector<double>(D, 0.0));

    for (std::size_t k = 0; k < K; ++k) {
        for (std::size_t d = 0; d < D; ++d) {
            model.vars[k][d] = std::max(gs.var[d], 1e-12);
        }
    }

    std::vector<std::vector<double>> picked;
    picked.reserve(K);

    std::size_t seen = 0;
    for (const auto& f : feats) {
        const auto& m = const_cast<Feature&>(f).getComputedMatrix();
        for (const auto& x : m) {
            ++seen;
            if (picked.size() < K) {
                picked.emplace_back(x.begin(), x.end());
            } else {
                std::uniform_int_distribution<std::size_t> ud(0, seen - 1);
                const std::size_t j = ud(_rng);
                if (j < K) {
                    picked[j].assign(x.begin(), x.end());
                }
            }
        }
    }

    if (picked.size() < K) {
        for (std::size_t k = 0; k < K; ++k) reinitComponent(model, k, gs);
        return;
    }

    for (std::size_t k = 0; k < K; ++k) {
        for (std::size_t d = 0; d < D; ++d) model.means[k][d] = picked[k][d];
    }
}

void GmmUbmTrainer::initModelFromLfv(GmmModel& model,
                                    const GlobalStats& gs,
                                    const std::vector<fs::path>& lvfFiles,
                                    const sv::io::FeatureSerdes& serdes)
{
    model.numGaussians = _opt.numGaussians;
    model.dim = gs.D;

    const std::size_t K = model.numGaussians;
    const std::size_t D = model.dim;

    model.weights.assign(K, 1.0 / static_cast<double>(K));
    model.means.assign(K, std::vector<double>(D, 0.0));
    model.vars.assign(K, std::vector<double>(D, 0.0));

    for (std::size_t k = 0; k < K; ++k) {
        for (std::size_t d = 0; d < D; ++d) model.vars[k][d] = std::max(gs.var[d], 1e-12);
    }

    std::vector<std::vector<double>> picked;
    picked.reserve(K);

    std::size_t seen = 0;
    for (const auto& p : lvfFiles)
    {
        auto f = serdes.load(p);
        const auto& m = f.getComputedMatrix();
        for (const auto& x : m) {
            ++seen;
            if (picked.size() < K) {
                picked.emplace_back(x.begin(), x.end());
            } else {
                std::uniform_int_distribution<std::size_t> ud(0, seen - 1);
                const std::size_t j = ud(_rng);
                if (j < K) {
                    picked[j].assign(x.begin(), x.end());
                }
            }
        }
    }

    if (picked.size() < K) {
        for (std::size_t k = 0; k < K; ++k) reinitComponent(model, k, gs);
        return;
    }

    for (std::size_t k = 0; k < K; ++k) {
        for (std::size_t d = 0; d < D; ++d) model.means[k][d] = picked[k][d];
    }
}

GmmModel GmmUbmTrainer::train(const std::vector<Feature>& feats)
{
    const auto gs = computeGlobalStats(feats);
    if (gs.frames == 0 || gs.D == 0) throw std::runtime_error("No frames to train UBM");

    GmmModel model;
    initModel(model, gs, feats);

    BwStats stats(model.numGaussians, model.dim);

    double prevAvgLL = -1e100;

    for (std::size_t it = 0; it < _opt.maxIterations; ++it)
    {
        stats.clearAccumulators();

        for (const auto& f : feats) {
            const auto& m = const_cast<Feature&>(f).getComputedMatrix();
            accumulateBwStats(stats, model, m);
        }

        const double avgLL = stats.totalLogLikelihood / std::max<std::size_t>(1, stats.totalFrames);

        if (_opt.verbose) {
            std::cout << "[UBM] iter " << it
                      << " frames=" << stats.totalFrames
                      << " avgLL=" << avgLL << "\n";
        }

        maximize(model, stats, gs);

        if (it > 0 && std::abs(avgLL - prevAvgLL) < 1e-4) {
            if (_opt.verbose) std::cout << "[UBM] converged.\n";
            break;
        }
        prevAvgLL = avgLL;
    }

    return model;
}

GmmModel GmmUbmTrainer::trainFromLfv(const std::vector<fs::path>& lvfFiles, const sv::io::FeatureSerdes& serdes)
{
    const auto gs = computeGlobalStatsFromLfv(lvfFiles, serdes);
    if (gs.frames == 0 || gs.D == 0) throw std::runtime_error("No frames to train UBM");

    GmmModel model;
    initModelFromLfv(model, gs, lvfFiles, serdes);

    BwStats stats(model.numGaussians, model.dim);

    double prevAvgLL = -1e100;

    for (std::size_t it = 0; it < _opt.maxIterations; ++it)
    {
        stats.clearAccumulators();

        for (const auto& p : lvfFiles) {
            auto f = serdes.load(p);
            const auto& m = f.getComputedMatrix();
            accumulateBwStats(stats, model, m);
        }

        const double avgLL = stats.totalLogLikelihood / std::max<std::size_t>(1, stats.totalFrames);

        if (_opt.verbose) {
            std::cout << "[UBM] iter " << it
                      << " files=" << lvfFiles.size()
                      << " frames=" << stats.totalFrames
                      << " avgLL=" << avgLL << "\n";
        }

        maximize(model, stats, gs);

        if (it > 0 && std::abs(avgLL - prevAvgLL) < 1e-4) {
            if (_opt.verbose) std::cout << "[UBM] converged.\n";
            break;
        }
        prevAvgLL = avgLL;
    }

    return model;
}

}

