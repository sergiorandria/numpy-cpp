/**
 * @file memristor.hpp
 * @brief Analog in-memory computing — ReRAM crossbar, Mythic/d-Matrix.
 *
 * Crossbar dot is O(1) analog V=IR via linalg::matmul, quantize uses
 * std::clamp and handles bits>=31 safely.
 */
#ifndef NP_MEMRISTOR_HPP
#define NP_MEMRISTOR_HPP

#include <algorithm>
#include <cmath>
#include <limits>

#include "api_macros.hpp"
#include "linalg.hpp"
#include "ndarray.hpp"

namespace np::analog
{

struct Crossbar
{
    ndarray<float> weights; // conductance
    Crossbar() = default;
    explicit Crossbar(ndarray<float> w) : weights(std::move(w))
    {
    }
    NP_NODISCARD ndarray<float> dot(const ndarray<float> &x) const
    {
        // O(1) analog V=IR: dot as matmul with weights^T
        // x is 1-D [N], weights is [N,M] -> use x as [N,1] then matmul
        auto xt = x.reshape({static_cast<int>(x.size()), 1});
        auto wt = weights.transpose();
        auto y = linalg::matmul(wt, xt);
        return y.reshape({static_cast<int>(y.size())});
    }
    NP_NODISCARD ndarray<float> quantize(int bits = 4) const
    {
        if (bits <= 0 || bits >= 31)
            throw std::invalid_argument("quantize: bits in [1,30]");
        ndarray<float> q(weights.shape);
        auto &qd = q.data();
        auto &wd = weights.data();
        float scale = static_cast<float>((1u << bits) - 1u);
        for (size_t i = 0; i < wd.size(); ++i)
        {
            float v = std::clamp(wd[i], -1.0f, 1.0f);
            qd[i] = std::round(v * scale) / scale;
        }
        return q;
    }
};

struct ReRAMFactory
{
    NP_NODISCARD static Crossbar crossbar(const ndarray<float> &w)
    {
        return Crossbar(w);
    }
};

} // namespace np::analog

#endif // NP_MEMRISTOR_HPP
