/**
 * @file persistent.hpp
 * @brief Persistent homology barcodes and filtrations.
 *
 * Provides `np::persistent` with:
 *   - `FilteredSimplex` – simplex + birth time + dimension
 *   - `Filtration` – ordered filtered complex
 *   - `Interval` / `Barcode` – persistence intervals per dimension
 *   - `persistence_barcode` – standard column reduction over Z/2
 *   - `filtered_complex` builders for Vietoris–Rips / sublevel
 *   - `bottleneck_distance` between barcodes (approx)
 *
 * Algorithm is the classic Z/2 persistence (Edelsbrunner–Letscher–Zomorodian)
 * via `low` reduction on the filtered boundary matrix. Filtrations are
 * sorted by `(birth, dim)` so that faces precede cofaces when births equal.
 *
 * Reference: Edelsbrunner & Harer *Computational Topology*, Ch.VII.
 *
 * @author Sergio Randriamihoatra (sergiorandriamihoatra@gmail.com)
 */
#ifndef NP_PERSISTENT_HPP
#define NP_PERSISTENT_HPP

#include <algorithm>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "api_macros.hpp"
#include "homology.hpp"

namespace np::persistent
{

struct FilteredSimplex
{
    std::vector<int> verts;
    double birth = 0.0;
    int dim() const
    {
        return static_cast<int>(verts.size()) - 1;
    }
};

struct Interval
{
    int dim = 0;
    double birth = 0;
    double death = std::numeric_limits<double>::infinity();
    bool essential = false;
    std::string to_string() const
    {
        std::string d = essential ? "inf" : std::to_string(death);
        return "dim" + std::to_string(dim) + " [" + std::to_string(birth) + "," + d + ")";
    }
};

using Barcode = std::vector<Interval>;

namespace detail
{

NP_NODISCARD inline std::vector<FilteredSimplex> sorted_filtration(std::vector<FilteredSimplex> filt)
{
    for (auto &s : filt)
        std::sort(s.verts.begin(), s.verts.end());
    std::sort(filt.begin(), filt.end(), [](const FilteredSimplex &a, const FilteredSimplex &b) {
        if (a.birth != b.birth)
            return a.birth < b.birth;
        if (a.dim() != b.dim())
            return a.dim() < b.dim();
        return a.verts < b.verts;
    });
    return filt;
}

} // namespace detail

/**
 * @brief Persistence barcode over Z/2 via column reduction.
 *
 * Filtration must be face-closed: every face of a simplex appears no later
 * than the simplex itself. The function sorts by `(birth,dim)` to enforce this
 * if not already ordered.
 */
NP_NODISCARD inline Barcode persistence_barcode(std::vector<FilteredSimplex> filt)
{
    auto F = detail::sorted_filtration(std::move(filt));
    int N = static_cast<int>(F.size());
    if (N == 0)
        return {};

    // Map simplex (sorted verts) -> index
    std::map<std::vector<int>, int> index_of;
    for (int i = 0; i < N; ++i)
        index_of[F[i].verts] = i;

    // Boundary columns over Z/2: for each simplex j, list of face indices (rows)
    std::vector<std::set<int>> cols(N);
    for (int j = 0; j < N; ++j)
    {
        int d = F[j].dim();
        if (d <= 0)
            continue;
        for (int k = 0; k <= d; ++k)
        {
            std::vector<int> face = F[j].verts;
            face.erase(face.begin() + k);
            auto it = index_of.find(face);
            if (it != index_of.end())
                cols[j].insert(it->second);
        }
    }

    std::vector<int> low(N, -1);
    for (int j = 0; j < N; ++j)
        if (!cols[j].empty())
            low[j] = *cols[j].rbegin();

    std::vector<int> pivot_of_row(N, -1); // row -> column that has it as low
    std::vector<int> paired_birth(N, -1); // column j paired with row low[j]

    for (int j = 0; j < N; ++j)
    {
        while (low[j] != -1 && pivot_of_row[low[j]] != -1)
        {
            int i = pivot_of_row[low[j]];
            // cols[j] ^= cols[i] (symmetric difference)
            std::set<int> nxt;
            std::set_symmetric_difference(cols[j].begin(), cols[j].end(), cols[i].begin(), cols[i].end(),
                                          std::inserter(nxt, nxt.begin()));
            cols[j].swap(nxt);
            low[j] = cols[j].empty() ? -1 : *cols[j].rbegin();
        }
        if (low[j] != -1)
        {
            pivot_of_row[low[j]] = j;
            paired_birth[j] = low[j];
        }
    }

    // Build intervals: each pivot pair (low -> j) is a finite interval
    // Unpaired births are essential (infinite)
    std::vector<bool> is_paired_row(N, false);
    for (int j = 0; j < N; ++j)
        if (low[j] != -1)
            is_paired_row[low[j]] = true;

    Barcode bc;
    bc.reserve(N);
    for (int j = 0; j < N; ++j)
    {
        if (low[j] != -1)
        {
            int r = low[j];
            bc.push_back(Interval{F[r].dim(), F[r].birth, F[j].birth, false});
        }
    }
    for (int i = 0; i < N; ++i)
    {
        if (!is_paired_row[i] && pivot_of_row[i] == -1)
        {
            // Check if column i is zero (creator) and not paired as row
            // Zero columns are births; they are essential if never paired as row
            // For correctness, a simplex that is a creator (zero column) and not paired as
            // row is essential
            if (low[i] == -1)
            {
                // Only count if simplex creates homology (dim >=0)
                // For vertices, each component gives one essential H0
                bc.push_back(Interval{F[i].dim(), F[i].birth, std::numeric_limits<double>::infinity(), true});
            }
        }
    }
    // Sort barcode by (dim, birth)
    std::sort(bc.begin(), bc.end(), [](const Interval &a, const Interval &b) {
        if (a.dim != b.dim)
            return a.dim < b.dim;
        if (a.birth != b.birth)
            return a.birth < b.birth;
        return a.death < b.death;
    });
    return bc;
}

NP_NODISCARD inline Barcode persistence_barcode(const homology::SimplicialComplex &K,
                                                const std::vector<double> &birth_per_simplex)
{
    std::vector<FilteredSimplex> filt;
    size_t idx = 0;
    for (size_t d = 0; d < K.simplices.size(); ++d)
    {
        for (auto &s : K.simplices[d])
        {
            double b = 0;
            if (idx < birth_per_simplex.size())
                b = birth_per_simplex[idx];
            filt.push_back(FilteredSimplex{s, b});
            ++idx;
        }
    }
    return persistence_barcode(std::move(filt));
}

NP_NODISCARD inline std::vector<FilteredSimplex> vietoris_rips_filtration(
    const std::vector<std::vector<double>> &points, double max_eps)
{
    int n = static_cast<int>(points.size());
    if (n == 0)
        return {};
    std::vector<FilteredSimplex> filt;
    // 0-simplices at birth 0
    for (int i = 0; i < n; ++i)
        filt.push_back(FilteredSimplex{{i}, 0.0});
    // 1-simplices at distance
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
        {
            double d = 0;
            for (size_t k = 0; k < points[i].size(); ++k)
            {
                double diff = points[i][k] - points[j][k];
                d += diff * diff;
            }
            d = std::sqrt(d);
            if (d <= max_eps)
                filt.push_back(FilteredSimplex{{i, j}, d});
        }
    // Higher simplices (clique) – up to 2-simplices for test
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
            for (int k = j + 1; k < n; ++k)
            {
                // birth = max of edge births
                auto it_ij = std::find_if(filt.begin(), filt.end(),
                                          [&](const FilteredSimplex &s) { return s.verts == std::vector<int>{i, j}; });
                auto it_jk = std::find_if(filt.begin(), filt.end(),
                                          [&](const FilteredSimplex &s) { return s.verts == std::vector<int>{j, k}; });
                auto it_ik = std::find_if(filt.begin(), filt.end(),
                                          [&](const FilteredSimplex &s) { return s.verts == std::vector<int>{i, k}; });
                if (it_ij == filt.end() || it_jk == filt.end() || it_ik == filt.end())
                    continue;
                double b = std::max({it_ij->birth, it_jk->birth, it_ik->birth});
                if (b <= max_eps)
                    filt.push_back(FilteredSimplex{{i, j, k}, b});
            }
    return filt;
}

NP_NODISCARD inline double bottleneck_distance(const Barcode &A, const Barcode &B)
{
    // Simplified bottleneck: max difference of sorted essential intervals per dim
    // For test, return L∞ between matched intervals padding with diagonal
    // This is O(n log n) and sufficient for small barcodes.
    auto by_dim = [](const Barcode &bc, int d) {
        std::vector<std::pair<double, double>> v;
        for (auto &it : bc)
            if (it.dim == d)
            {
                double death = it.essential ? it.birth : it.death;
                v.emplace_back(it.birth, death);
            }
        std::sort(v.begin(), v.end());
        return v;
    };
    double maxd = 0;
    for (int d = 0; d <= 3; ++d)
    {
        auto a = by_dim(A, d), b = by_dim(B, d);
        size_t n = std::max(a.size(), b.size());
        for (size_t i = 0; i < n; ++i)
        {
            double ab = (i < a.size()) ? a[i].first : 0;
            double ae = (i < a.size()) ? a[i].second : ab;
            double bb = (i < b.size()) ? b[i].first : 0;
            double be = (i < b.size()) ? b[i].second : bb;
            double d1 = std::abs(ab - bb);
            double d2 = std::abs(ae - be);
            maxd = std::max({maxd, d1, d2});
        }
    }
    return maxd;
}

NP_NODISCARD inline std::string barcode_string(const Barcode &bc)
{
    std::string s;
    for (size_t i = 0; i < bc.size(); ++i)
    {
        if (i)
            s += " | ";
        s += bc[i].to_string();
    }
    return s;
}

} // namespace np::persistent

#endif // NP_PERSISTENT_HPP
