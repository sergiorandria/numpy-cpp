/**
 * @file test_statistics.cpp
 * @brief Tests for statistical functions (statistics.hpp).
 *
 * Covers median, percentile, quantile, average, ptp, cov, corrcoef,
 * histogram, bincount, digitize and the NaN-skipping family.
 */
#include "test_util.hpp"
#include <np/np.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

int main()
{
    using namespace np;
    using test::approx;
    using test::check;

    const double nan = std::numeric_limits<double>::quiet_NaN();

    // =================================================================
    // median
    // =================================================================
    {
        auto a = ndarray<double>::from_data(std::vector<int>{5}, {3.0, 1.0, 2.0, 5.0, 4.0});
        check(approx(median(a), 3.0), "median: odd length");
        auto b = ndarray<double>::from_data(std::vector<int>{6}, {3.0, 1.0, 2.0, 5.0, 4.0, 6.0});
        check(approx(median(b), 3.5), "median: even length");

        auto m = ndarray<double>::from_data(std::vector<int>{3, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0});
        auto r = median(m, 0);
        check(r.ndim() == 1 && r.size() == 3, "median axis: rank");
        check(approx(r.at(0), 4.0), "median axis: col0");
        check(approx(r.at(2), 6.0), "median axis: col2");
        auto ri = median(m, -1);
        check(approx(ri.at(0), 2.0), "median negative axis");
    }

    // =====================================================================
    // percentile / quantile
    // =====================================================================
    {
        auto a = arange<double>(10); // 0..9
        check(approx(percentile(a, 50.0), 4.5), "percentile: 50");
        check(approx(percentile(a, 0.0), 0.0), "percentile: 0");
        check(approx(percentile(a, 100.0), 9.0), "percentile: 100");
        check(approx(percentile(a, 25.0), 2.25), "percentile: linear interp");
        check(approx(quantile(a, 0.25), 2.25), "quantile: 0.25");
        bool threw = false;
        try
        {
            percentile(a, 101.0);
        }
        catch (const std::invalid_argument &)
        {
            threw = true;
        }
        check(threw, "percentile: out-of-range q throws");

        auto m = ndarray<double>::from_data(std::vector<int>{2, 4}, {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0});
        auto p = percentile(m, 50.0, 1);
        check(p.shape == std::vector<int>{2}, "percentile axis: shape");
        check(approx(p.at(0), 1.5), "percentile axis: row0");
        check(approx(p.at(1), 5.5), "percentile axis: row1");
    }

    // =====================================================================
    // average (weighted mean)
    // =====================================================================
    {
        auto a = arange<double>(1.0, 5.0); // [1,2,3,4]
        check(approx(average(a), 2.5), "average: unweighted");
        auto w = arange<double>(1.0, 5.0); // [1,2,3,4]
        const double exp = (1.0 * 1 + 2.0 * 2 + 3.0 * 3 + 4.0 * 4) / (1 + 2 + 3 + 4);
        check(approx(average(a, w), exp), "average: weighted");

        auto m = ndarray<double>::from_data(std::vector<int>{2, 2}, {1.0, 2.0, 3.0, 4.0});
        auto r = average(m, 0);
        check(r.shape == std::vector<int>{2}, "average axis: shape");
        check(approx(r.at(0), 2.0), "average axis: col0");
        check(approx(r.at(1), 3.0), "average axis: col1");
        auto aw = ndarray<double>::from_data(std::vector<int>{2}, {0.25, 0.75});
        auto r2 = average(m, 0, aw);
        check(approx(r2.at(0), 2.5), "average axis weighted: col0");
    }

    // =====================================================================
    // ptp (free + member)
    // =====================================================================
    {
        auto a = ndarray<double>::from_data(std::vector<int>{4}, {1.0, 4.0, -2.0, 9.0});
        check(ptp(a) == 11.0, "ptp: free scalar");
        check(a.ptp() == 11.0, "ptp: member scalar");

        auto m = ndarray<double>::from_data(std::vector<int>{2, 3}, {1.0, 5.0, 3.0, 0.0, 2.0, 9.0});
        auto r = ptp(m, 1);
        check(r.shape == std::vector<int>{2}, "ptp axis: shape");
        check(r.at(0) == 4.0, "ptp axis: row0");
        check(r.at(1) == 9.0, "ptp axis: row1");
        auto rk = m.ptp(0, true);
        check(rk.shape == std::vector<int>{1, 3}, "ptp keepdims: shape");
        check(rk.at(0, 0) == 1.0 && rk.at(0, 1) == 3.0 && rk.at(0, 2) == 6.0, "ptp keepdims: values");
    }

    // =====================================================================
    // cov / corrcoef
    // =====================================================================
    {
        auto x = ndarray<double>::from_data(std::vector<int>{4}, {1.0, 2.0, 3.0, 4.0});
        auto y = ndarray<double>::from_data(std::vector<int>{4}, {2.0, 4.0, 6.0, 8.0});

        auto c = cov(x, y);
        check(c.shape == std::vector<int>{2, 2}, "cov: 2x2 shape");
        check(approx(c.at(0, 0), 5.0 / 3.0), "cov: var x (ddof=1)");
        check(approx(c.at(1, 1), 20.0 / 3.0), "cov: var y (ddof=1)");
        check(approx(c.at(0, 1), 10.0 / 3.0), "cov: cov xy");

        auto cc = corrcoef(x, y);
        check(cc.shape == std::vector<int>{2, 2}, "corrcoef: 2x2 shape");
        check(approx(cc.at(0, 1), 1.0), "corrcoef: perfect correlation");

        auto x2 = ndarray<double>::from_data(std::vector<int>{2, 3}, {1.0, 2.0, 3.0, 1.0, 8.0, 3.0});
        auto c2 = cov(x2);
        check(c2.shape == std::vector<int>{2, 2}, "cov: matrix rows");
        check(approx(c2.at(0, 0), 1.0), "cov: row0 variance");

        auto single = ndarray<double>::from_data(std::vector<int>{3}, {1.0, 2.0, 3.0});
        auto s = cov(single);
        check(s.shape == std::vector<int>{1, 1}, "cov: 1-D -> 1x1");
    }

    // =====================================================================
    // var / std / nanvar / nanstd ddof rules (match NumPy: NaN, never throw)
    // =====================================================================
    {
        auto a = ndarray<double>::from_data(std::vector<int>{2}, {1.0, 2.0});
        check(approx(var(a), 0.25), "var: population default");
        check(approx(var(a, 1), 0.5), "var: ddof=1");
        check(std::isnan(var(a, 2)), "var: ddof >= n -> NaN");
        check(approx(var(a, -1), 0.5 / 3.0), "var: negative ddof allowed");
        check(std::isnan(np::std(a, 2)), "std: ddof >= n -> NaN");
        check(approx(nanvar(a, 1), 0.5), "nanvar: ddof=1");
        check(std::isnan(nanvar(a, 2)), "nanvar: ddof >= n -> NaN");
        check(std::isnan(nanstd(a, 5)), "nanstd: ddof > n -> NaN");

        auto m = ndarray<double>::from_data(std::vector<int>{2, 2}, {1.0, 2.0, 3.0, 4.0});
        auto va = var(m, 1, 1);
        check(approx(va.at(0), 0.5) && approx(va.at(1), 0.5), "var axis: ddof=1");
        auto va_big = var(m, 1, 2);
        check(std::isnan(va_big.at(0)) && std::isnan(va_big.at(1)), "var axis: ddof >= slice -> NaN");

        // cov with ddof swallowing the sample count -> NaN (NumPy agrees).
        auto one = ndarray<double>::from_data(std::vector<int>{2}, {1.0, 2.0});
        auto ck = cov(one, one, 2);
        check(std::isnan(ck.at(0, 0)), "cov: ddof >= k -> NaN");

        // corrcoef with a constant row -> NaN (zero variance, undefined).
        auto xc = ndarray<double>::from_data(std::vector<int>{2, 3}, {1.0, 1.0, 1.0, 1.0, 2.0, 3.0});
        auto ccx = corrcoef(xc);
        check(std::isnan(ccx.at(0, 0)) && std::isnan(ccx.at(0, 1)), "corrcoef: constant row -> NaN");
        check(approx(ccx.at(1, 1), 1.0), "corrcoef: varying row self -> 1");
    }

    // =====================================================================
    // histogram
    // =====================================================================
    {
        auto a = ndarray<double>::from_data(std::vector<int>{8}, {0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5});
        const double lo = 0.0, hi = 4.0;
        auto h = histogram(a, 4, std::pair<double, double>{lo, hi});
        check(h.counts.shape == std::vector<int>{4}, "histogram: bins shape");
        check(h.counts.size() == 4, "histogram: counts size");
        std::size_t total = 0;
        for (std::size_t i = 0; i < h.counts.size(); ++i)
            total += h.counts.at(i);
        check(total == a.size(), "histogram: all values counted");
        check(h.edges.size() == 5, "histogram: edges size");
        check(h.edges.at(0) == lo && h.edges.at(h.edges.size() - 1) == hi, "histogram: edge values");

        auto e = ndarray<double>::from_data(std::vector<int>{3}, {0.0, 2.0, 4.0});
        auto h2 = histogram(a, e);
        check(h2.counts.size() == 2, "histogram: edge overload bins");
    }

    // =====================================================================
    // bincount / digitize
    // =====================================================================
    {
        auto a = ndarray<int>::from_data(std::vector<int>{6}, {0, 1, 1, 1, 3, 3});
        auto b = bincount(a);
        check(b.size() == 4, "bincount: length");
        check(b.at(0) == 1.0 && b.at(1) == 3.0 && b.at(2) == 0.0 && b.at(3) == 2.0, "bincount: counts");

        auto w = ndarray<double>::from_data(std::vector<int>{6}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
        auto bw = bincount(a, w);
        check(approx(bw.at(1), 9.0), "bincount: weighted");

        auto d = digitize(a, ndarray<double>::from_data(std::vector<int>{3}, {0.0, 2.0, 4.0}));
        check(d.size() == a.size(), "digitize: same length");
        check(d.at(3) == 1 && d.at(4) == 2 && d.at(5) == 2, "digitize: indices");
    }

    // =====================================================================
    // NaN-skipping family
    // =====================================================================
    {
        auto a = ndarray<double>::from_data(std::vector<int>{6}, {1.0, nan, 3.0, nan, 5.0, 6.0});
        check(nanmin(a) == 1.0, "nanmin");
        check(nanmax(a) == 6.0, "nanmax");
        check(approx(nansum(a), 15.0), "nansum");
        check(approx(nanprod(a), 90.0), "nanprod");
        check(approx(nanmean(a), 15.0 / 4.0), "nanmean");
        check(approx(nanmedian(a), 4.0), "nanmedian");
        check(approx(nanstd(a), 1.920286436967152), "nanstd");
        check(std::isnan(nanpercentile(a, 50.0) - nanpercentile(a, 50.0)) == false &&
                  approx(nanpercentile(a, 50.0), 4.0),
              "nanpercentile");

        auto all_nan = ndarray<double>::from_data(std::vector<int>{3}, {nan, nan, nan});
        check(std::isnan(nanmean(all_nan)), "nanmean: all-NaN -> NaN");
        bool threw = false;
        try
        {
            nanmin(all_nan);
        }
        catch (const std::invalid_argument &)
        {
            threw = true;
        }
        check(threw, "nanmin: all-NaN throws");

        auto m = ndarray<double>::from_data(std::vector<int>{2, 3}, {1.0, nan, 3.0, nan, 5.0, 6.0});
        auto r = nansum(m, 1);
        check(r.shape == std::vector<int>{2}, "nansum axis: shape");
        check(approx(r.at(0), 4.0), "nansum axis: row0");
        check(approx(r.at(1), 11.0), "nansum axis: row1");
    }

    // =====================================================================
    // nanargmin / nanargmax
    // =====================================================================
    {
        auto a = ndarray<double>::from_data(std::vector<int>{6}, {3.0, nan, 1.0, nan, 5.0, 2.0});
        check(nanargmax(a) == 4, "nanargmax: flattened index");
        check(nanargmin(a) == 2, "nanargmin: flattened index");

        auto m = ndarray<double>::from_data(std::vector<int>{2, 4}, {1.0, nan, 5.0, 3.0, 8.0, 2.0, nan, 6.0});
        auto mx = nanargmax(m, 1);
        check(mx.shape == std::vector<int>{2}, "nanargmax axis: shape");
        check(mx.at(0) == 2, "nanargmax axis: row0");
        check(mx.at(1) == 0, "nanargmax axis: row1");
        auto mn = nanargmin(m, 0);
        check(mn.at(0) == 0 && mn.at(1) == 1 && mn.at(2) == 0, "nanargmin axis: values");

        auto all_nan = ndarray<double>::from_data(std::vector<int>{2}, {nan, nan});
        bool threw = false;
        try
        {
            (void)nanargmax(all_nan);
        }
        catch (const std::invalid_argument &)
        {
            threw = true;
        }
        check(threw, "nanargmax: all-NaN throws");
    }

    // =====================================================================
    // nancumsum / nancumprod
    // =====================================================================
    {
        auto a = ndarray<double>::from_data(std::vector<int>{4}, {1.0, nan, 3.0, nan});
        auto cs = nancumsum(a);
        check(cs.shape == std::vector<int>{4}, "nancumsum: 1-D shape");
        check(approx(cs.at(0), 1.0) && approx(cs.at(1), 1.0) && approx(cs.at(2), 4.0) && approx(cs.at(3), 4.0),
              "nancumsum: nan as zero");
        auto cp = nancumprod(a);
        check(approx(cp.at(0), 1.0) && approx(cp.at(1), 1.0) && approx(cp.at(2), 3.0) && approx(cp.at(3), 3.0),
              "nancumprod: nan as one");

        auto m = ndarray<double>::from_data(std::vector<int>{2, 3}, {1.0, nan, 3.0, nan, 5.0, 6.0});
        auto cs0 = nancumsum(m, 0);
        check(cs0.shape == std::vector<int>{2, 3}, "nancumsum axis: shape");
        check(approx(cs0.at(0, 0), 1.0) && approx(cs0.at(1, 0), 1.0), "nancumsum axis: col0");
        check(approx(cs0.at(0, 2), 3.0) && approx(cs0.at(1, 2), 9.0), "nancumsum axis: col2");
        auto cp1 = nancumprod(m, 1);
        check(cp1.shape == std::vector<int>{2, 3}, "nancumprod axis: shape");
        check(approx(cp1.at(0, 0), 1.0) && approx(cp1.at(0, 1), 1.0) && approx(cp1.at(0, 2), 3.0),
              "nancumprod axis: row0");
        check(approx(cp1.at(1, 0), 1.0) && approx(cp1.at(1, 1), 5.0) && approx(cp1.at(1, 2), 30.0),
              "nancumprod axis: row1");

        auto lead = ndarray<double>::from_data(std::vector<int>{2}, {nan, 2.0});
        auto lcs = nancumsum(lead);
        check(approx(lcs.at(0), 0.0) && approx(lcs.at(1), 2.0), "nancumsum: leading nan as zero");
    }

    if (test::failures() == 0)
    {
        std::printf("OK statistics\n");
        return 0;
    }
    return 1;
}