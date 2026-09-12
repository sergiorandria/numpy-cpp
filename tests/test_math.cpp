/**
 * @file test_math.cpp
 * @brief Tests for element-wise mathematical functions (math.hpp).
 *
 * Verifies trigonometric, hyperbolic, exponential, rounding, and arithmetic
 * functions against known values.
 */
#include "test_util.hpp"
#include <np/np.hpp>

#include <cmath>
#include <complex>
#include <limits>
#include <numbers>
#include <stdexcept>

int main()
{
    using namespace np;

    // --- Trigonometric functions ---
    {
        auto x = arange(-std::numbers::pi, std::numbers::pi, 0.5);
        auto s = sin(x);
        auto c = cos(x);
        auto t = tan(x);

        test::check(s.shape == x.shape, "sin: shape preserved");
        test::check(c.shape == x.shape, "cos: shape preserved");
        test::check(test::approx(s.at(0), std::sin(-std::numbers::pi)), "sin value");
        test::check(test::approx(c.at(0), std::cos(-std::numbers::pi)), "cos value");
    }

    // --- Inverse trig ---
    {
        auto x = linspace(-0.9, 0.9, 10);
        auto as = arcsin(x);
        auto ac = arccos(x);
        auto at = arctan(x);

        test::check(test::approx(as.at(5), std::asin(x.at(5))), "arcsin");
        test::check(test::approx(ac.at(5), std::acos(x.at(5))), "arccos");
        test::check(test::approx(at.at(5), std::atan(x.at(5))), "arctan");
    }

    // --- arctan2 and hypot ---
    {
        auto y = ones<double>({3});
        auto x = ones<double>({3});
        x.fill(1.0);
        y.fill(1.0);

        auto atan2_result = arctan2(y, x);
        test::check(test::approx(atan2_result.at(0), std::numbers::pi / 4.0), "arctan2(1, 1) = pi/4");

        auto hyp = hypot(y, x);
        test::check(test::approx(hyp.at(0), std::sqrt(2.0)), "hypot(1, 1) = sqrt(2)");
    }

    // --- Degree/radian conversion ---
    {
        auto rad = linspace(0.0, std::numbers::pi, 5);
        auto deg = degrees(rad);
        test::check(test::approx(deg.at(4), 180.0), "pi radians = 180 degrees");

        auto back = radians(deg);
        test::check(test::approx(back.at(4), std::numbers::pi), "180 deg = pi rad");
    }

    // --- Hyperbolic functions ---
    {
        auto x = linspace(-1.0, 1.0, 5);
        auto sh = sinh(x);
        auto ch = cosh(x);
        auto th = tanh(x);

        test::check(test::approx(sh.at(2), std::sinh(0.0)), "sinh(0) = 0");
        test::check(test::approx(ch.at(2), std::cosh(0.0)), "cosh(0) = 1");
        test::check(test::approx(th.at(2), std::tanh(0.0)), "tanh(0) = 0");
    }

    // --- Exponential and logarithmic ---
    {
        auto x = linspace(0.1, 2.0, 10);
        auto e = exp(x);
        auto l = log(e);

        test::check(test::approx(l.at(0), x.at(0), 1e-6), "log(exp(x)) = x");

        auto l10 = log10(x);
        auto l2 = log2(x);
        test::check(test::approx(l10.at(5), std::log10(x.at(5))), "log10");
        test::check(test::approx(l2.at(5), std::log2(x.at(5))), "log2");

        auto sq = sqrt(x);
        test::check(test::approx(sq.at(5), std::sqrt(x.at(5))), "sqrt");

        auto cb = cbrt(x);
        test::check(test::approx(cb.at(5), std::cbrt(x.at(5))), "cbrt");
    }

    // --- Power and square ---
    {
        auto base = linspace(1.0, 3.0, 5);
        auto exponent = full({5}, 2.0);
        auto p = power(base, exponent);

        test::check(test::approx(p.at(0), 1.0), "1^2 = 1");
        test::check(test::approx(p.at(4), 9.0), "3^2 = 9");

        auto sq = square(base);
        test::check(test::approx(sq.at(0), 1.0), "square(1) = 1");
        test::check(test::approx(sq.at(4), 9.0), "square(3) = 9");
    }

    // --- Rounding ---
    {
        auto x = asarray(std::vector<double>{-1.7, -0.5, 0.0, 0.5, 1.7, 2.3});

        auto f = floor(x);
        test::check(f.at(0) == -2.0, "floor(-1.7) = -2");
        test::check(f.at(4) == 1.0, "floor(1.7) = 1");

        auto c = ceil(x);
        test::check(c.at(0) == -1.0, "ceil(-1.7) = -1");
        test::check(c.at(4) == 2.0, "ceil(1.7) = 2");

        auto tr = trunc(x);
        test::check(tr.at(0) == -1.0, "trunc(-1.7) = -1");
        test::check(tr.at(4) == 1.0, "trunc(1.7) = 1");
    }

    // --- Arithmetic ---
    {
        auto x = asarray(std::vector<double>{-3.5, -1.0, 0.0, 1.0, 3.5});

        auto ab = absolute(x);
        test::check(ab.at(0) == 3.5, "abs(-3.5) = 3.5");
        test::check(ab.at(1) == 1.0, "abs(-1.0) = 1.0");

        auto sg = sign(x);
        test::check(sg.at(0) == -1.0, "sign(-3.5) = -1");
        test::check(sg.at(2) == 0.0, "sign(0) = 0");
        test::check(sg.at(4) == 1.0, "sign(3.5) = 1");

        auto a = asarray(std::vector<double>{1.0, 5.0, 3.0});
        auto b = asarray(std::vector<double>{4.0, 2.0, 6.0});

        auto mx = maximum(a, b);
        test::check(mx.at(0) == 4.0, "max(1, 4) = 4");
        test::check(mx.at(1) == 5.0, "max(5, 2) = 5");

        auto mn = minimum(a, b);
        test::check(mn.at(0) == 1.0, "min(1, 4) = 1");
        test::check(mn.at(1) == 2.0, "min(5, 2) = 2");
    }

    // --- Reciprocal ---
    {
        auto x = asarray(std::vector<double>{1.0, 2.0, 4.0});
        auto r = reciprocal(x);
        test::check(test::approx(r.at(0), 1.0), "1/1 = 1");
        test::check(test::approx(r.at(1), 0.5), "1/2 = 0.5");
        test::check(test::approx(r.at(2), 0.25), "1/4 = 0.25");
    }

    // --- New ufuncs: copysign, logaddexp, logaddexp2 ---
    {
        auto mag = asarray(std::vector<double>{-3.0, 4.0, -5.0});
        auto sns = asarray(std::vector<double>{-1.0, 1.0, -1.0});
        auto cs = copysign(mag, sns);
        test::check(cs.at(0) == -3.0, "copysign(-3, -1) = -3");
        test::check(cs.at(1) == 4.0, "copysign(4, 1) = 4");
        test::check(cs.at(2) == -5.0, "copysign(-5, -1) = -5");

        auto a = asarray(std::vector<double>{-std::numeric_limits<double>::infinity(),
                                             std::numeric_limits<double>::infinity(), 1e-300, 1e300});
        auto b = asarray(std::vector<double>{-std::numeric_limits<double>::infinity(),
                                             std::numeric_limits<double>::infinity(), 1e-300, 1e-300});
        auto le = logaddexp(a, b);
        test::check(std::isinf(le.at(0)) && le.at(0) < 0, "logaddexp(-inf, -inf) = -inf");
        test::check(std::isinf(le.at(1)) && le.at(1) > 0, "logaddexp(inf, inf) = inf");
        test::check(test::approx(le.at(2), std::log1p(1.0), 1e-6), "logaddexp(1e-300, 1e-300) = ln(2) ~ 0.693");
        test::check(test::approx(le.at(3), 1e300 + std::log1p(0.0), 1e-6), "logaddexp(1e300, 1e-300) ~ 1e300");

        auto le2 = logaddexp2(a, b);
        test::check(std::isinf(le2.at(0)) && le2.at(0) < 0, "logaddexp2(-inf, -inf) = -inf");
        test::check(std::isinf(le2.at(1)) && le2.at(1) > 0, "logaddexp2(inf, inf) = inf");
        test::check(test::approx(le2.at(3), a.at(3), 1e-9), "logaddexp2(1e300, 1e-300) ~ 1e300");
    }

    // --- New ufuncs: divide, true_divide, floor_divide ---
    {
        auto a = asarray(std::vector<double>{10.0, -10.0, 9.0});
        auto b = asarray(std::vector<double>{3.0, 3.0, -2.0});
        auto dv = divide(a, b);
        test::check(test::approx(dv.at(0), 10.0 / 3.0), "divide(10, 3) = 10/3");
        auto td = true_divide(a, b);
        test::check(test::approx(td.at(0), 10.0 / 3.0), "true_divide(10, 3) = 10/3");
        auto fd = floor_divide(a, b);
        test::check(fd.at(0) == 3.0, "floor_divide(10, 3) = 3");
        test::check(fd.at(1) == -4.0, "floor_divide(-10, 3) = -4");
        test::check(fd.at(2) == -5.0, "floor_divide(9, -2) = -5");

        auto ia = asarray(std::vector<int>{7, -7});
        auto ib = asarray(std::vector<int>{2, 3});
        auto idv = divide(ia, ib);
        test::check(idv.at(0) == 3, "divide(7, 2) = 3 (int, matches operator/)");
        auto ifd = floor_divide(ia, ib);
        test::check(ifd.at(0) == 3, "floor_divide(7, 2) = 3");
        test::check(ifd.at(1) == -3, "floor_divide(-7, 3) = -3");
    }

    // --- New ufuncs: round / around (half-to-even) ---
    {
        auto x = asarray(std::vector<double>{0.5, 1.5, 2.5, -0.5, -1.5, 2.5});
        auto r = round(x);
        test::check(r.at(0) == 0.0, "round(0.5) = 0 (half-to-even)");
        test::check(r.at(1) == 2.0, "round(1.5) = 2 (half-to-even)");
        test::check(r.at(2) == 2.0, "round(2.5) = 2 (half-to-even)");
        test::check(r.at(3) == 0.0, "round(-0.5) = 0 (half-to-even)");
        test::check(r.at(4) == -2.0, "round(-1.5) = -2 (half-to-even)");

        auto r2 = around(x, 1);
        test::check(r2.at(5) == 2.5, "around(2.5, 1) = 2.5");

        auto xd = asarray(std::vector<double>{12.345, 678.9});
        auto rd = round(xd, 1);
        test::check(test::approx(rd.at(0), 12.3), "round(12.345, 1) = 12.3");
        auto rd0 = round(xd, -1);
        test::check(test::approx(rd0.at(1), 680.0), "round(678.9, -1) = 680");

        auto i = asarray(std::vector<int>{3, 8});
        auto ri = round(i);
        test::check(ri.at(0) == 3 && ri.at(1) == 8, "round(int) is identity");
    }

    // --- sign(NaN) propagates; complex sign unit vector ---
    {
        auto x = asarray(std::vector<double>{std::numeric_limits<double>::quiet_NaN()});
        auto sg = sign(x);
        test::check(std::isnan(sg.at(0)), "sign(NaN) = NaN");

        auto c = asarray(std::vector<std::complex<double>>{{3.0, 4.0}});
        auto sc = sign(c);
        test::check(test::approx_c(sc.at(0), std::complex<double>{0.6, 0.8}), "sign(3+4i) = (3+4i)/5");
    }

    // --- power with exact integer exponentiation ---
    {
        auto base = asarray(std::vector<int>{2, 3, -2});
        auto expo = asarray(std::vector<int>{10, 0, 3});
        auto p = power(base, expo);
        test::check(p.at(0) == 1024, "power(2, 10) = 1024 exact");
        test::check(p.at(1) == 1, "power(3, 0) = 1 exact");
        test::check(p.at(2) == -8, "power(-2, 3) = -8 exact");

        auto pn = power(asarray(std::vector<int>{2}), asarray(std::vector<int>{-1}));
        test::check(pn.at(0) == 0, "power(2, -1) = 0 (int truncation)");
    }

    // --- fma single-pass correctness ---
    {
        const double A = 1.0 + std::ldexp(1.0, -27);
        const double B = A;
        const double C = -(1.0 + std::ldexp(1.0, -52));
        auto a = asarray(std::vector<double>{A});
        auto b = asarray(std::vector<double>{B});
        auto c = asarray(std::vector<double>{C});
        auto res = fma(a, b, c);
        test::check(res.at(0) == std::fma(A, B, C), "fma(A,B,C) == std::fma (single rounding step)");
        test::check(res.at(0) != A * B + C, "fma(A,B,C) != A*B+C (two-pass loses 2^-54)");
    }

    // --- out= overloads (unary, binary, ternary, aliases) ---
    {
        auto x = linspace(-1.0, 1.0, 6);
        auto out = zeros<double>(x.shape);
        auto &sref = sin(x, out);
        test::check(&sref == &out, "out= returns the same ndarray");
        test::check(test::approx(out.at(0), std::sin(x.at(0))), "sin out= value");

        auto mx = asarray(std::vector<double>{1.0, 5.0, 3.0});
        auto mn = asarray(std::vector<double>{4.0, 2.0, 6.0});
        auto mout = zeros<double>({3});
        maximum(mx, mn, mout);
        test::check(mout.at(0) == 4.0 && mout.at(1) == 5.0 && mout.at(2) == 6.0, "maximum out= value");

        // broadcast into out
        auto mat = asarray(std::vector<double>{1.0, 5.0, 3.0, 7.0}, {2, 2});
        auto row = asarray(std::vector<double>{4.0, 2.0});
        auto bout = zeros<double>({2, 2});
        maximum(mat, row, bout);
        test::check(bout.at(0, 0) == 4.0 && bout.at(0, 1) == 5.0 && bout.at(1, 0) == 4.0 && bout.at(1, 1) == 7.0,
                    "maximum out= broadcast value");

        // wrong-shape out must throw
        bool threw_sin = false, threw_max = false;
        try
        {
            auto bad = zeros<double>({2});
            sin(x, bad);
        }
        catch (const std::invalid_argument &)
        {
            threw_sin = true;
        }
        try
        {
            auto bad = zeros<double>({2});
            maximum(mx, mn, bad);
        }
        catch (const std::invalid_argument &)
        {
            threw_max = true;
        }
        test::check(threw_sin, "sin out= throws on shape mismatch");
        test::check(threw_max, "maximum out= throws on shape mismatch");

        // aliases with out=
        auto aout = zeros<double>({3});
        abs(mx, aout);
        test::check(aout.at(0) == 1.0, "abs() out= alias");
        auto dout = zeros<double>({3});
        mod(mx, mn, dout);
        test::check(test::approx(dout.at(0), std::remainder(1.0, 4.0)), "mod() out= alias -> remainder");

        // ternary out=
        auto fout = zeros<double>({1});
        auto fa = asarray(std::vector<double>{2.0});
        auto fb = asarray(std::vector<double>{3.0});
        auto fc = asarray(std::vector<double>{4.0});
        fma(fa, fb, fc, fout);
        test::check(fout.at(0) == 10.0, "fma out= value");

        // nan_to_num with out=
        auto nanx = asarray(
            std::vector<double>{std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()});
        auto nout = zeros<double>({2});
        nan_to_num(nanx, nout, 7.0, 99.0, -99.0);
        test::check(nout.at(0) == 7.0 && nout.at(1) == 99.0, "nan_to_num out= value");

        // square out= (SIMD path on contiguous double)
        auto sqout = zeros<double>(x.shape);
        square(x, sqout);
        test::check(test::approx(sqout.at(3), x.at(3) * x.at(3)), "square out= value");
    }

    // --- divide SIMD fast path vs elementwise ---
    {
        auto a = linspace(1.0, 100.0, 1000);
        auto b = linspace(0.001, 1.0, 1000);
        auto d = divide(a, b);
        test::check(test::approx(d.at(0), a.at(0) / b.at(0)), "divide SIMD value");
        test::check(test::approx(d.at(999), a.at(999) / b.at(999)), "divide SIMD value (tail)");
    }

    return test::failures() ? 1 : 0;
}
