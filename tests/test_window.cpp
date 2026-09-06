/**
 * @file test_window.cpp
 * @brief Tests for window.hpp (bartlett, blackman, hamming, hanning, kaiser)
 */
#include "test_util.hpp"
#include <cmath>
#include <np/np.hpp>
#include <np/window.hpp>

int main()
{
    using namespace np;
    // edge cases M=0,1, negative
    {
        test::check(bartlett(0).size() == 0, "bartlett 0");
        test::check(blackman(0).size() == 0, "blackman 0");
        test::check(hamming(0).size() == 0, "hamming 0");
        test::check(hanning(0).size() == 0, "hanning 0");
        test::check(kaiser(0, 5).size() == 0, "kaiser 0");
        test::check(bartlett(1).at(0) == 1.0, "bartlett 1");
        test::check(kaiser(1, 0).at(0) == 1.0, "kaiser 1");
        bool threw = false;
        try
        {
            bartlett(-1);
        }
        catch (...)
        {
            threw = true;
        }
        test::check(threw, "bartlett negative");
    }
    // symmetry and known values
    {
        auto w = bartlett(5);
        test::check(w.size() == 5 && test::approx(w.at(0), 0.0) && test::approx(w.at(2), 1.0) &&
                        test::approx(w.at(4), 0.0),
                    "bartlett symmetry");
        auto bh = bartlett(6);
        test::check(test::approx(bh.at(0), 0.0) && test::approx(bh.at(5), 0.0), "bartlett 6");
    }
    {
        auto w = blackman(5);
        test::check(w.size() == 5 && test::approx(w.at(0), 0.0, 1e-9) && w.at(2) > 0.8, "blackman");
        auto w2 = blackman(7);
        test::check(test::approx(w2.at(0), w2.at(6)), "blackman symmetric");
    }
    {
        auto w = hamming(5);
        test::check(test::approx(w.at(0), 0.08, 1e-9) && test::approx(w.at(2), 1.0), "hamming");
        test::check(test::approx(w.at(0), w.at(4)), "hamming sym");
    }
    {
        auto w = hanning(8);
        test::check(test::approx(w.at(0), 0.0) && test::approx(w.at(7), 0.0) && w.at(4) > 0.9, "hanning");
        auto w2 = hann(8);
        test::check(w2.size() == 8 && test::approx(w2.at(1), w.at(1)), "hann alias");
    }
    {
        auto w = kaiser(5, 14.0);
        test::check(w.size() == 5 && test::approx(w.at(0), w.at(4)) && w.at(2) == 1.0, "kaiser beta14");
        auto w0 = kaiser(5, 0.0);
        test::check(w0.at(0) == 1.0 && w0.at(2) == 1.0, "kaiser beta0");
        bool threw = false;
        try
        {
            kaiser(-1, 0);
        }
        catch (...)
        {
            threw = true;
        }
        test::check(threw, "kaiser negative");
    }
    if (test::failures() == 0)
        std::printf("OK window\n");
    return test::failures() ? 1 : 0;
}
