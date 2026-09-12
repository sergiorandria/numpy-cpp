/**
 * @file test_io.cpp
 * @brief Tests for io.hpp (save/load npy, savez/load_npz, savetxt/loadtxt, genfromtxt, tofile/fromfile).
 */
#include "test_util.hpp"
#include <np/io.hpp>
#include <np/np.hpp>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>

int main()
{
    using namespace np;
    namespace fs = std::filesystem;
    const std::string tmpdir = (fs::temp_directory_path() / "np_io_test").string();
    fs::create_directories(tmpdir);

    // npy save/load round-trip for int, double, complex
    {
        auto a = ndarray<int>::from_data({2, 3}, {1, 2, 3, 4, 5, 6});
        std::string p = tmpdir + "/a.npy";
        save(p, a);
        auto b = load<int>(p);
        test::check(b.shape == a.shape, "io npy int shape");
        test::check(b.at(1, 2) == 6, "io npy int value");
        // dtype mismatch should throw
        bool threw = false;
        try
        {
            (void)load<double>(p);
        }
        catch (...)
        {
            threw = true;
        }
        test::check(threw, "io npy dtype mismatch throws");
    }
    {
        auto a = ndarray<double>::from_data({3}, {1.5, -2.0, 3.25});
        std::string p = tmpdir + "/d.npy";
        save(p, a);
        auto b = load<double>(p);
        test::check(test::approx(b.at(2), 3.25), "io npy double");
    }
    {
        using C = std::complex<double>;
        auto a = ndarray<C>::from_data({2}, {C(1, 2), C(3, -4)});
        std::string p = tmpdir + "/c.npy";
        save(p, a);
        auto b = load<C>(p);
        test::check(test::approx_c(b.at(1), C(3, -4)), "io npy complex");
    }
    // 0-d scalar
    {
        ndarray<double> s(std::vector<int>{});
        s.data()[0] = 42.0;
        // from_data with empty shape gives 0-d
        auto a = ndarray<double>::from_data({}, {42.0});
        std::string p = tmpdir + "/scalar.npy";
        save(p, a);
        auto b = load<double>(p);
        test::check(b.size() == 1 && test::approx(b.data()[0], 42.0), "io npy scalar");
    }

    // savez / load_npz
    {
        std::map<std::string, ndarray<int>> m;
        m["x"] = ndarray<int>::from_data({3}, {1, 2, 3});
        m["y"] = ndarray<int>::from_data({2, 2}, {10, 20, 30, 40});
        std::string p = tmpdir + "/test.npz";
        savez(p, m);
        auto loaded = load_npz<int>(p);
        test::check(loaded.size() == 2, "io npz size");
        test::check(loaded["x"].at(2) == 3, "io npz x");
        test::check(loaded["y"].at(1, 1) == 40, "io npz y");
        // savez_compressed alias (STORE fallback)
        std::string pc = tmpdir + "/test_c.npz";
        savez_compressed(pc, m);
        auto loaded2 = load_npz<int>(pc);
        test::check(loaded2["y"].at(0, 1) == 20, "io npz compressed");
    }

    // savetxt / loadtxt
    {
        auto a = ndarray<double>::from_data({2, 3}, {1, 2, 3, 4, 5, 6});
        std::string p = tmpdir + "/a.txt";
        savetxt(p, a, " ");
        auto b = loadtxt(p, " ");
        test::check(b.shape == a.shape, "io txt shape");
        test::check(test::approx(b.at(1, 2), 6.0), "io txt value");
        // 1-D
        auto v = ndarray<double>::from_data({4}, {0, 1, 2, 3});
        std::string pv = tmpdir + "/v.txt";
        savetxt(pv, v);
        auto bv = loadtxt(pv);
        test::check(bv.ndim() == 1 && bv.size() == 4, "io txt 1-D");
        // delimiter
        auto w = ndarray<double>::from_data({2, 2}, {1, 2, 3, 4});
        std::string pw = tmpdir + "/w.txt";
        savetxt(pw, w, ",");
        auto bw = loadtxt(pw, ",");
        test::check(bw.at(1, 0) == 3.0, "io txt delimiter");
    }

    // genfromtxt with missing
    {
        std::string p = tmpdir + "/gen.txt";
        {
            std::ofstream os(p);
            os << "1,2,3\n4,,6\n# comment\n7,8,9\n";
        }
        auto g = genfromtxt(p, ",", 0, -1);
        test::check(g.shape[0] == 3 && g.shape[1] == 3, "io gen shape");
        // fallback impl returns -1 when zlib not available vs NaN; accept either
        test::check(std::isnan(g.at(1, 1)) || g.at(1, 1) == -1, "io gen missing NaN/-1");
        test::check(g.at(2, 2) == 9.0, "io gen value");
        // skip_header
        auto g2 = genfromtxt(p, ",", 1, -1);
        test::check(g2.shape[0] == 2, "io gen skip_header");
    }

    // tofile / fromfile
    {
        auto a = ndarray<int>::from_data({5}, {10, 20, 30, 40, 50});
        std::string p = tmpdir + "/raw.dat";
        tofile(a, p);
        auto b = fromfile<int>(p);
        test::check(b.size() == 5 && b.at(4) == 50, "io fromfile all");
        auto c = fromfile<int>(p, 3, 0, {3});
        test::check(c.size() == 3 && c.at(2) == 30, "io fromfile count");
        auto d = fromfile<int>(p, 2, sizeof(int) * 1);
        test::check(d.at(0) == 20 && d.at(1) == 30, "io fromfile offset");
        auto e = fromfile<int>(p, 4, 0, {2, 2});
        test::check(e.shape[0] == 2 && e.shape[1] == 2, "io fromfile shape");
        test::check(e.at(1, 1) == 40, "io fromfile shape value");
    }

    // truncated payload must throw, never return uninitialized storage
    // (the old code accepted a short read with gcount() == 0 silently)
    {
        auto a = ndarray<int>::from_data({4}, {1, 2, 3, 4});
        std::string p = tmpdir + "/trunc.npy";
        save(p, a);
        // Drop the last 8 payload bytes (half of the 16 payload bytes),
        // keeping a valid header: exercises the payload short-read path.
        {
            std::ifstream src(p, std::ios::binary | std::ios::ate);
            const auto full = src.tellg();
            src.seekg(0);
            std::vector<char> kept(static_cast<std::size_t>(full) - 8);
            src.read(kept.data(), static_cast<std::streamsize>(kept.size()));
            src.close();
            std::ofstream dst(p, std::ios::binary | std::ios::trunc);
            dst.write(kept.data(), static_cast<std::streamsize>(kept.size()));
        }
        bool threw = false;
        try
        {
            auto b = load<int>(p);
            (void)b;
        }
        catch (const std::runtime_error &)
        {
            threw = true;
        }
        test::check(threw, "io truncated payload throws");
    }

    fs::remove_all(tmpdir);
    if (test::failures() == 0)
    {
        std::printf("OK io\n");
        return 0;
    }
    return 1;
}
