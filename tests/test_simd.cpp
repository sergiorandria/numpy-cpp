/**
 * @file test_simd.cpp
 * @brief Test suite for SIMD optimizations.
 *
 * Tests vectorized operations across different instruction sets:
 * SSE2, AVX, AVX2, AVX-512, ARM NEON.
 */

#include "test_util.hpp"
#include <np/simd.hpp>

#include <cmath>
#include <iostream>
#include <vector>

// --- Feature Detection Tests ---

void test_feature_detection()
{
    std::cout << "\n=== SIMD Feature Detection ===\n";
    std::cout << "SSE2:    " << (np::simd::Features::has_sse2 ? "YES" : "NO") << "\n";
    std::cout << "SSE3:    " << (np::simd::Features::has_sse3 ? "YES" : "NO") << "\n";
    std::cout << "SSSE3:   " << (np::simd::Features::has_ssse3 ? "YES" : "NO") << "\n";
    std::cout << "SSE4.1:  " << (np::simd::Features::has_sse41 ? "YES" : "NO") << "\n";
    std::cout << "SSE4.2:  " << (np::simd::Features::has_sse42 ? "YES" : "NO") << "\n";
    std::cout << "AVX:     " << (np::simd::Features::has_avx ? "YES" : "NO") << "\n";
    std::cout << "AVX2:    " << (np::simd::Features::has_avx2 ? "YES" : "NO") << "\n";
    std::cout << "AVX-512: " << (np::simd::Features::has_avx512 ? "YES" : "NO") << "\n";
    std::cout << "NEON:    " << (np::simd::Features::has_neon ? "YES" : "NO") << "\n";

    std::cout << "\nVector widths (elements):\n";
    std::cout << "float:  " << np::simd::VectorWidth<float>::value << "\n";
    std::cout << "double: " << np::simd::VectorWidth<double>::value << "\n";
}

// --- Addition Tests ---

void test_add_f32()
{
    constexpr std::size_t n = 1024;
    std::vector<float> a(n), b(n), result(n), expected(n);

    // Initialize test data
    for (std::size_t i = 0; i < n; ++i)
    {
        a[i] = static_cast<float>(i) * 0.5f;
        b[i] = static_cast<float>(i) * 0.25f;
        expected[i] = a[i] + b[i];
    }

    // Test vectorized addition
    np::simd::add_vectorized(a.data(), b.data(), result.data(), n);

    // Verify results
    for (std::size_t i = 0; i < n; ++i)
    {
        if (!test::approx(result[i], expected[i], 1e-5f))
        {
            test::check(false, "add_vectorized f32 mismatch");
            return;
        }
    }
    test::check(true, "add_vectorized f32");
}

void test_add_f64()
{
    constexpr std::size_t n = 512;
    std::vector<double> a(n), b(n), result(n), expected(n);

    for (std::size_t i = 0; i < n; ++i)
    {
        a[i] = static_cast<double>(i) * 0.5;
        b[i] = static_cast<double>(i) * 0.25;
        expected[i] = a[i] + b[i];
    }

    np::simd::add_vectorized(a.data(), b.data(), result.data(), n);

    for (std::size_t i = 0; i < n; ++i)
    {
        if (!test::approx(result[i], expected[i], 1e-10))
        {
            test::check(false, "add_vectorized f64 mismatch");
            return;
        }
    }
    test::check(true, "add_vectorized f64");
}

// --- Subtraction Tests ---

void test_sub_f32()
{
    constexpr std::size_t n = 1024;
    std::vector<float> a(n), b(n), result(n), expected(n);

    for (std::size_t i = 0; i < n; ++i)
    {
        a[i] = static_cast<float>(i) * 1.5f;
        b[i] = static_cast<float>(i) * 0.75f;
        expected[i] = a[i] - b[i];
    }

    np::simd::sub_vectorized(a.data(), b.data(), result.data(), n);

    for (std::size_t i = 0; i < n; ++i)
    {
        if (!test::approx(result[i], expected[i], 1e-5f))
        {
            test::check(false, "sub_vectorized f32 mismatch");
            return;
        }
    }
    test::check(true, "sub_vectorized f32");
}

void test_sub_f64()
{
    constexpr std::size_t n = 512;
    std::vector<double> a(n), b(n), result(n), expected(n);

    for (std::size_t i = 0; i < n; ++i)
    {
        a[i] = static_cast<double>(i) * 1.5;
        b[i] = static_cast<double>(i) * 0.75;
        expected[i] = a[i] - b[i];
    }

    np::simd::sub_vectorized(a.data(), b.data(), result.data(), n);

    for (std::size_t i = 0; i < n; ++i)
    {
        if (!test::approx(result[i], expected[i], 1e-10))
        {
            test::check(false, "sub_vectorized f64 mismatch");
            return;
        }
    }
    test::check(true, "sub_vectorized f64");
}

// --- Multiplication Tests ---

void test_mul_f32()
{
    constexpr std::size_t n = 1024;
    std::vector<float> a(n), b(n), result(n), expected(n);

    for (std::size_t i = 0; i < n; ++i)
    {
        a[i] = static_cast<float>(i) * 0.1f;
        b[i] = static_cast<float>(i) * 0.2f;
        expected[i] = a[i] * b[i];
    }

    np::simd::mul_vectorized(a.data(), b.data(), result.data(), n);

    for (std::size_t i = 0; i < n; ++i)
    {
        if (!test::approx(result[i], expected[i], 1e-5f))
        {
            test::check(false, "mul_vectorized f32 mismatch");
            return;
        }
    }
    test::check(true, "mul_vectorized f32");
}

void test_mul_f64()
{
    constexpr std::size_t n = 512;
    std::vector<double> a(n), b(n), result(n), expected(n);

    for (std::size_t i = 0; i < n; ++i)
    {
        a[i] = static_cast<double>(i) * 0.1;
        b[i] = static_cast<double>(i) * 0.2;
        expected[i] = a[i] * b[i];
    }

    np::simd::mul_vectorized(a.data(), b.data(), result.data(), n);

    for (std::size_t i = 0; i < n; ++i)
    {
        if (!test::approx(result[i], expected[i], 1e-10))
        {
            test::check(false, "mul_vectorized f64 mismatch");
            return;
        }
    }
    test::check(true, "mul_vectorized f64");
}

// --- Division Tests ---

void test_div_f32()
{
    constexpr std::size_t n = 1024;
    std::vector<float> a(n), b(n), result(n), expected(n);

    for (std::size_t i = 0; i < n; ++i)
    {
        a[i] = static_cast<float>(i + 1) * 10.0f;
        b[i] = static_cast<float>(i + 1) * 2.0f;
        expected[i] = a[i] / b[i];
    }

    np::simd::div_vectorized(a.data(), b.data(), result.data(), n);

    for (std::size_t i = 0; i < n; ++i)
    {
        if (!test::approx(result[i], expected[i], 1e-5f))
        {
            test::check(false, "div_vectorized f32 mismatch");
            return;
        }
    }
    test::check(true, "div_vectorized f32");
}

void test_div_f64()
{
    constexpr std::size_t n = 512;
    std::vector<double> a(n), b(n), result(n), expected(n);

    for (std::size_t i = 0; i < n; ++i)
    {
        a[i] = static_cast<double>(i + 1) * 10.0;
        b[i] = static_cast<double>(i + 1) * 2.0;
        expected[i] = a[i] / b[i];
    }

    np::simd::div_vectorized(a.data(), b.data(), result.data(), n);

    for (std::size_t i = 0; i < n; ++i)
    {
        if (!test::approx(result[i], expected[i], 1e-10))
        {
            test::check(false, "div_vectorized f64 mismatch");
            return;
        }
    }
    test::check(true, "div_vectorized f64");
}

// --- Sum Reduction Tests ---

void test_sum_f32()
{
    constexpr std::size_t n = 1024;
    std::vector<float> data(n);
    float expected = 0.0f;

    for (std::size_t i = 0; i < n; ++i)
    {
        data[i] = static_cast<float>(i) * 0.1f;
        expected += data[i];
    }

    float result = np::simd::sum_vectorized(data.data(), n);

    test::check(test::approx(result, expected, 1e-3f), "sum_vectorized f32");
}

void test_sum_f64()
{
    constexpr std::size_t n = 512;
    std::vector<double> data(n);
    double expected = 0.0;

    for (std::size_t i = 0; i < n; ++i)
    {
        data[i] = static_cast<double>(i) * 0.1;
        expected += data[i];
    }

    double result = np::simd::sum_vectorized(data.data(), n);

    test::check(test::approx(result, expected, 1e-8), "sum_vectorized f64");
}

// --- Edge Cases ---

void test_misaligned_sizes()
{
    // Test with sizes not perfectly divisible by vector width
    for (std::size_t n : {1, 3, 7, 15, 17, 31, 63, 127})
    {
        std::vector<float> a(n), b(n), result(n), expected(n);

        for (std::size_t i = 0; i < n; ++i)
        {
            a[i] = static_cast<float>(i);
            b[i] = static_cast<float>(i) * 2.0f;
            expected[i] = a[i] + b[i];
        }

        np::simd::add_vectorized(a.data(), b.data(), result.data(), n);

        bool ok = true;
        for (std::size_t i = 0; i < n; ++i)
        {
            if (!test::approx(result[i], expected[i], 1e-5f))
            {
                ok = false;
                break;
            }
        }
        if (!ok)
        {
            test::check(false, "misaligned sizes");
            return;
        }
    }
    test::check(true, "misaligned sizes");
}

void test_large_arrays()
{
    // Test with large array to stress SIMD paths
    constexpr std::size_t n = 1000000;
    std::vector<float> a(n), b(n), result(n);

    for (std::size_t i = 0; i < n; ++i)
    {
        a[i] = static_cast<float>(i % 1000) * 0.001f;
        b[i] = static_cast<float>(i % 500) * 0.002f;
    }

    // Just verify it completes without errors
    np::simd::add_vectorized(a.data(), b.data(), result.data(), n);
    np::simd::mul_vectorized(a.data(), b.data(), result.data(), n);
    float sum = np::simd::sum_vectorized(result.data(), n);

    test::check(!std::isnan(sum) && !std::isinf(sum), "large array sum is finite");
}

void test_negative_numbers()
{
    constexpr std::size_t n = 256;
    std::vector<double> a(n), b(n), result(n), expected(n);

    for (std::size_t i = 0; i < n; ++i)
    {
        a[i] = static_cast<double>(static_cast<int>(i) - 128) * 0.5;
        b[i] = static_cast<double>(static_cast<int>(i) - 64) * 0.25;
        expected[i] = a[i] * b[i];
    }

    np::simd::mul_vectorized(a.data(), b.data(), result.data(), n);

    for (std::size_t i = 0; i < n; ++i)
    {
        if (!test::approx(result[i], expected[i], 1e-10))
        {
            test::check(false, "negative numbers mul mismatch");
            return;
        }
    }
    test::check(true, "negative numbers mul");
}

// --- Main Test Runner ---

int main()
{
    test_feature_detection();

    std::cout << "\n=== Running SIMD Operation Tests ===\n";

    // Addition
    test_add_f32();
    test_add_f64();

    // Subtraction
    test_sub_f32();
    test_sub_f64();

    // Multiplication
    test_mul_f32();
    test_mul_f64();

    // Division
    test_div_f32();
    test_div_f64();

    // Sum reduction
    test_sum_f32();
    test_sum_f64();

    // Edge cases
    test_misaligned_sizes();
    test_large_arrays();
    test_negative_numbers();

    std::cout << "\n=== SIMD Tests Complete ===\n";
    return test::failures() ? 1 : 0;
}
