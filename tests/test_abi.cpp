/**
 * @file test_abi.cpp
 * @brief Tests for abi.hpp (version macros, inline namespace v1 integration).
 *
 * Every subsystem lives in `np::v1::<sub>`; the inline namespace makes
 * `np::<sub>::...` and `np::v1::<sub>::...` the same entity. These tests
 * pin the version macros, the `np::abi` helpers, and type/function identity
 * across both spellings for a sample of subsystems.
 */
#include <complex>
#include <string>
#include <type_traits>
#include <vector>

#include <np/np.hpp>

#include "test_util.hpp"

// Compile-time identity: unqualified and version-qualified names denote
// the same entities (would fail if v1 were a non-inline fork).
static_assert(std::is_same_v<np::ndarray<double>, np::v1::ndarray<double>>, "core ndarray identity");
static_assert(std::is_same_v<np::bundle::VectorBundle, np::v1::bundle::VectorBundle>, "bundle identity");
static_assert(std::is_same_v<np::random::Generator, np::v1::random::Generator>, "random identity");
static_assert(std::is_same_v<np::manifold::SphereManifold, np::v1::manifold::SphereManifold>, "manifold identity");
static_assert(std::is_same_v<np::differential::KForm, np::v1::differential::KForm>, "differential identity");
static_assert(std::is_same_v<np::lattice::LatticeFactory, np::v1::lattice::LatticeFactory>, "lattice identity");
static_assert(std::is_same_v<np::bigint, np::v1::bigint>, "bigint identity");

using Cplx = std::complex<double>;
using FftFn = std::vector<Cplx> (*)(const std::vector<Cplx> &);

int main()
{
    // — Version macros mirror CMake project(VERSION 1.0.0) + NP_ABI_VERSION —
    test::check(NP_VERSION_MAJOR == 1 && NP_VERSION_MINOR == 0 && NP_VERSION_PATCH == 0, "version macros");
    test::check(NP_ABI_VERSION == 1, "abi version macro");
    test::check(np::abi::version() == NP_ABI_VERSION, "abi::version()");
    test::check(np::abi::version_major() == 1 && np::abi::version_minor() == 0 && np::abi::version_patch() == 0,
                "abi::version_*()");
    test::check(std::string(np::abi::version_string()) == "1.0.0 (ABI v1)", "abi::version_string()");
    test::check(std::string(np::abi::abi_tag()) == "np-abi-v1", "abi::abi_tag()");

    // — Function identity across spellings (same address => same entity) —
    {
        const auto f1 = static_cast<FftFn>(&np::fft::fft);
        const auto f2 = static_cast<FftFn>(&np::v1::fft::fft);
        test::check(f1 == f2, "fft identity");
        const std::vector<Cplx> x{{1.0, 0.0}, {0.0, 0.0}};
        test::check(f1(x) == f2(x), "fft both spellings agree");
    }
    // — Core API reachable through the versioned spelling —
    {
        const auto a = np::v1::arange<double>(0, 3);
        test::check(a._numel() == 3 && a(2) == 2.0, "v1 core arange");
    }
    // — Subsystem call through the versioned spelling —
    {
        const auto E = np::v1::bundle::tangent_bundle(np::manifold::SphereManifold(2));
        test::check(E.rank == 2, "v1 bundle tangent_bundle");
    }
    return test::failures() ? 1 : 0;
}
