/**
 * @file abi.hpp
 * @brief ABI versioning for the numpy-cpp header-only library.
 *
 * All `np::` subsystems live in the versioned inline namespace `np::v1`.
 * Every subsystem header reopens it with an explicit `inline` specifier
 * (`namespace np::inline v1`, or the split `namespace np::inline v1` /
 * `namespace <sub>` form for nested subsystems), so `np::<sub>::...` and
 * `np::v1::<sub>::...` name the same entity while distinct ABI versions get
 * distinct mangled names. The explicit `inline` keeps Clang
 * (`-Winline-namespace-reopened-noninline`) and MSVC quiet.
 *
 * Version policy (see CHANGELOG.md):
 *   - `NP_VERSION_MAJOR` bump  → ABI break allowed (`v1` → `v2` rename via
 *     `git grep -l 'np::v1' include src | xargs sed -i 's/np::v1/np::v2/'`,
 *     then update `NP_ABI_VERSION` and the guard below).
 *   - `NP_VERSION_MINOR` bump  → additive only, no renames, no removals.
 *   - `NP_VERSION_PATCH` bump  → no ABI-relevant change.
 *
 * `NP_ABI_VERSION` mirrors the CMake `NP_ABI_VERSION` cache option
 * (passed as `-DNP_ABI_VERSION=N` on the `numpy-cpp` INTERFACE target);
 * the `#if` guard below keeps the header and the build in sync.
 *
 * Reference: CMake `project(numpy-cpp VERSION 1.0.0)` in `CMakeLists.txt`.
 *
 * @author Sergio Randriamihoatra (sergiorandriamihoatra@gmail.com)
 */
#ifndef NP_ABI_HPP
#define NP_ABI_HPP

// Library version — mirrors CMake `project(... VERSION 1.0.0)`.
#define NP_VERSION_MAJOR 1
#define NP_VERSION_MINOR 0
#define NP_VERSION_PATCH 0

// ABI version — selects the inline namespace `v<N>`.
#ifndef NP_ABI_VERSION
#define NP_ABI_VERSION 1
#endif

// This tree's namespaces are explicitly `np::v1::*`; keep the guard in sync
// on a major-version ABI break (see the bump procedure above).
#if NP_ABI_VERSION != 1
#error "NP_ABI_VERSION != 1 requires renaming namespace np::v1::* (see abi.hpp)."
#endif

static_assert(__cplusplus >= 202002L, "numpy-cpp requires C++20 or later");

namespace np
{
inline namespace v1
{

namespace abi
{

/// ABI version selected at compile time (`-DNP_ABI_VERSION=N`).
[[nodiscard]] inline constexpr int version() noexcept
{
    return NP_ABI_VERSION;
}
/// Library version triple.
[[nodiscard]] inline constexpr int version_major() noexcept
{
    return NP_VERSION_MAJOR;
}
[[nodiscard]] inline constexpr int version_minor() noexcept
{
    return NP_VERSION_MINOR;
}
[[nodiscard]] inline constexpr int version_patch() noexcept
{
    return NP_VERSION_PATCH;
}
/// Human-readable library + ABI version.
[[nodiscard]] inline constexpr const char *version_string() noexcept
{
    return "1.0.0 (ABI v1)";
}
/// Link-time tag, e.g. embedded in diagnostics.
[[nodiscard]] inline constexpr const char *abi_tag() noexcept
{
    return "np-abi-v1";
}

} // namespace abi

} // namespace v1
} // namespace np

#endif // NP_ABI_HPP
