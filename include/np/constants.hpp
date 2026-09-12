/**
 * @file constants.hpp
 * @brief NumPy constants (np.e, np.pi, np.inf, np.nan, ...).
 *
 * Reference: https://numpy.org/doc/2.2/reference/constants.html
 *
 * Provides C++ equivalents of NumPy's scalar constants:
 *   e, euler_gamma, inf, nan, pi, newaxis
 * All constants are `inline constexpr` (or inline const where
 * non-constexpr) so the header remains header-only.
 *
 *   np::constants::e            == 2.718281828459045...
 *   np::constants::euler_gamma  == 0.5772156649015329...
 *   np::constants::pi           == 3.141592653589793...
 *   np::constants::inf          == +infinity (double)
 *   np::constants::nan          == quiet NaN (double)
 *   np::constants::newaxis      == std::nullopt (sentinel for None)
 *
 * Top-level aliases `np::e`, `np::pi`, etc. mirror the Python
 * `numpy.<name>` names for ergonomic `np::pi` access. `newaxis`
 * is also available as `np::newaxis`.
 *
 * @author Sergio Randriamihoatra (sergiorandriamihoatra@gmail.com)
 */
#ifndef NP_CONSTANTS_HPP
#define NP_CONSTANTS_HPP

#include <cmath>
#include <limits>
#include <numbers>
#include <optional>

#include "api_macros.hpp"

namespace np
{
namespace constants
{
/**
 * @brief Euler's number e.
 * Reference: https://numpy.org/doc/2.2/reference/constants.html#numpy.e
 */
inline constexpr double e = std::numbers::e_v<double>;

/**
 * @brief Euler-Mascheroni constant gamma.
 * Reference: https://numpy.org/doc/2.2/reference/constants.html#numpy.euler_gamma
 */
inline constexpr double euler_gamma = 0.57721566490153286060651209008240243104215933593992;

/**
 * @brief Pi.
 * Reference: https://numpy.org/doc/2.2/reference/constants.html#numpy.pi
 */
inline constexpr double pi = std::numbers::pi_v<double>;

/**
 * @brief Positive infinity (IEEE 754).
 * Reference: https://numpy.org/doc/2.2/reference/constants.html#numpy.inf
 */
inline constexpr double inf = std::numeric_limits<double>::infinity();

/**
 * @brief Not a Number (quiet NaN, IEEE 754).
 * Reference: https://numpy.org/doc/2.2/reference/constants.html#numpy.nan
 */
inline constexpr double nan = std::numeric_limits<double>::quiet_NaN();

/**
 * @brief Alias for None – use as `arr[None]` placeholder.
 * NumPy: `numpy.newaxis is None` is True; we expose it as the
 * canonical "add new axis" sentinel (mirrors Python `None`).
 * Reference: https://numpy.org/doc/2.2/reference/constants.html#numpy.newaxis
 */
inline constexpr std::nullopt_t newaxis = std::nullopt;

// Additional NumPy scalar aliases (often imported as constants)
// Note: avoid `NAN`/`INF` macro clash with <cmath> (they are macros).
// Also avoid NZERO/PZERO clash with <bits/xopen_lim.h> (defines NZERO as 20).
#ifdef NZERO
#undef NZERO
#endif
#ifdef PZERO
#undef PZERO
#endif
inline constexpr double NINF = -std::numeric_limits<double>::infinity();
inline constexpr double PINF = std::numeric_limits<double>::infinity();
inline constexpr double NaN = std::numeric_limits<double>::quiet_NaN();
inline constexpr double Inf = std::numeric_limits<double>::infinity();
inline constexpr double PZERO = 0.0;
inline constexpr double NZERO = -0.0;
} // namespace constants

// Top-level mirrors – `np::pi`, `np::e`, etc. as in Python `numpy.pi`
inline constexpr double e = constants::e;
inline constexpr double euler_gamma = constants::euler_gamma;
inline constexpr double pi = constants::pi;
inline constexpr double inf = constants::inf;
inline constexpr double nan = constants::nan;
inline constexpr double NINF = constants::NINF;
inline constexpr double PINF = constants::PINF;
inline constexpr double PZERO = constants::PZERO;
inline constexpr double NZERO = constants::NZERO;
inline constexpr std::nullopt_t newaxis = constants::newaxis;

} // namespace np

#endif // NP_CONSTANTS_HPP
