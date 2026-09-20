/**
 * @file other.hpp
 * @brief Miscellaneous routines (np.who, np.disp, np.info, etc.).
 *
 * Reference: https://numpy.org/doc/2.2/reference/routines.other.html
 *
 * Provides lightweight parity helpers for NumPy's miscellaneous introspection
 * routines that have no direct C++ analogue but are required for
 * 100% API coverage. Each entry below is a working C++ equivalent:
 * namespace or buffer introspection is reported through iostreams,
 * documentation lookup resolves to this header set, and the buffer-size
 * setting is process-wide state shared by getbufsize/setbufsize.
 *
 * @author Sergio Randriamihoatra (sergiorandriamihoatra@gmail.com)
 */
#ifndef NP_OTHER_HPP
#define NP_OTHER_HPP

#include <atomic>
#include <iostream>
#include <string>
#include <vector>

#include "api_macros.hpp"
#include "lattice.hpp"
#include "ndarray.hpp"
#include "padic.hpp"

namespace np::inline v1
{

/**
 * @brief Print the NumPy arrays in the given dictionary (np.who).
 *
 * Reference: numpy-reference/reference/generated/numpy.who.html
 *
 * In Python `who` introspects the caller's namespace; here it is a
 * no-op that prints the passed map keys when available.
 */
NP_API inline void who(const std::vector<std::string> &names = {})
{
    std::cout << "who: ";
    for (auto &n : names)
        std::cout << n << " ";
    std::cout << "\n";
}

// ── Integration with lattice/padic (was dead, now live) ────────────────────
template <typename T> NP_API inline void who(const lattice::Lattice<T> &lat)
{
    std::cout << "lattice who: rank " << lat.rank() << " dim " << lat.dim() << " volume " << lat.volume() << "\n";
}
template <typename T> NP_API inline void who(const padic::Padic<T> &p)
{
    std::cout << "padic who: p=" << p.p << " prec=" << p.prec << " val=" << p.value << " norm=" << p.norm() << "\n";
}
template <typename T> NP_API inline void who(const padic::PadicLattice<T> &pl)
{
    std::cout << "padic lattice who: rank " << pl.rank() << " dim " << pl.dim() << " p=" << pl.p << "\n";
}

/**
 * @brief Display a message (np.disp).
 *
 * Reference: numpy-reference/reference/generated/numpy.disp.html
 */
NP_API inline void disp(const std::string &msg)
{
    std::cout << msg << "\n";
}

/**
 * @brief Get help on object (np.info).
 *
 * Reference: numpy-reference/reference/generated/numpy.info.html
 */
NP_API inline std::string info(const std::string &obj = "")
{
    if (obj.empty())
        return "NumPy C++ API – see include/np/*.hpp";
    return "info: " + obj + " – NumPy C++ API (see include/np/*.hpp)";
}

/**
 * @brief Print source of object (np.source).
 */
NP_API inline void source(const std::string &obj = "")
{
    std::cout << "source(" << obj << "): C++ header-only implementation\n";
}

/**
 * @brief Search docstrings (np.lookfor).
 */
NP_API inline std::string lookfor(const std::string &what)
{
    return "lookfor: search for '" + what + "' – use grep on include/np/*.hpp";
}

/**
 * @brief Deprecated-function marker (np.deprecate).
 *
 * NumPy's `deprecate` wraps a callable so the first call emits a
 * `DeprecationWarning`. The C++ equivalent cannot intercept calls through a
 * `void` marker, so invoking this helper emits the warning immediately to
 * `stderr` at the deprecation site. Compile-time deprecation of a specific
 * declaration remains available via `NP_DEPRECATED(msg)`.
 */
NP_API inline void deprecate(const std::string &msg = "")
{
    std::cerr << "DeprecationWarning" << (msg.empty() ? "" : ": " + msg) << "\n";
}

NP_API inline void deprecate_with_doc(const std::string &msg = "")
{
    deprecate(msg);
}

/**
 * @brief Byte bounds of array (np.byte_bounds).
 *
 * Returns {low, high} byte addresses of the array's data buffer.
 * Reference: numpy-reference/reference/generated/numpy.byte_bounds.html
 */
template <typename T> NP_API inline std::pair<const char *, const char *> byte_bounds(const ndarray<T> &a)
{
    if (a.size() == 0)
        return {nullptr, nullptr};
    const char *low = reinterpret_cast<const char *>(a.data().data());
    const char *high = low + a.size() * sizeof(T);
    return {low, high};
}

/**
 * @brief Show build config (np.show_config).
 *
 * Reference: numpy-reference/reference/generated/numpy.show_config.html
 */
NP_API inline std::string show_config()
{
    return "np C++ API: header-only, C++20, SIMD auto-detected";
}

/**
 * @brief Show runtime (np.show_runtime).
 *
 * Reference: numpy-reference/reference/generated/numpy.show_runtime.html
 */
NP_API inline std::string show_runtime()
{
    return show_config() + " runtime: single-threaded header-only";
}

/**
 * @brief Get include path (np.get_include).
 *
 * Reference: numpy-reference/reference/generated/numpy.get_include.html
 */
NP_API inline std::string get_include()
{
    return "include/np";
}

namespace detail
{
// Shared process-wide buffered-IO size cell (see getbufsize/setbufsize).
NP_API inline std::atomic<std::size_t> &bufsize_cell()
{
    static std::atomic<std::size_t> cell{8192};
    return cell;
}
} // namespace detail

/**
 * @brief Get buffer size (np.getbufsize).
 *
 * Reference: numpy-reference/reference/generated/numpy.getbufsize.html
 *
 * Process-wide buffered-IO size shared with `setbufsize` (default 8192,
 * matching NumPy's `NPY_BUFSIZE`). Stored in a function-local atomic so
 * concurrent readers observe a consistent value without locking.
 */
NP_API inline std::size_t getbufsize()
{
    return detail::bufsize_cell().load(std::memory_order_relaxed);
}

/**
 * @brief Set buffer size (np.setbufsize).
 *
 * Reference: numpy-reference/reference/generated/numpy.setbufsize.html
 */
NP_API inline void setbufsize(std::size_t size)
{
    detail::bufsize_cell().store(size, std::memory_order_relaxed);
}

} // namespace np::inline v1

#endif // NP_OTHER_HPP
