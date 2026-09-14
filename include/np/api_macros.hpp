/**
 * @file api_macros.hpp
 * @brief API visibility and documentation macros for NumPy C++ library.
 *
 * Defines macros to mark API functions as public, internal, or private.
 * These are primarily for documentation and code organization.
 *
 * @author Sergio Randriamihoatra (sergiorandriamihoatra@gmail.com)
 */
#ifndef NP_API_MACROS_HPP
#define NP_API_MACROS_HPP

// ABI version + inline namespace pre-declaration. Every subsystem header
// includes this file before any `namespace np::inline v1...` reopening.
#include "abi.hpp"

/**
 * @def NP_API
 * @brief Marks a function/class as part of the public API.
 *
 * Public API functions are:
 * - Documented in API reference
 * - Stable across minor versions
 * - Safe for end users to call
 *
 * Example:
 * @code
 * NP_API template <typename T>
 * auto sum(const ndarray<T>& arr) -> T;
 * @endcode
 */
#define NP_API

/**
 * @def NP_INTERNAL
 * @brief Marks a function as internal API (exposed but not for direct use).
 *
 * Internal API functions are:
 * - In public headers (template implementations, helpers)
 * - Not documented in user-facing API reference
 * - May change without notice
 * - Used by other library functions
 *
 * Use for:
 * - Template helpers that must be in headers
 * - Validation functions shared across modules
 * - Implementation details exposed for inlining
 *
 * Example:
 * @code
 * NP_INTERNAL inline auto validate_axis(int axis, int ndim) -> int;
 * @endcode
 */
#define NP_INTERNAL

/**
 * @def NP_PRIVATE
 * @brief Marks implementation details (should be in detail:: namespace or
 * .cpp).
 *
 * Private functions are:
 * - Implementation-only
 * - Should be in detail:: namespace or anonymous namespace
 * - Not callable from outside the module
 *
 * Example:
 * @code
 * namespace detail {
 *     NP_PRIVATE auto compute_kernel(...) -> void;
 * }
 * @endcode
 */
#define NP_PRIVATE

/**
 * @def NP_DEPRECATED(msg)
 * @brief Marks a function as deprecated with a message.
 *
 * Example:
 * @code
 * NP_DEPRECATED("Use sum() instead")
 * auto array_sum(...);
 * @endcode
 */
#if defined(__GNUC__) || defined(__clang__)
#define NP_DEPRECATED(msg) [[deprecated(msg)]]
#elif defined(_MSC_VER)
#define NP_DEPRECATED(msg) __declspec(deprecated(msg))
#else
#define NP_DEPRECATED(msg)
#endif

/**
 * @def NP_NODISCARD
 * @brief Warns if function result is not used.
 *
 * Use for:
 * - Pure functions (no side effects)
 * - Functions returning new arrays
 *
 * Example:
 * @code
 * NP_NODISCARD auto copy() const -> ndarray<T>;
 * @endcode
 */
#if __cplusplus >= 201703L
#define NP_NODISCARD [[nodiscard]]
#elif defined(__GNUC__) || defined(__clang__)
#define NP_NODISCARD __attribute__((warn_unused_result))
#elif defined(_MSC_VER)
#define NP_NODISCARD _Check_return_
#else
#define NP_NODISCARD
#endif

/**
 * @def NP_CONSTEXPR
 * @brief Marks functions that are constexpr-capable.
 *
 * Used for functions that work at compile-time when possible.
 */
#define NP_CONSTEXPR constexpr

/**
 * @def NP_INLINE
 * @brief Strong inline hint for performance-critical functions.
 */
#if defined(__GNUC__) || defined(__clang__)
#define NP_INLINE inline __attribute__((always_inline))
#elif defined(_MSC_VER)
#define NP_INLINE __forceinline
#else
#define NP_INLINE inline
#endif // __GNUC__

#if defined(__GNUC__) || defined(__clang__)
#define NP_SYMBOL_VISIBILITY(V) __attribute__((__visibility__(#V)))
#else
#define NP_SYMBOL_VISIBILITY(V)
#endif // __GNUC__ or __clang__

#if defined(__GNUC__) || defined(__clang__)
#define NP_HIDDEN NP_SYMBOL_VISIBILITY(hidden)
#define NP_VISIBLE NP_SYMBOL_VISIBILITY(default)
#else
#define NP_HIDDEN
#define NP_VISIBLE
#endif

// Convenience alias requested by task
#define NP_VISIBILITY(V) NP_SYMBOL_VISIBILITY(V)
#endif // NP_API_MACROS_HPP
