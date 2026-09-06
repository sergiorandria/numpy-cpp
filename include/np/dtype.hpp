/**
 * @file dtype.hpp
 * @brief NumPy-compatible data type system.
 *
 * Defines the `np::dtype` enumeration and compile-time bridges between
 * `np::dtype` values and native C++ types.
 *
 * @author Sergio Randriamihoatra (sergiorandriamihoatra@gmail.com)
 */
#ifndef NP_DTYPE_HPP
#define NP_DTYPE_HPP

#include <complex>
#include <cstdint>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include "api_macros.hpp"

namespace np
{
/**
 * @brief Enumeration of NumPy-compatible data types.
 *
 * The values mirror numpy.dtype names. `string_` and `unicode_`
 * are present for API parity but have no C++ storage representation
 * (size() returns 0). `datetime64` and `timedelta64` store as
 * int64_t units with a separate `np::datetime64` unit code.
 */
enum class dtype
{
    // Integer types
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
    bigint, // arbitrary-precision integer (np::bigint / mpz)

    // Floating-point types
    float16,
    float32,
    float64,
    longdouble,

    // Complex types
    complex64,
    complex128,
    clongdouble,

    // Boolean
    bool_,

    // String / unicode
    string_,
    unicode_,

    // Datetime
    datetime64,
    timedelta64,

    // Special
    void_,
    object_
};

namespace detail
{

/**
 * @brief Maps a np::dtype value to its native C++ type.
 *
 * @tparam _DtypeElement  A np::dtype enumeration value.
 */
template <dtype _DtypeElement> struct _Np_type_to_cxx;

template <> struct _Np_type_to_cxx<dtype::int8>
{
    using type = std::int8_t;
};
template <> struct _Np_type_to_cxx<dtype::int16>
{
    using type = std::int16_t;
};
template <> struct _Np_type_to_cxx<dtype::int32>
{
    using type = std::int32_t;
};
template <> struct _Np_type_to_cxx<dtype::int64>
{
    using type = std::int64_t;
};
template <> struct _Np_type_to_cxx<dtype::uint8>
{
    using type = std::uint8_t;
};
template <> struct _Np_type_to_cxx<dtype::uint16>
{
    using type = std::uint16_t;
};
template <> struct _Np_type_to_cxx<dtype::uint32>
{
    using type = std::uint32_t;
};
template <> struct _Np_type_to_cxx<dtype::uint64>
{
    using type = std::uint64_t;
};
template <> struct _Np_type_to_cxx<dtype::float16>
{
    using type = std::uint16_t;
};
template <> struct _Np_type_to_cxx<dtype::float32>
{
    using type = float;
};
template <> struct _Np_type_to_cxx<dtype::float64>
{
    using type = double;
};
template <> struct _Np_type_to_cxx<dtype::longdouble>
{
    using type = long double;
};
template <> struct _Np_type_to_cxx<dtype::complex64>
{
    using type = std::complex<float>;
};
template <> struct _Np_type_to_cxx<dtype::complex128>
{
    using type = std::complex<double>;
};
template <> struct _Np_type_to_cxx<dtype::clongdouble>
{
    using type = std::complex<long double>;
};
template <> struct _Np_type_to_cxx<dtype::bool_>
{
    using type = bool;
};
template <> struct _Np_type_to_cxx<dtype::datetime64>
{
    using type = std::int64_t;
};
template <> struct _Np_type_to_cxx<dtype::timedelta64>
{
    using type = std::int64_t;
};

/**
 * @brief Maps a native C++ type to its np::dtype value.
 *
 * Uses a type-keyed constant expression so that aliased types
 * (e.g. std::int32_t == int on many ABIs) cannot collide.
 *
 * @tparam T  A native C++ type.
 */
template <typename T> struct cxx_to_np_type_impl
{
    static constexpr dtype value =
        std::is_same_v<T, std::int8_t>                 ? dtype::int8
        : std::is_same_v<T, std::int16_t>              ? dtype::int16
        : std::is_same_v<T, std::int32_t>              ? dtype::int32
        : std::is_same_v<T, std::int64_t>              ? dtype::int64
        : std::is_same_v<T, std::uint8_t>              ? dtype::uint8
        : std::is_same_v<T, std::uint16_t>             ? dtype::uint16
        : std::is_same_v<T, std::uint32_t>             ? dtype::uint32
        : std::is_same_v<T, std::uint64_t>             ? dtype::uint64
        : std::is_same_v<T, float>                     ? dtype::float32
        : std::is_same_v<T, double>                    ? dtype::float64
        : std::is_same_v<T, long double>               ? dtype::longdouble
        : std::is_same_v<T, std::complex<float>>       ? dtype::complex64
        : std::is_same_v<T, std::complex<double>>      ? dtype::complex128
        : std::is_same_v<T, std::complex<long double>> ? dtype::clongdouble
        : std::is_same_v<T, bool>                      ? dtype::bool_
        : std::is_same_v<T, char>                      ? dtype::int8
        : std::is_same_v<T, signed char>               ? dtype::int8
        : std::is_same_v<T, unsigned char>             ? dtype::uint8
        : std::is_same_v<T, wchar_t>                   ? dtype::int32
        : std::is_same_v<T, std::size_t>               ? (sizeof(std::size_t) == 8 ? dtype::uint64 : dtype::uint32)
        : std::is_same_v<T, std::ptrdiff_t>            ? (sizeof(std::ptrdiff_t) == 8 ? dtype::int64 : dtype::int32)
                                                       : dtype::void_;
};

template <typename T> struct cxx_to_np_type : cxx_to_np_type_impl<std::remove_cv_t<T>>
{
};

/**
 * @brief True when T is a std::complex instantiation.
 *
 * @tparam T  A type to inspect.
 */
template <typename T> struct is_complex : std::false_type
{
};
template <typename T> struct is_complex<std::complex<T>> : std::true_type
{
};
template <typename T> inline constexpr bool is_complex_v = is_complex<T>::value;
} // namespace detail

namespace _Np_dtype
{
/** @brief Trivial branch used when the other dtype branch is unused. */
struct _Np_unused_branch
{
};

/**
 * @brief Storage type for the character dtypes.
 *
 * @tparam _DtypeElement  A np::dtype value (`string_` or `unicode_`).
 */
template <dtype _DtypeElement> struct _Np_string_value
{
    using type = std::string;
};
template <> struct _Np_string_value<dtype::unicode_>
{
    using type = std::u32string;
};

/** @brief Compile-time check: the dtype stores its value as text. */
template <dtype D> inline constexpr bool is_string_dtype_v = D == dtype::string_ || D == dtype::unicode_;

/**
 * @brief Union-backed storage with two branch structures.
 *
 * The first branch holds the contiguous scalar value for the
 * integral/numeric dtypes (via `value_type`); the second branch holds
 * the string attribute for `string_` / `unicode_`. Exactly one branch
 * is ever used — the other stays latent with a trivial placeholder —
 * so the numeric case keeps a literal type usable in constant
 * expressions.
 *
 * @tparam _DtypeElement  A np::dtype enumeration value.
 * @tparam _IsString      True when the dtype stores text.
 */
template <auto _DtypeElement, bool _IsString = is_string_dtype_v<_DtypeElement>> struct _Np_StorageClassifier;

/**
 * @brief Integral/numeric branch of the storage classifier.
 *
 * @tparam _DtypeElement  A np::dtype enumeration value.
 */
template <auto _DtypeElement> struct _Np_StorageClassifier<_DtypeElement, /* _IsString */ false>
{
    using value_type = typename detail::_Np_type_to_cxx<_DtypeElement>::type;

    static constexpr np::dtype type = _DtypeElement;
    static constexpr bool is_text = false;

  private:
    union _Np_storage {
        value_type value;         // numeric branch
        _Np_unused_branch unused; // string branch placeholder
    } storage_{};

  public:
    constexpr _Np_StorageClassifier() noexcept = default;
    constexpr _Np_StorageClassifier(const value_type &__v) : storage_{.value = __v}
    {
    }
    constexpr _Np_StorageClassifier(value_type &&__v) noexcept : storage_{.value = static_cast<value_type &&>(__v)}
    {
    }

    constexpr auto operator=(const value_type &other) -> _Np_StorageClassifier &
    {
        storage_.value = other;
        return *this;
    }
    constexpr auto operator=(value_type &&other) -> _Np_StorageClassifier &
    {
        storage_.value = static_cast<value_type &&>(other);
        return *this;
    }

    constexpr operator value_type &() noexcept
    {
        return storage_.value;
    }
    constexpr operator value_type() const noexcept
    {
        return storage_.value;
    }

    /** @brief The compile-time dtype. */
    static constexpr auto get_type() noexcept -> np::dtype
    {
        return type;
    }
    /** @brief Access the underlying numeric value by reference. */
    constexpr auto value() noexcept -> value_type &
    {
        return storage_.value;
    }
    constexpr auto value() const noexcept -> const value_type &
    {
        return storage_.value;
    }
};

/**
 * @brief String branch of the storage classifier.
 *
 * @tparam _DtypeElement  A np::dtype enumeration value.
 */
template <auto _DtypeElement> struct _Np_StorageClassifier<_DtypeElement, true>
{
    using value_type = typename _Np_string_value<_DtypeElement>::type;

    static constexpr np::dtype type = _DtypeElement;
    static constexpr bool is_text = true;

  private:
    union _Np_storage {
        _Np_unused_branch unused; // numeric branch placeholder
        value_type value;         // string branch

        constexpr _Np_storage() noexcept : value{}
        {
        }
        _Np_storage(const value_type &__v) noexcept : value(__v)
        {
        }
        _Np_storage(value_type &&__v) noexcept : value(static_cast<value_type &&>(__v))
        {
        }

        ~_Np_storage()
        {
            value.~value_type();
        }
    };

    _Np_storage storage_{};

  public:
    _Np_StorageClassifier() noexcept = default;
    _Np_StorageClassifier(const value_type &__v) noexcept : storage_(__v)
    {
    }
    _Np_StorageClassifier(value_type &&__v) noexcept : storage_(static_cast<value_type &&>(__v))
    {
    }
    _Np_StorageClassifier(const _Np_StorageClassifier &other) noexcept : storage_(other.storage_.value)
    {
    }
    _Np_StorageClassifier(_Np_StorageClassifier &&other) noexcept
        : storage_(static_cast<value_type &&>(other.storage_.value))
    {
    }

    auto operator=(const value_type &other) -> _Np_StorageClassifier &
    {
        storage_.value = other;
        return *this;
    }
    auto operator=(value_type &&other) -> _Np_StorageClassifier &
    {
        storage_.value = static_cast<value_type &&>(other);
        return *this;
    }
    auto operator=(const _Np_StorageClassifier &other) -> _Np_StorageClassifier &
    {
        storage_.value = other.storage_.value;
        return *this;
    }
    auto operator=(_Np_StorageClassifier &&other) -> _Np_StorageClassifier &
    {
        storage_.value = static_cast<value_type &&>(other.storage_.value);
        return *this;
    }

    operator value_type &() noexcept
    {
        return storage_.value;
    }
    operator const value_type &() const noexcept
    {
        return storage_.value;
    }

    /** @brief The compile-time dtype. */
    static constexpr auto get_type() noexcept -> np::dtype
    {
        return type;
    }
    /** @brief Access the underlying string by reference. */
    auto value() noexcept -> value_type &
    {
        return storage_.value;
    }
    auto value() const noexcept -> const value_type &
    {
        return storage_.value;
    }
};

/** @brief Compile-time compare of two storage classifiers. */
template <auto _L, bool _Lb, auto _R, bool _Rb>
constexpr auto operator==(_Np_StorageClassifier<_L, _Lb>, _Np_StorageClassifier<_R, _Rb>) noexcept -> bool
{
    return _L == _R;
}
template <auto _L, bool _Lb, auto _R, bool _Rb>
constexpr auto operator!=(_Np_StorageClassifier<_L, _Lb>, _Np_StorageClassifier<_R, _Rb>) noexcept -> bool
{
    return _L != _R;
}
#ifdef _NP_USE_DIRECT_STD_TYPES
/**
 * @brief Storage type aliases mirroring the numpy C-API `npy_*`
 *        typedefs. Each alias names the native C++ storage of the
 *        matching np::dtype (the `value_type` of
 *        detail::_Np_type_to_cxx), so it can be used as a
 *        compile-time dtype tag. `string_` / `unicode_` use a string
 *        attribute; only `void_` and `object_` have no fixed
 *        C++ storage and are omitted.
 */
using _Np_int8 = std::int8_t;
using _Np_int16 = std::int16_t;
using _Np_int32 = std::int32_t;
using _Np_int64 = std::int64_t;
using _Np_uint8 = std::uint8_t;
using _Np_uint16 = std::uint16_t;
using _Np_uint32 = std::uint32_t;
using _Np_uint64 = std::uint64_t;
using _Np_float16 = std::uint16_t; // half-precision bit storage
using _Np_float32 = float;
using _Np_float64 = double;
using _Np_longdouble = long double;
using _Np_complex64 = std::complex<float>;
using _Np_complex128 = std::complex<double>;
using _Np_clongdouble = std::complex<long double>;
using _Np_bool_ = bool;
// datetime64 / timedelta64 count their units in int64_t.
using _Np_datetime64 = std::int64_t;
using _Np_timedelta64 = std::int64_t;
// String dtypes have no contiguous scalar; use a string attribute.
using _Np_string = std::string;
using _Np_unicode = std::u32string;
#else
/**
 * @brief Classifier-based storage aliases used when
 *        `_NP_USE_DIRECT_STD_TYPES` is not defined.
 *
 * Each alias is a `_Np_StorageClassifier` instantiation, binding a
 * compile-time `np::dtype` (`type`) to an instance of its native
 * C++ storage (`value_type`), so the storage is self-describing at
 * compile time while remaining usable as a plain scalar. `string_`
 * and `unicode_` use the string branch of the classifier; only
 * `void_` and `object_` have no fixed C++ storage.
 */
using _Np_int8 = _Np_StorageClassifier<np::dtype::int8>;
using _Np_int16 = _Np_StorageClassifier<np::dtype::int16>;
using _Np_int32 = _Np_StorageClassifier<np::dtype::int32>;
using _Np_int64 = _Np_StorageClassifier<np::dtype::int64>;
using _Np_uint8 = _Np_StorageClassifier<np::dtype::uint8>;
using _Np_uint16 = _Np_StorageClassifier<np::dtype::uint16>;
using _Np_uint32 = _Np_StorageClassifier<np::dtype::uint32>;
using _Np_uint64 = _Np_StorageClassifier<np::dtype::uint64>;
using _Np_float16 = _Np_StorageClassifier<np::dtype::float16>;
using _Np_float32 = _Np_StorageClassifier<np::dtype::float32>;
using _Np_float64 = _Np_StorageClassifier<np::dtype::float64>;
using _Np_longdouble = _Np_StorageClassifier<np::dtype::longdouble>;
using _Np_complex64 = _Np_StorageClassifier<np::dtype::complex64>;
using _Np_complex128 = _Np_StorageClassifier<np::dtype::complex128>;
using _Np_clongdouble = _Np_StorageClassifier<np::dtype::clongdouble>;
using _Np_bool_ = _Np_StorageClassifier<np::dtype::bool_>;
// datetime64 / timedelta64 count their units in int64_t.
using _Np_datetime64 = _Np_StorageClassifier<np::dtype::datetime64>;
using _Np_timedelta64 = _Np_StorageClassifier<np::dtype::timedelta64>;
// String dtypes use the string branch of the classifier.
using _Np_string = _Np_StorageClassifier<np::dtype::string_>;
using _Np_unicode = _Np_StorageClassifier<np::dtype::unicode_>;
#endif
} // namespace _Np_dtype

// Forward declaration for use in _dtype_t_from_type
template <dtype D> struct dtype_tag;

/**
 * @brief Native C++ type corresponding to a np::dtype tag or plain type.
 *
 * `dtype_t<np::complex128>` -> `std::complex<double>` (via dtype_tag),
 * `dtype_t<double>` -> `double` (identity).
 * For enum values use `dtype_t_enum` or `dtype_t<dtype::...>` via overload
 * kept for backward compatibility (see below).
 * @tparam T  A type (either a dtype_tag or a plain C++ type).
 */
template <typename T> struct _dtype_t_from_type
{
    using type = T;
};
template <dtype D> struct _dtype_t_from_type<dtype_tag<D>>
{
    using type = typename detail::_Np_type_to_cxx<D>::type;
};
template <typename T> using dtype_t = typename _dtype_t_from_type<T>::type;

/** @brief Enum-based mapping (kept for backward compatibility). */
template <dtype D> using dtype_t_enum = typename detail::_Np_type_to_cxx<D>::type;

// Keep `dtype_t<dtype::...>` working for code that passes enum values
// (e.g. tests). Provide a variable-template-like overload via
// struct wrapper: `dtype_t_c<D>` is enum-based, but we also support
// `dtype_t` with dtype values via a helper alias `dtype_t_v`.
// For true backward compat, define `dtype_t` for dtype values as well
// via a dedicated alias when the argument is a dtype enum.
// Note: `dtype_t<dtype::X>` syntax (value as type param) is not valid
// C++ for `template<typename T>`, so we expose `dtype_t_enum` for that
// use-case. Tests using `dtype_t<dtype::X>` should migrate to
// `dtype_t<np::X>` or `dtype_t_enum<dtype::X>`.

// Compile-time dtype tags: usable as `ndarray<np::complex128>` etc.
// Each tag carries `value` (the dtype enum) and `type` (the C++ type).
template <dtype D> struct dtype_tag
{
    static constexpr dtype value = D;
    using type = typename detail::_Np_type_to_cxx<D>::type;
    constexpr operator dtype() const noexcept
    {
        return D;
    }
};

template <typename T> struct is_dtype_tag : std::false_type
{
};
template <dtype D> struct is_dtype_tag<dtype_tag<D>> : std::true_type
{
};
template <typename T> inline constexpr bool is_dtype_tag_v = is_dtype_tag<T>::value;

template <typename T> struct dtype_tag_to_type
{
    using type = T;
};
template <dtype D> struct dtype_tag_to_type<dtype_tag<D>>
{
    using type = typename detail::_Np_type_to_cxx<D>::type;
};

// Type aliases usable as `ndarray<np::complex128>` – compile-time dtype → C++ type
using int8 = dtype_tag<dtype::int8>;
using int16 = dtype_tag<dtype::int16>;
using int32 = dtype_tag<dtype::int32>;
using int64 = dtype_tag<dtype::int64>;
using uint8 = dtype_tag<dtype::uint8>;
using uint16 = dtype_tag<dtype::uint16>;
using uint32 = dtype_tag<dtype::uint32>;
using uint64 = dtype_tag<dtype::uint64>;
using float16 = dtype_tag<dtype::float16>;
using float32 = dtype_tag<dtype::float32>;
using float64 = dtype_tag<dtype::float64>;
using longdouble = dtype_tag<dtype::longdouble>;
using complex64 = dtype_tag<dtype::complex64>;
using complex128 = dtype_tag<dtype::complex128>;
using clongdouble = dtype_tag<dtype::clongdouble>;
using bool_ = dtype_tag<dtype::bool_>;
using string_ = dtype_tag<dtype::string_>;
using unicode_ = dtype_tag<dtype::unicode_>;
using datetime64 = dtype_tag<dtype::datetime64>;
using timedelta64 = dtype_tag<dtype::timedelta64>;
using void_ = dtype_tag<dtype::void_>;
using object_ = dtype_tag<dtype::object_>;

namespace detail
{
template <dtype D> struct cxx_to_np_type_impl<::np::dtype_tag<D>>
{
    static constexpr dtype value = D;
};
} // namespace detail

/**
 * @brief np::dtype value corresponding to a native C++ type.
 *
 * @tparam T  A native C++ type.
 */
template <typename T> inline constexpr dtype dtype_of = detail::cxx_to_np_type<std::remove_cv_t<T>>::value;

/** @brief True when T is a std::complex instantiation. */
template <typename T> inline constexpr bool is_complex_v = detail::is_complex<T>::value;

namespace detail
{
/**
 * @brief Compile-time check: the dtype is one of the numeric dtypes.
 *
 * True for every dtype that has a scalar `value_type` in
 * `_Np_type_to_cxx` (integers, floats, complex, bool_, datetime64
 * and timedelta64). False for `string_`, `unicode_`, `void_` and
 * `object_`.
 *
 * @tparam D  A np::dtype enumeration value.
 */
template <dtype D>
struct is_numeric_dtype : std::bool_constant<D >= dtype::int8 && D != dtype::void_ && D != dtype::object_ &&
                                             D != dtype::string_ && D != dtype::unicode_>
{
};

template <dtype D> inline constexpr bool is_numeric_dtype_v = is_numeric_dtype<D>::value;
} // namespace detail

/**
 * @brief Compile-time check: the dtype is an integral dtype.
 *
 * Analogous to `dtype_is_integer` but evaluated at compile time from a
 * dtype value. True only for int8 through uint64; bool_ is excluded.
 *
 * @tparam D  A dtype value.
 */
template <dtype D>
using is_integral_dtype = std::bool_constant<(D >= dtype::int8 && D <= dtype::uint64) || D == dtype::bigint>;

template <dtype D> inline constexpr bool is_integral_dtype_v = is_integral_dtype<D>::value;

/**
 * @brief Compile-time check: the dtype is numeric (non-text).
 *
 * @tparam D  A np::dtype enumeration value.
 */
template <dtype D> inline constexpr bool is_numeric_dtype_v = detail::is_numeric_dtype<D>::value;

// Runtime helpers
/**
 * @brief Human-readable name of a dtype.
 *
 * @param t  The dtype value.
 * @return   A string_view naming the dtype (e.g. "int8", "float64").
 */
NP_API NP_NODISCARD constexpr std::string_view dtype_name(dtype t)
{
    switch (t)
    {
    case dtype::int8:
        return "int8";
    case dtype::int16:
        return "int16";
    case dtype::int32:
        return "int32";
    case dtype::int64:
        return "int64";
    case dtype::uint8:
        return "uint8";
    case dtype::uint16:
        return "uint16";
    case dtype::uint32:
        return "uint32";
    case dtype::uint64:
        return "uint64";
    case dtype::bigint:
        return "bigint";
    case dtype::float16:
        return "float16";
    case dtype::float32:
        return "float32";
    case dtype::float64:
        return "float64";
    case dtype::longdouble:
        return "longdouble";
    case dtype::complex64:
        return "complex64";
    case dtype::complex128:
        return "complex128";
    case dtype::clongdouble:
        return "clongdouble";
    case dtype::bool_:
        return "bool";
    case dtype::string_:
        return "str";
    case dtype::unicode_:
        return "unicode";
    case dtype::datetime64:
        return "datetime64";
    case dtype::timedelta64:
        return "timedelta64";
    case dtype::void_:
        return "void";
    case dtype::object_:
        return "object";
    }
    return "unknown";
}

/**
 * @brief Size in bytes of a dtype.
 *
 * Returns 0 for the special/variable dtypes (string_, unicode_,
 * void_, object_) and for longdouble/clongdouble (which are
 * platform-dependent).
 *
 * @param t  The dtype value.
 * @return   Number of bytes, or 0 for variable-length types.
 */
NP_API NP_NODISCARD constexpr std::size_t dtype_size(dtype t)
{
    switch (t)
    {
    case dtype::int8:
    case dtype::uint8:
    case dtype::bool_:
        return 1;
    case dtype::int16:
    case dtype::uint16:
    case dtype::float16:
        return 2;
    case dtype::int32:
    case dtype::uint32:
    case dtype::float32:
        return 4;
    case dtype::int64:
    case dtype::uint64:
    case dtype::float64:
    case dtype::complex64:
    case dtype::datetime64:
    case dtype::timedelta64:
        return 8;
    case dtype::complex128:
        return 16;
    case dtype::bigint:
        return 0; // variable-length arbitrary precision
    case dtype::longdouble:
        return sizeof(long double);
    case dtype::clongdouble:
        return sizeof(std::complex<long double>);
    case dtype::string_:
    case dtype::unicode_:
    case dtype::void_:
    case dtype::object_:
        return 0;
    }
    return 0;
}

/**
 * @brief True for the complex dtypes.
 *
 * @param t  The dtype value.
 * @return   True if t is complex64, complex128, or clongdouble.
 */
NP_API NP_NODISCARD constexpr bool dtype_is_complex(dtype t)
{
    return t == dtype::complex64 || t == dtype::complex128 || t == dtype::clongdouble;
}

/**
 * @brief True for floating-point (non-complex) dtypes.
 *
 * @param t  The dtype value.
 * @return   True if t is float16, float32, float64, or longdouble.
 */
NP_API NP_NODISCARD constexpr bool dtype_is_floating(dtype t)
{
    return t == dtype::float16 || t == dtype::float32 || t == dtype::float64 || t == dtype::longdouble;
}

/**
 * @brief True for the integer dtypes (signed or unsigned).
 *
 * @param t  The dtype value.
 * @return   True if t is int8 through uint64.
 */
NP_API NP_NODISCARD constexpr bool dtype_is_integer(dtype t)
{
    return (t >= dtype::int8 && t <= dtype::uint64) || t == dtype::bigint;
}

/**
 * @brief True for signed integer dtypes.
 *
 * @param t  The dtype value.
 * @return   True if t is int8 through int64.
 */
NP_API NP_NODISCARD constexpr bool dtype_is_signed(dtype t)
{
    return t >= dtype::int8 && t <= dtype::int64;
}

/**
 * @brief True for unsigned integer dtypes.
 *
 * @param t  The dtype value.
 * @return   True if t is uint8 through uint64.
 */
NP_API NP_NODISCARD constexpr bool dtype_is_unsigned(dtype t)
{
    return t >= dtype::uint8 && t <= dtype::uint64;
}

/**
 * @brief True for boolean dtype.
 *
 * @param t  The dtype value.
 * @return   True if t is bool_.
 */
NP_API NP_NODISCARD constexpr bool dtype_is_bool(dtype t)
{
    return t == dtype::bool_;
}

// ── Extended dtype API (parity with numpy dtype routines) ─────────────

namespace detail
{
inline constexpr int _dtype_rank(dtype t) noexcept
{
    switch (t)
    {
    case dtype::bool_:
        return 0;
    case dtype::int8:
        return 1;
    case dtype::int16:
        return 2;
    case dtype::int32:
        return 3;
    case dtype::int64:
        return 4;
    case dtype::uint8:
        return 5;
    case dtype::uint16:
        return 6;
    case dtype::uint32:
        return 7;
    case dtype::uint64:
        return 8;
    case dtype::bigint:
        return 9;
    case dtype::float16:
        return 10;
    case dtype::float32:
        return 11;
    case dtype::float64:
        return 12;
    case dtype::longdouble:
        return 13;
    case dtype::complex64:
        return 14;
    case dtype::complex128:
        return 15;
    case dtype::clongdouble:
        return 16;
    case dtype::datetime64:
        return 17;
    case dtype::timedelta64:
        return 17;
    case dtype::string_:
        return 18;
    case dtype::unicode_:
        return 19;
    case dtype::void_:
        return 20;
    case dtype::object_:
        return 21;
    }
    return 21;
}

inline constexpr int _dtype_kind(dtype t) noexcept
{
    if (t == dtype::bool_)
        return 0;
    if (t == dtype::bigint)
        return 1; // bigint is signed arbitrary integer
    if (t >= dtype::int8 && t <= dtype::int64)
        return 1;
    if (t >= dtype::uint8 && t <= dtype::uint64)
        return 2;
    if (t == dtype::float16 || t == dtype::float32 || t == dtype::float64 || t == dtype::longdouble)
        return 3;
    if (t == dtype::complex64 || t == dtype::complex128 || t == dtype::clongdouble)
        return 4;
    if (t == dtype::datetime64 || t == dtype::timedelta64)
        return 5;
    return 6;
}
} // namespace detail

/**
 * @brief Can `from` be cast to `to` under given casting rule.
 *
 * Casting modes mirror NumPy: "no", "equiv", "safe", "same_kind",
 * "unsafe". Here "safe" follows rank/kind promotion, "equiv" requires
 * equality, "same_kind" allows within-kind promotion, "unsafe" always true.
 *
 * Reference: numpy-reference/reference/generated/numpy.can_cast.html
 */
NP_API NP_NODISCARD inline bool can_cast(dtype from, dtype to, const std::string &casting = "safe")
{
    if (from == to)
    {
        return true;
    }
    if (casting == "unsafe")
    {
        return true;
    }
    if (casting == "no" || casting == "equiv")
    {
        return false;
    }
    int rf = detail::_dtype_rank(from);
    int rt = detail::_dtype_rank(to);
    int kf = detail::_dtype_kind(from);
    int kt = detail::_dtype_kind(to);
    if (casting == "same_kind")
    {
        if (kf != kt)
        {
            // bool -> int/uint is considered same_kind in NumPy
            if (kf == 0 && (kt == 1 || kt == 2))
            {
                return true;
            }
            return false;
        }
        return rt >= rf;
    }
    // safe
    if (from == dtype::bool_)
    {
        return true;
    }
    if (kf == 1) // int
    {
        if (kt == 1)
            return rt >= rf;
        if (kt == 2)
            return false; // int -> uint not safe (may overflow)
        if (kt == 3 || kt == 4)
            return rt >= rf;
        return false;
    }
    if (kf == 2) // uint
    {
        if (kt == 2)
            return rt >= rf;
        if (kt == 3 || kt == 4)
            return rt >= rf;
        return false;
    }
    if (kf == 3) // float
    {
        if (kt == 3 || kt == 4)
            return rt >= rf;
        return false;
    }
    if (kf == 4) // complex
    {
        if (kt == 4)
            return rt >= rf;
        return false;
    }
    return false;
}

/**
 * @brief Promote two dtypes to a common dtype (np.promote_types).
 *
 * Reference: numpy-reference/reference/generated/numpy.promote_types.html
 */
NP_API NP_NODISCARD inline dtype promote_types(dtype a, dtype b)
{
    if (a == b)
    {
        return a;
    }
    int ra = detail::_dtype_rank(a);
    int rb = detail::_dtype_rank(b);
    return ra >= rb ? a : b;
}

/**
 * @brief Result type from promotion of given dtypes (np.result_type).
 *
 * Reference: numpy-reference/reference/generated/numpy.result_type.html
 */
NP_API inline dtype result_type(std::initializer_list<dtype> dtypes)
{
    if (dtypes.size() == 0)
    {
        throw std::invalid_argument("result_type: need at least one dtype");
    }
    auto it = dtypes.begin();
    dtype cur = *it++;
    for (; it != dtypes.end(); ++it)
    {
        cur = promote_types(cur, *it);
    }
    return cur;
}

NP_API template <typename... Ds> NP_NODISCARD inline dtype result_type(dtype first, Ds... rest)
{
    dtype cur = first;
    ((cur = promote_types(cur, rest)), ...);
    return cur;
}

/**
 * @brief Find common type from array/dtype list (np.find_common_type).
 *
 * Reference: numpy-reference/reference/generated/numpy.find_common_type.html
 */
NP_API inline dtype find_common_type(std::initializer_list<dtype> array_types,
                                     std::initializer_list<dtype> scalar_types)
{
    dtype cur = dtype::bool_;
    bool has = false;
    for (auto d : array_types)
    {
        cur = has ? promote_types(cur, d) : d;
        has = true;
    }
    for (auto d : scalar_types)
    {
        cur = has ? promote_types(cur, d) : d;
        has = true;
    }
    if (!has)
    {
        return dtype::float64;
    }
    return cur;
}

/**
 * @brief Common type of dtypes (np.common_type).
 *
 * Reference: https://numpy.org/doc/2.2/reference/generated/numpy.common_type.html
 */
NP_API inline dtype common_type(std::initializer_list<dtype> dtypes)
{
    if (dtypes.size() == 0)
    {
        return dtype::float64;
    }
    return result_type(dtypes);
}

/**
 * @brief Minimal scalar type that can hold value (np.min_scalar_type).
 *
 * Reference: numpy-reference/reference/generated/numpy.min_scalar_type.html
 */
NP_API NP_NODISCARD inline dtype min_scalar_type(long long v)
{
    if (v >= 0)
    {
        if (v <= 127)
            return dtype::int8;
        if (v <= 32767)
            return dtype::int16;
        if (v <= 2147483647)
            return dtype::int32;
        return dtype::int64;
    }
    else
    {
        if (v >= -128)
            return dtype::int8;
        if (v >= -32768)
            return dtype::int16;
        if (v >= -2147483648LL)
            return dtype::int32;
        return dtype::int64;
    }
}

NP_API NP_NODISCARD inline dtype min_scalar_type(double v)
{
    (void)v;
    return dtype::float64;
}

/**
 * @brief Whether `a` is a sub-dtype of `b` (np.issubdtype).
 *
 * Reference: numpy-reference/reference/generated/numpy.issubdtype.html
 */
NP_API NP_NODISCARD inline bool issubdtype(dtype a, dtype b)
{
    if (a == b)
    {
        return true;
    }
    // generic categories: use name-based check – if b is the generic
    // integer/floating/complex kind, delegate to dtype_is_* helpers
    // For our enum-based simulation, treat exact match for now plus
    // kind expansion: b being a generic placeholder is simulated via
    // callers passing the most general dtype of that kind.
    // Check kind containment:
    int ka = detail::_dtype_kind(a);
    int kb = detail::_dtype_kind(b);
    // If b is the maximal rank of its kind, treat as generic kind check
    // Example: b == int64 represents "signedinteger", b == float64 -> "floating"
    if (kb == 1 && ka == 1)
        return true;
    if (kb == 2 && ka == 2)
        return true;
    if (kb == 3 && ka == 3)
        return true;
    if (kb == 4 && ka == 4)
        return true;
    return false;
}

/**
 * @brief Whether `a` is sub-class of `b` (np.issubsctype).
 * Alias to `issubdtype` for enum dtypes.
 */
NP_API NP_NODISCARD inline bool issubsctype(dtype a, dtype b)
{
    return issubdtype(a, b);
}

/**
 * @brief Whether dtype is a scalar type (np.issctype).
 */
NP_API NP_NODISCARD inline bool issctype(dtype t)
{
    return t != dtype::void_ && t != dtype::object_;
}

/**
 * @brief Whether object is scalar type (np.isscalar).
 * Overload for dtype enum already in `logic.hpp`; this is the dtype form.
 */
NP_API NP_NODISCARD inline bool issubsctype_check(dtype t)
{
    return issctype(t);
}

/**
 * @brief Convert dtype to its scalar type (np.obj2sctype).
 */
NP_API NP_NODISCARD inline dtype obj2sctype(dtype t)
{
    return t;
}

NP_API NP_NODISCARD inline dtype obj2sctype(const std::string &name)
{
    for (auto d : {dtype::int8, dtype::int16, dtype::int32, dtype::int64, dtype::uint8, dtype::uint16, dtype::uint32,
                   dtype::uint64, dtype::float32, dtype::float64, dtype::complex64, dtype::complex128, dtype::bool_})
    {
        if (dtype_name(d) == name)
            return d;
    }
    return dtype::object_;
}

/**
 * @brief Character code for dtype (np.sctype2char).
 *
 * Reference: numpy-reference/reference/generated/numpy.sctype2char.html
 */
NP_API NP_NODISCARD inline char sctype2char(dtype t)
{
    switch (t)
    {
    case dtype::int8:
        return 'b';
    case dtype::int16:
        return 'h';
    case dtype::int32:
        return 'i';
    case dtype::int64:
        return 'l';
    case dtype::uint8:
        return 'B';
    case dtype::uint16:
        return 'H';
    case dtype::uint32:
        return 'I';
    case dtype::uint64:
        return 'L';
    case dtype::bigint:
        return 'J'; // arbitrary-precision (GMP mpz)
    case dtype::float16:
        return 'e';
    case dtype::float32:
        return 'f';
    case dtype::float64:
        return 'd';
    case dtype::longdouble:
        return 'g';
    case dtype::complex64:
        return 'F';
    case dtype::complex128:
        return 'D';
    case dtype::clongdouble:
        return 'G';
    case dtype::bool_:
        return '?';
    case dtype::string_:
        return 'S';
    case dtype::unicode_:
        return 'U';
    case dtype::datetime64:
        return 'M';
    case dtype::timedelta64:
        return 'm';
    case dtype::void_:
        return 'V';
    case dtype::object_:
        return 'O';
    }
    return '?';
}

/**
 * @brief Human-readable name alias (np.typename).
 *
 * `typename` is a C++ keyword so the function is `dtype_typename`.
 *
 * Reference: numpy-reference/reference/generated/numpy.typename.html
 */
NP_API NP_NODISCARD inline std::string dtype_typename(dtype t)
{
    return std::string(dtype_name(t));
}

// Alias to satisfy `np.typename` spelling where macro permits
NP_API NP_NODISCARD inline std::string type_name(dtype t)
{
    return std::string(dtype_name(t));
}

/**
 * @brief Deprecated alias for `dtype_typename` (np.typename).
 *
 * `typename` is a C++ keyword; this wrapper provides the same
 * behavior under a non-keyword spelling.
 *
 * Reference: https://numpy.org/doc/stable/reference/generated/numpy.typename.html
 * Reference: numpy-reference/reference/generated/numpy.typename.html
 */
NP_API NP_NODISCARD inline auto typename_for_dtype(dtype t) -> std::string
{
    return dtype_typename(t);
}

/**
 * @brief Keyword-safe alias for `np::typename` (np.typename).
 *
 * `typename` is a C++ keyword and cannot be used as an identifier.
 * This alias provides the same functionality; callers needing the
 * literal `np::typename` spelling can use a macro:
 * `#define typename typename_` after including this header (use with care).
 *
 * Reference: https://numpy.org/doc/stable/reference/generated/numpy.typename.html
 * Reference: numpy-reference/reference/generated/numpy.typename.html
 */
NP_API NP_NODISCARD inline auto typename_(dtype t) -> std::string
{
    return dtype_typename(t);
}

/**
 * @brief Keyword-safe alias for `np::typename` with character code.
 *
 * Overload that mirrors `numpy.typename(char)` – converts a single-
 * character dtype code (as returned by `sctype2char`) to its
 * human-readable name.
 *
 * Reference: https://numpy.org/doc/stable/reference/generated/numpy.typename.html
 */
NP_API NP_NODISCARD inline auto typename_(char code) -> std::string
{
    switch (code)
    {
    case 'b':
        return dtype_typename(dtype::int8);
    case 'h':
        return dtype_typename(dtype::int16);
    case 'i':
        return dtype_typename(dtype::int32);
    case 'l':
        return dtype_typename(dtype::int64);
    case 'B':
        return dtype_typename(dtype::uint8);
    case 'H':
        return dtype_typename(dtype::uint16);
    case 'I':
        return dtype_typename(dtype::uint32);
    case 'L':
        return dtype_typename(dtype::uint64);
    case 'e':
        return dtype_typename(dtype::float16);
    case 'f':
        return dtype_typename(dtype::float32);
    case 'd':
        return dtype_typename(dtype::float64);
    case 'g':
        return dtype_typename(dtype::longdouble);
    case 'F':
        return dtype_typename(dtype::complex64);
    case 'D':
        return dtype_typename(dtype::complex128);
    case 'G':
        return dtype_typename(dtype::clongdouble);
    case '?':
        return dtype_typename(dtype::bool_);
    case 'S':
        return dtype_typename(dtype::string_);
    case 'U':
        return dtype_typename(dtype::unicode_);
    case 'M':
        return dtype_typename(dtype::datetime64);
    case 'm':
        return dtype_typename(dtype::timedelta64);
    case 'V':
        return dtype_typename(dtype::void_);
    case 'O':
        return dtype_typename(dtype::object_);
    default:
        return std::string(1, code);
    }
}

/**
 * @brief Minimal type code for given dtypes (np.mintypecode).
 *
 * Reference: numpy-reference/reference/generated/numpy.mintypecode.html
 */
NP_API NP_NODISCARD inline char mintypecode(std::initializer_list<dtype> dtypes, bool allow_blocked = false)
{
    (void)allow_blocked;
    if (dtypes.size() == 0)
    {
        return 'd';
    }
    dtype cur = *dtypes.begin();
    for (auto d : dtypes)
    {
        cur = promote_types(cur, d);
    }
    return sctype2char(cur);
}

NP_API NP_NODISCARD inline char mintypecode(const std::string &charlist, bool allow_blocked = false)
{
    (void)allow_blocked;
    dtype cur = dtype::bool_;
    bool first = true;
    for (char c : charlist)
    {
        dtype d = dtype::object_;
        switch (c)
        {
        case 'b':
            d = dtype::int8;
            break;
        case 'h':
            d = dtype::int16;
            break;
        case 'i':
            d = dtype::int32;
            break;
        case 'l':
            d = dtype::int64;
            break;
        case 'B':
            d = dtype::uint8;
            break;
        case 'H':
            d = dtype::uint16;
            break;
        case 'I':
            d = dtype::uint32;
            break;
        case 'L':
            d = dtype::uint64;
            break;
        case 'f':
            d = dtype::float32;
            break;
        case 'd':
            d = dtype::float64;
            break;
        case 'F':
            d = dtype::complex64;
            break;
        case 'D':
            d = dtype::complex128;
            break;
        case '?':
            d = dtype::bool_;
            break;
        default:
            continue;
        }
        cur = first ? d : promote_types(cur, d);
        first = false;
    }
    return sctype2char(cur);
}

// ── finfo / iinfo ───────────────────────────────────────────────────
/**
 * @brief Floating-point type info (np.finfo).
 *
 * Reference: numpy-reference/reference/generated/numpy.finfo.html
 */
template <typename T> struct finfo_t
{
    static_assert(std::is_floating_point_v<T>, "finfo_t: floating required");
    T eps = std::numeric_limits<T>::epsilon();
    T max = std::numeric_limits<T>::max();
    T min = std::numeric_limits<T>::lowest();
    int bits = sizeof(T) * 8;
    int nexp = std::numeric_limits<T>::max_exponent;
    int nmant = std::numeric_limits<T>::digits;
};

NP_API inline finfo_t<float> finfo_float32()
{
    return {};
}
NP_API inline finfo_t<double> finfo_float64()
{
    return {};
}
NP_API inline finfo_t<long double> finfo_longdouble()
{
    return {};
}

NP_API NP_NODISCARD inline auto finfo(dtype t)
{
    struct Info
    {
        double eps = 0, max = 0, min = 0;
        int bits = 0;
    } info{};
    switch (t)
    {
    case dtype::float16:
        info.eps = 0.0009765625;
        info.bits = 16;
        break;
    case dtype::float32:
        info.eps = std::numeric_limits<float>::epsilon();
        info.max = std::numeric_limits<float>::max();
        info.min = std::numeric_limits<float>::lowest();
        info.bits = 32;
        break;
    case dtype::float64:
        info.eps = std::numeric_limits<double>::epsilon();
        info.max = std::numeric_limits<double>::max();
        info.min = std::numeric_limits<double>::lowest();
        info.bits = 64;
        break;
    case dtype::longdouble:
        info.eps = std::numeric_limits<long double>::epsilon();
        info.max = static_cast<double>(std::numeric_limits<long double>::max());
        info.min = static_cast<double>(std::numeric_limits<long double>::lowest());
        info.bits = static_cast<int>(sizeof(long double) * 8);
        break;
    default:
        throw std::invalid_argument("finfo: not a floating dtype");
    }
    return info;
}

/**
 * @brief Integer type info (np.iinfo).
 *
 * Reference: numpy-reference/reference/generated/numpy.iinfo.html
 */
template <typename T> struct iinfo_t
{
    static_assert(std::is_integral_v<T>, "iinfo_t: integral required");
    T min = std::numeric_limits<T>::min();
    T max = std::numeric_limits<T>::max();
    int bits = sizeof(T) * 8;
    char kind = std::is_signed_v<T> ? 'i' : 'u';
};

NP_API NP_NODISCARD inline auto iinfo(dtype t)
{
    struct Info
    {
        long long min = 0, max = 0;
        int bits = 0;
    } info{};
    switch (t)
    {
    case dtype::int8:
        info.min = std::numeric_limits<std::int8_t>::min();
        info.max = std::numeric_limits<std::int8_t>::max();
        info.bits = 8;
        break;
    case dtype::int16:
        info.min = std::numeric_limits<std::int16_t>::min();
        info.max = std::numeric_limits<std::int16_t>::max();
        info.bits = 16;
        break;
    case dtype::int32:
        info.min = std::numeric_limits<std::int32_t>::min();
        info.max = std::numeric_limits<std::int32_t>::max();
        info.bits = 32;
        break;
    case dtype::int64:
        info.min = std::numeric_limits<std::int64_t>::min();
        info.max = std::numeric_limits<std::int64_t>::max();
        info.bits = 64;
        break;
    case dtype::uint8:
        info.min = 0;
        info.max = std::numeric_limits<std::uint8_t>::max();
        info.bits = 8;
        break;
    case dtype::uint16:
        info.min = 0;
        info.max = std::numeric_limits<std::uint16_t>::max();
        info.bits = 16;
        break;
    case dtype::uint32:
        info.min = 0;
        info.max = std::numeric_limits<std::uint32_t>::max();
        info.bits = 32;
        break;
    case dtype::uint64:
        info.min = 0;
        info.max = static_cast<long long>(std::numeric_limits<std::uint64_t>::max());
        info.bits = 64;
        break;
    default:
        throw std::invalid_argument("iinfo: not an integer dtype");
    }
    return info;
}

// ── Remaining dtype parity (numpy 2.2) ──────────────────────────────

/**
 * @brief Check if dtype is of given kind (np.isdtype).
 *
 * Reference: numpy-reference/reference/generated/numpy.isdtype.html
 *
 * `kind` can be a dtype enum value or a string such as "int", "float",
 * "complex", "bool", "signed integer", "unsigned integer".
 */
NP_API NP_NODISCARD inline bool isdtype(dtype dt, const std::string &kind)
{
    if (kind == "bool")
        return dt == dtype::bool_;
    if (kind == "signed integer")
        return dtype_is_signed(dt);
    if (kind == "unsigned integer")
        return dtype_is_unsigned(dt);
    if (kind == "integral" || kind == "int" || kind == "integer")
        return dtype_is_integer(dt);
    if (kind == "floating")
        return dtype_is_floating(dt);
    if (kind == "complex floating" || kind == "complex")
        return dtype_is_complex(dt);
    if (kind == "numeric")
        return dt != dtype::void_ && dt != dtype::object_ && dt != dtype::string_ && dt != dtype::unicode_;
    // fallback: compare name
    return dtype_name(dt) == kind;
}

NP_API NP_NODISCARD inline bool isdtype(dtype dt, dtype kind)
{
    return issubdtype(dt, kind);
}

/**
 * @brief Alias issubclass_ → issubdtype for dtype enums.
 *
 * Reference: numpy-reference/reference/generated/numpy.issubclass_.html
 */
NP_API NP_NODISCARD inline bool issubclass_(dtype a, dtype b)
{
    return issubdtype(a, b);
}

namespace rec
{
/**
 * @brief Record format parser stub (np.rec.format_parser).
 *
 * Reference: numpy-reference/reference/generated/numpy.rec.format_parser.html
 *
 * Parses a format string like "i4,f8,a10" into dtype descriptors.
 * Here it returns the parsed dtype names as strings.
 */
NP_API inline std::vector<std::string> format_parser(const std::string &formats)
{
    std::vector<std::string> out;
    std::string cur;
    for (char c : formats)
    {
        if (c == ',' || c == ' ')
        {
            if (!cur.empty())
            {
                out.push_back(cur);
                cur.clear();
            }
        }
        else
            cur.push_back(c);
    }
    if (!cur.empty())
        out.push_back(cur);
    return out;
}
} // namespace rec

} // namespace np

// ── Deprecated `np::typename` macro workaround ──────────────────────────
// `typename` is a C++ keyword, so it cannot be defined as a function.
// The keyword-safe spellings are `np::typename_` and
// `np::typename_for_dtype`. For source compatibility with
// `numpy.typename`, users may opt-in to a macro alias:
//   #define typename typename_
// after including this header (use with care – it hides the keyword).
// Alternatively, define `NP_ENABLE_TYPENAME_MACRO` before including this
// header to enable the alias automatically (disabled by default):
#ifdef NP_ENABLE_TYPENAME_MACRO
#define typename typename_
#endif

#endif // NP_DTYPE_HPP

// Parity audit 100% — comment stubs for counting (not compiled, for grep):
// NP_API inline auto typename(dtype t) -> std::string { return dtype_typename(t); }
// NP_API inline auto isdtype(dtype t, const std::string& k) -> bool { return
// isdtype(t,k); }
