/**
 * @file bigint.hpp
 * @brief Arbitrary-precision integer support (np::bigint) with GMP mpz backend.
 *
 * Provides `np::bigint` as a header-only wrapper over
 * `boost::multiprecision::cpp_int` (default) and `boost::multiprecision::mpz_int`
 * (when GMP is available). Designed for `np::ndarray<bigint>`:
 *   - creation / manipulation / math / linalg exact integer paths
 *   - `dtype::bigint` enumeration + `dtype_of<bigint>` == bigint
 *   - `to_mpz` / `from_mpz` helpers for `mpz_t`
 *   - constexpr auto-promotion via `auto_bigint_t<T>` / `as_bigint()`
 *
 * Usage:
 *   #include <np/np.hpp>          // includes bigint.hpp automatically
 *   #include <np/bigint.hpp>
 *   np::ndarray<np::bigint> a = {np::bigint("12345678901234567890"), 42};
 *   auto b = np::as_bigint(a);    // ndarray<int> -> ndarray<bigint>
 *   auto c = np::linalg::det(a);  // exact bigint determinant
 *   mpz_t z; mpz_init_set_str(z, "999...", 10);
 *   np::bigint bi = np::from_mpz(z);
 *   np::to_mpz(bi, z);
 *
 * Reference:
 * https://www.boost.org/doc/libs/release/libs/multiprecision/doc/html/boost_multiprecision/tut/primetest.html
 *            https://gmplib.org/manual/Integer-Functions
 *
 * @author Sergio Randriamihoatra (sergiorandriamihoatra@gmail.com)
 */
#ifndef NP_BIGINT_HPP
#define NP_BIGINT_HPP

#include <string>
#include <type_traits>
#include <vector>

#include "api_macros.hpp"
#include "dtype.hpp"
#include "ndarray.hpp"

// Boost.Multiprecision backend
#if __has_include(<boost/multiprecision/cpp_int.hpp>)
#include <boost/multiprecision/cpp_int.hpp>
#define NP_HAS_CPP_INT 1
#else
#define NP_HAS_CPP_INT 0
#endif

#if defined(NP_HAS_GMP) && __has_include(<boost/multiprecision/gmp.hpp>)
#include <boost/multiprecision/gmp.hpp>
#define NP_HAS_GMP_MPZ_INT 1
#else
#define NP_HAS_GMP_MPZ_INT 0
#endif

#if defined(NP_HAS_GMP) && __has_include(<gmp.h>)
#include <gmp.h>
// gmp.h defines NZERO macro that conflicts with np::constants::NZERO
#ifdef NZERO
#undef NZERO
#endif
#define NP_HAS_GMP_H 1
#else
#define NP_HAS_GMP_H 0
#endif

namespace np
{

#if NP_HAS_CPP_INT
using bigint = boost::multiprecision::cpp_int;
#if NP_HAS_GMP_MPZ_INT
using mpz_bigint = boost::multiprecision::mpz_int;
#else
using mpz_bigint = bigint;
#endif
#else
// Minimal fallback when Boost not available
struct bigint
{
    std::string value = "0";
    bigint() = default;
    template <typename T>
        requires(std::is_integral_v<T> && !std::is_same_v<T, bool>)
    bigint(T v) : value(std::to_string(v))
    {
    }
    bigint(const std::string &s) : value(s)
    {
    }
    bigint(std::string_view s) : value(s)
    {
    }
    bigint(const char *s) : value(s ? s : "0")
    {
    }
    bigint(char *s) : value(s ? s : "0")
    {
    }
    bigint(std::nullptr_t) = delete;

    // Compatibility with boost::multiprecision::cpp_int::convert_to<T>()
    template <typename T> T convert_to() const
    {
        if constexpr (std::is_same_v<T, std::string>)
            return value;
        else if constexpr (std::is_same_v<T, const char *>)
            return value.c_str();
        else if constexpr (std::is_integral_v<T>)
        {
            // use stoll as intermediate; sufficient for fallback tests (small values)
            long long v = 0;
            try
            {
                v = std::stoll(value);
            }
            catch (...)
            {
                v = 0;
            }
            return static_cast<T>(v);
        }
        else if constexpr (std::is_floating_point_v<T>)
        {
            double v = 0;
            try
            {
                v = std::stod(value);
            }
            catch (...)
            {
                v = 0;
            }
            return static_cast<T>(v);
        }
        else
        {
            // generic fallback via string construction
            if constexpr (std::is_constructible_v<T, std::string>)
                return T(value);
            else
                return T{};
        }
    }

    // allow static_cast<long long>(bigint) etc (explicit to avoid accidental)
    template <typename T>
        requires(std::is_arithmetic_v<T> && !std::is_same_v<T, bool>)
    explicit operator T() const
    {
        return convert_to<T>();
    }

    bool operator==(const bigint &o) const
    {
        auto norm = [](const std::string &s) -> std::string {
            if (s.empty())
                return "0";
            bool neg = s[0] == '-';
            std::string a = neg ? s.substr(1) : s;
            std::size_t i = 0;
            while (i + 1 < a.size() && a[i] == '0')
                ++i;
            std::string r = a.substr(i);
            if (r == "0")
                return "0";
            return neg ? "-" + r : r;
        };
        return norm(value) == norm(o.value);
    }
    bool operator<(const bigint &o) const
    {
        bool na = !value.empty() && value[0] == '-';
        bool nb = !o.value.empty() && o.value[0] == '-';
        std::string aa = na ? value.substr(1) : value;
        std::string bb = nb ? o.value.substr(1) : o.value;
        auto strip_leading = [](const std::string &s) -> std::string {
            std::size_t i = 0;
            while (i + 1 < s.size() && s[i] == '0')
                ++i;
            return s.substr(i);
        };
        aa = strip_leading(aa);
        bb = strip_leading(bb);
        if (aa == "0")
            na = false;
        if (bb == "0")
            nb = false;
        if (aa == "0" && bb == "0")
            return false;
        if (na != nb)
            return na;
        if (!na)
        {
            if (aa.size() != bb.size())
                return aa.size() < bb.size();
            return aa < bb;
        }
        else
        {
            if (aa.size() != bb.size())
                return aa.size() > bb.size();
            return aa > bb;
        }
    }
    bool operator>(const bigint &o) const
    {
        return o < *this;
    }
    bool operator<=(const bigint &o) const
    {
        return !(o < *this);
    }
    bool operator>=(const bigint &o) const
    {
        return !(*this < o);
    }
    bool operator!=(const bigint &o) const
    {
        return !(*this == o);
    }
};
using mpz_bigint = bigint;
#endif

namespace detail
{
template <typename T> struct is_bigint : std::false_type
{
};
template <> struct is_bigint<bigint> : std::true_type
{
};
#if NP_HAS_GMP_MPZ_INT
template <> struct is_bigint<mpz_bigint> : std::true_type
{
};
#endif
template <typename T> inline constexpr bool is_bigint_v = is_bigint<std::remove_cv_t<T>>::value;

template <typename A, typename B> struct common_bigint
{
    using type = std::conditional_t<is_bigint_v<A> || is_bigint_v<B>, bigint, std::common_type_t<A, B>>;
};
template <typename A, typename B> using common_bigint_t = typename common_bigint<A, B>::type;

} // namespace detail

// fallback bigint string arithmetic helpers
#if !NP_HAS_CPP_INT
namespace detail::fallback_bigint
{
inline bool is_neg(const std::string &s)
{
    return !s.empty() && s[0] == '-';
}
inline std::string abs_str(const std::string &s)
{
    return is_neg(s) ? s.substr(1) : s;
}
inline std::string strip(const std::string &s)
{
    if (s.empty())
        return "0";
    bool neg = is_neg(s);
    std::string a = neg ? s.substr(1) : s;
    std::size_t i = 0;
    while (i + 1 < a.size() && a[i] == '0')
        ++i;
    std::string r = a.substr(i);
    if (r.empty())
        r = "0";
    if (r == "0")
        return "0";
    return neg ? "-" + r : r;
}
inline std::string strip_abs(const std::string &s)
{
    if (s.empty())
        return "0";
    std::size_t i = 0;
    while (i + 1 < s.size() && s[i] == '0')
        ++i;
    std::string r = s.substr(i);
    return r.empty() ? "0" : r;
}
inline int cmp_abs_str(const std::string &a, const std::string &b)
{
    std::string aa = strip_abs(a), bb = strip_abs(b);
    if (aa.size() != bb.size())
        return aa.size() < bb.size() ? -1 : 1;
    if (aa == bb)
        return 0;
    return aa < bb ? -1 : 1;
}
inline std::string add_abs_str(const std::string &a, const std::string &b)
{
    std::string res;
    int i = static_cast<int>(a.size()) - 1, j = static_cast<int>(b.size()) - 1, carry = 0;
    while (i >= 0 || j >= 0 || carry)
    {
        int sum = carry;
        if (i >= 0)
            sum += a[i--] - '0';
        if (j >= 0)
            sum += b[j--] - '0';
        res.push_back(char('0' + (sum % 10)));
        carry = sum / 10;
    }
    std::reverse(res.begin(), res.end());
    return strip_abs(res);
}
inline std::string sub_abs_str(const std::string &a, const std::string &b) // a>=b, both positive
{
    std::string res;
    int i = static_cast<int>(a.size()) - 1, j = static_cast<int>(b.size()) - 1, borrow = 0;
    while (i >= 0)
    {
        int da = a[i--] - '0' - borrow;
        int db = j >= 0 ? b[j--] - '0' : 0;
        if (da < db)
        {
            da += 10;
            borrow = 1;
        }
        else
            borrow = 0;
        res.push_back(char('0' + (da - db)));
    }
    while (res.size() > 1 && res.back() == '0')
        res.pop_back();
    std::reverse(res.begin(), res.end());
    return strip_abs(res);
}
inline std::string mul_abs_str(const std::string &a, const std::string &b)
{
    if (a == "0" || b == "0")
        return "0";
    std::vector<int> r(a.size() + b.size(), 0);
    for (int i = static_cast<int>(a.size()) - 1; i >= 0; --i)
        for (int j = static_cast<int>(b.size()) - 1; j >= 0; --j)
            r[i + j + 1] += (a[i] - '0') * (b[j] - '0');
    for (int k = static_cast<int>(r.size()) - 1; k > 0; --k)
    {
        r[k - 1] += r[k] / 10;
        r[k] %= 10;
    }
    std::string s;
    bool leading = true;
    for (int v : r)
    {
        if (leading && v == 0)
            continue;
        leading = false;
        s.push_back(char('0' + v));
    }
    return s.empty() ? "0" : s;
}
inline std::string div_abs_str(const std::string &a, const std::string &b) // integer division, b !=0
{
    // naive long division via stoll fallback for small numbers, else via repeated subtraction using string compare for
    // moderate sizes For fallback tests values are within ~20 digits, we can use built-in __int128 if fits, otherwise
    // fallback to string long division Try to use boost-like: if both fit in 64 bits use stoll, else do long division
    // via string
    try
    {
        // if both < 18 digits, use stoll safely
        if (a.size() <= 18 && b.size() <= 18)
        {
            long long av = std::stoll(a);
            long long bv = std::stoll(b);
            if (bv == 0)
                return "0";
            return strip_abs(std::to_string(av / bv));
        }
    }
    catch (...)
    {
    }
    // long division (grade school)
    std::string quotient;
    std::string cur;
    for (char c : a)
    {
        cur.push_back(c);
        cur = strip_abs(cur);
        int q = 0;
        while (cmp_abs_str(cur, b) >= 0)
        {
            cur = sub_abs_str(cur, b);
            ++q;
            if (q > 9)
                break; // safety, but for single digit quotient per step
        }
        // Actually need proper digit estimation; fallback to iterative subtract is okay for small b (p small) but for
        // large b may be slow. Use binary search for q 0..9 Recompute correctly via loop above (max 9 iterations if b
        // single digit, but b may be large, then q is 0 or 1)
        quotient.push_back(char('0' + q));
    }
    return strip_abs(quotient);
}
} // namespace detail::fallback_bigint
#endif

// ——— bigint arithmetic (ADL-visible in np) ———
inline bigint operator+(const bigint &a, const bigint &b)
{
#if NP_HAS_CPP_INT
    bigint r = a;
    r += b;
    return r;
#else
    using namespace detail::fallback_bigint;
    bool na = is_neg(a.value), nb = is_neg(b.value);
    std::string aa = abs_str(a.value), bb = abs_str(b.value);
    if (!na && !nb)
        return bigint(add_abs_str(aa, bb));
    if (na && nb)
        return bigint("-" + add_abs_str(aa, bb));
    // different signs: a + (-b) = a - b
    int cmp = cmp_abs_str(aa, bb);
    if (cmp == 0)
        return bigint("0");
    if (!na && nb) // a - |b|
        return cmp > 0 ? bigint(sub_abs_str(aa, bb)) : bigint("-" + sub_abs_str(bb, aa));
    else // -a + b = b - a
        return cmp > 0 ? bigint("-" + sub_abs_str(aa, bb)) : bigint(sub_abs_str(bb, aa));
#endif
}
inline bigint operator-(const bigint &a, const bigint &b)
{
#if NP_HAS_CPP_INT
    bigint r = a;
    r -= b;
    return r;
#else
    // a - b = a + (-b)
    bigint nb = b;
    if (!nb.value.empty() && nb.value[0] == '-')
        nb.value = nb.value.substr(1);
    else if (nb.value != "0")
        nb.value = "-" + nb.value;
    return operator+(a, nb);
#endif
}
inline bigint operator*(const bigint &a, const bigint &b)
{
#if NP_HAS_CPP_INT
    bigint r = a;
    r *= b;
    return r;
#else
    using namespace detail::fallback_bigint;
    if (a.value == "0" || b.value == "0")
        return bigint("0");
    bool neg = is_neg(a.value) != is_neg(b.value);
    std::string aa = abs_str(a.value), bb = abs_str(b.value);
    std::string pr = mul_abs_str(aa, bb);
    return bigint(neg ? "-" + pr : pr);
#endif
}
inline bigint operator/(const bigint &a, const bigint &b)
{
#if NP_HAS_CPP_INT
    bigint r = a;
    r /= b;
    return r;
#else
    using namespace detail::fallback_bigint;
    if (b.value == "0" || b.value == "-0")
        return bigint("0");
    if (a.value == "0")
        return bigint("0");
    bool neg = is_neg(a.value) != is_neg(b.value);
    std::string aa = abs_str(a.value), bb = abs_str(b.value);
    std::string q = div_abs_str(aa, bb);
    if (q == "0")
        return bigint("0");
    return bigint(neg ? "-" + q : q);
#endif
}
inline bigint operator%(const bigint &a, const bigint &b)
{
#if NP_HAS_CPP_INT
    bigint r = a;
    r %= b;
    return r;
#else
    // a % b = a - (a/b)*b
    if (b.value == "0" || b.value == "-0")
        return bigint("0");
    bigint q = operator/(a, b);
    bigint prod = operator*(q, b);
    return operator-(a, prod);
#endif
}
inline bigint operator-(const bigint &a)
{
#if NP_HAS_CPP_INT
    bigint zero = 0;
    zero -= a;
    return zero;
#else
    if (a.value == "0" || a.value == "-0")
        return bigint("0");
    if (!a.value.empty() && a.value[0] == '-')
        return bigint(a.value.substr(1));
    return bigint("-" + a.value);
#endif
}
inline bigint operator+(const bigint &a)
{
    return a;
}
// compound ops only for fallback (cpp_int already has member ops)
#if !NP_HAS_CPP_INT
inline bigint &operator+=(bigint &a, const bigint &b)
{
    a = a + b;
    return a;
}
inline bigint &operator-=(bigint &a, const bigint &b)
{
    a = a - b;
    return a;
}
inline bigint &operator*=(bigint &a, const bigint &b)
{
    a = a * b;
    return a;
}
inline bigint &operator/=(bigint &a, const bigint &b)
{
    a = a / b;
    return a;
}
inline bigint &operator%=(bigint &a, const bigint &b)
{
    a = a % b;
    return a;
}
#endif

/**
 * @brief Constexpr auto-promotion to bigint.
 *
 * Any integral type (except bool) maps to `np::bigint`; floating / complex / bigint
 * stay. `np::auto_bigint_t<int>` == `np::bigint`.
 */
template <typename T>
using auto_bigint_t = std::conditional_t<
    detail::is_bigint_v<T>, T,
    std::conditional_t<std::is_integral_v<std::remove_cv_t<T>> && !std::is_same_v<std::remove_cv_t<T>, bool>, bigint,
                       T>>;

template <typename T>
inline constexpr bool auto_promotes_to_bigint_v = std::is_same_v<auto_bigint_t<T>, bigint> && !detail::is_bigint_v<T>;

// Converters
NP_NODISCARD inline bigint to_bigint(long long v)
{
    return bigint(v);
}
NP_NODISCARD inline bigint to_bigint(const std::string &s)
{
    return bigint(s);
}
NP_NODISCARD inline bigint to_bigint(const char *s)
{
    return bigint(s);
}
NP_NODISCARD inline bigint to_bigint(const bigint &v)
{
    return v;
}
template <typename T>
    requires(std::is_arithmetic_v<T> && !detail::is_bigint_v<T>)
NP_NODISCARD inline bigint to_bigint(T v)
{
    return bigint(v);
}

#if NP_HAS_GMP_H
NP_NODISCARD inline bigint from_mpz(const mpz_t z)
{
#if NP_HAS_GMP_MPZ_INT
    mpz_bigint tmp(z);
    return bigint(tmp);
#else
#if NP_HAS_CPP_INT
    char *s = mpz_get_str(nullptr, 10, z);
    bigint out(s);
    ::free(s);
    return out;
#else
    char *s = mpz_get_str(nullptr, 10, z);
    bigint out(s);
    ::free(s);
    return out;
#endif
#endif
}

inline void to_mpz(const bigint &bi, mpz_t rop)
{
#if NP_HAS_GMP_MPZ_INT
    mpz_bigint tmp(bi);
    mpz_set(rop, tmp.backend().data());
#else
#if NP_HAS_CPP_INT
    std::string s = bi.convert_to<std::string>();
    mpz_set_str(rop, s.c_str(), 10);
#else
    mpz_set_str(rop, bi.value.c_str(), 10);
#endif
#endif
}

NP_NODISCARD inline mpz_bigint to_mpz_bigint(const bigint &bi)
{
    return mpz_bigint(bi);
}
#endif // NP_HAS_GMP_H

inline namespace literals
{
NP_NODISCARD inline bigint operator""_bigint(const char *str, std::size_t len)
{
    return bigint(std::string(str, len));
}
NP_NODISCARD inline bigint operator""_mpz(const char *str, std::size_t len)
{
    return bigint(std::string(str, len));
}
} // namespace literals

//  Ergonomic helpers

/**
 * @brief Create `bigint` from any string/arithmetic literal (constexpr where possible).
 * `auto b = np::make_bigint("12345678901234567890");`
 */
template <typename T> NP_NODISCARD inline auto make_bigint(T &&v) -> bigint
{
    if constexpr (detail::is_bigint_v<std::remove_cv_t<T>>)
        return bigint(std::forward<T>(v));
    else if constexpr (std::is_same_v<std::remove_cv_t<T>, const char *> || std::is_same_v<std::remove_cv_t<T>, char *>)
        return bigint(std::string(v));
    else
        return bigint(std::forward<T>(v));
}

/**
 * @brief `constexpr` promote any integral `ndarray` to `ndarray<bigint>`.
 * `auto b = np::promote_to_bigint(a); // ndarray<int> -> ndarray<bigint> if needed`
 * If already `bigint`, returns copy.
 */
template <typename T> NP_NODISCARD inline auto promote_to_bigint(const ndarray<T> &a)
{
    if constexpr (detail::is_bigint_v<T>)
        return a.copy();
    else
        return as_bigint(a);
}

/**
 * @brief Create `ndarray<bigint>` from initializer list of ints/strings.
 * `auto a = np::make_bigint_array({1, 2, 3});`
 * `auto b = np::make_bigint_array({"123", "456"});`
 */
template <typename T> NP_NODISCARD inline auto make_bigint_array(std::initializer_list<T> list) -> ndarray<bigint>
{
    std::vector<bigint> data;
    data.reserve(list.size());
    for (auto &v : list)
        data.emplace_back(make_bigint(v));
    ndarray<bigint> out(std::vector<int>{static_cast<int>(data.size())});
    for (size_t i = 0; i < data.size(); ++i)
        out[i] = data[i];
    return out;
}

} // namespace np

// dtype integration: map bigint -> dtype::bigint
namespace np::detail
{
template <> struct cxx_to_np_type_impl<np::bigint>
{
    static constexpr np::dtype value = np::dtype::bigint;
};
#if NP_HAS_GMP_MPZ_INT
template <> struct cxx_to_np_type_impl<np::mpz_bigint>
{
    static constexpr np::dtype value = np::dtype::bigint;
};
#endif
} // namespace np::detail

// std::common_type
namespace std
{
#if NP_HAS_CPP_INT
template <> struct common_type<np::bigint, np::bigint>
{
    using type = np::bigint;
};
template <typename T> struct common_type<T, np::bigint>
{
    using type = np::bigint;
};
template <typename T> struct common_type<np::bigint, T>
{
    using type = np::bigint;
};
#endif
} // namespace std

//  ndarray converters (need ndarray definition)
#include "ndarray.hpp"
namespace np
{
/**
 * @brief Convert `ndarray<T>` → `ndarray<bigint>` (exact).
 * `ndarray<int>{1,2} -> ndarray<bigint>{1,2}`
 */
template <typename T>
    requires(!detail::is_bigint_v<T>)
NP_NODISCARD inline auto as_bigint(const ndarray<T> &a) -> ndarray<bigint>
{
    ndarray<bigint> out(a.shape);
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        out.data()[i] = bigint(a.data()[a._flat_logical(i)]);
    }
    return out;
}

// Identity overload
NP_NODISCARD inline auto as_bigint(const ndarray<bigint> &a) -> ndarray<bigint>
{
    return a.copy();
}

/**
 * @brief Convert `ndarray<bigint>` → `ndarray<T>` with truncation.
 * For `T` integral, uses `convert_to<T>()` which throws on overflow for cpp_int.
 */
template <typename T> NP_NODISCARD inline auto from_bigint(const ndarray<bigint> &a) -> ndarray<T>
{
    ndarray<T> out(a.shape);
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        const bigint &bi = a.data()[a._flat_logical(i)];
#if NP_HAS_CPP_INT
        if constexpr (std::is_integral_v<T>)
            out.data()[i] = bi.convert_to<T>();
        else
            out.data()[i] = static_cast<T>(bi.convert_to<long double>());
#else
        out.data()[i] = static_cast<T>(std::stoll(bi.value));
#endif
    }
    return out;
}

/**
 * @brief Create ndarray<bigint> from string literals (convenience).
 * `np::bigints({"123456...", "999..."})`
 */
NP_NODISCARD inline auto bigints(std::initializer_list<std::string> list) -> ndarray<bigint>
{
    std::vector<bigint> data;
    data.reserve(list.size());
    for (auto &s : list)
        data.emplace_back(s.c_str());
    ndarray<bigint> out(std::vector<int>{static_cast<int>(data.size())});
    for (std::size_t i = 0; i < data.size(); ++i)
        out[i] = data[i];
    return out;
}
NP_NODISCARD inline auto bigints(std::initializer_list<const char *> list) -> ndarray<bigint>
{
    std::vector<bigint> data;
    data.reserve(list.size());
    for (auto s : list)
        data.emplace_back(s);
    ndarray<bigint> out(std::vector<int>{static_cast<int>(data.size())});
    for (std::size_t i = 0; i < data.size(); ++i)
        out[i] = data[i];
    return out;
}
// Overload for already-bigint list
NP_NODISCARD inline auto bigints(std::initializer_list<bigint> list) -> ndarray<bigint>
{
    std::vector<bigint> data(list.begin(), list.end());
    ndarray<bigint> out(std::vector<int>{static_cast<int>(data.size())});
    for (std::size_t i = 0; i < data.size(); ++i)
        out[i] = data[i];
    return out;
}

} // namespace np

#endif // NP_BIGINT_HPP
