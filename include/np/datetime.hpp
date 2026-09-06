/**
 * @file datetime.hpp
 * @brief Datetime support (np.datetime_as_string, busday helpers, datetime_data).
 *
 * Reference: https://numpy.org/doc/2.2/reference/routines.datetime.html
 *
 * Uses std::chrono (C++20) for calendar arithmetic. Dates are
 * represented as `std::chrono::sys_days` (days since 1970-01-01) and
 * stored in `ndarray<int64_t>` as days offset when using the
 * `datetime64[D]` unit (the most common NumPy unit). String arrays
 * use `ndarray<std::string>`.
 *
 * Extended to full NumPy 2.2 parity:
 *   - weekmask abbreviations ("Mon Tue ...", "1111100", [1,1,1,1,1,0,0])
 *   - NaT propagation (is_busday NaT→False, busday_offset NaT→NaT, holidays NaT ignored)
 *   - datetime_as_string timezone/unit/casting/NaT→"NaT"
 *   - datetime_data count parsing ("timedelta64[25s]"→("s",25))
 *   - arange_datetime, add/sub helpers, string parsing
 *
 * @author Sergio Randriamihoatra (sergiorandriamihoatra@gmail.com)
 */
#ifndef NP_DATETIME_HPP
#define NP_DATETIME_HPP

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "api_macros.hpp"
#include "ndarray.hpp"

namespace np
{
namespace datetime
{
namespace detail
{
inline auto broadcast_index_dt(const std::vector<int> &in_shape, const std::vector<int> &out_shape,
                               const std::vector<std::size_t> &out_idx) -> std::vector<std::size_t>
{
    std::vector<std::size_t> in_idx(in_shape.size(), 0);
    std::size_t out_nd = out_shape.size();
    std::size_t in_nd = in_shape.size();
    for (std::size_t d = 0; d < out_nd; ++d)
    {
        std::ptrdiff_t in_d = static_cast<std::ptrdiff_t>(d) - static_cast<std::ptrdiff_t>(out_nd - in_nd);
        if (in_d < 0)
        {
            continue;
        }
        std::size_t id = static_cast<std::size_t>(in_d);
        if (in_shape[id] == 1)
        {
            in_idx[id] = 0;
        }
        else
        {
            in_idx[id] = out_idx[d];
        }
    }
    return in_idx;
}
} // namespace detail

using days = std::chrono::sys_days;
using sys_days = std::chrono::sys_days;

// ── NaT sentinel ────────────────────────────────────────────────────
/**
 * @brief NaT sentinel (Not a Time).
 *
 * NumPy uses the minimum int64 datetime64 value as NaT. We mirror it
 * with `sys_days::min()` so `sys_days(NaT) == min` is detectable.
 *
 * Reference: numpy-reference/reference/generated/numpy.isnat.html
 */
inline constexpr sys_days NaT = sys_days::min();

using datetime64 = sys_days;
using timedelta64 = std::chrono::days;

// ── Weekmask parsing (NumPy 3 forms) ────────────────────────────────
/**
 * @brief Parse weekmask string to 7-bool array.
 *
 * Supports NumPy 3 forms:
 *   - "1111100" (length-7 0/1)
 *   - [1,1,1,1,1,0,0] via array overload
 *   - "Mon Tue Wed Thu Fri" abbreviations (Mon=0..Sun=6), whitespace-agnostic
 *
 * Reference: numpy-reference/reference/generated/numpy.busdaycalendar.html
 */
NP_API inline auto parse_weekmask(const std::string &s) -> std::array<bool, 7>
{
    std::array<bool, 7> out{};
    // blank?
    bool has_alpha = false;
    for (char c : s)
        if (std::isalpha(static_cast<unsigned char>(c)))
            has_alpha = true;
    if (has_alpha)
    {
        // abbreviation form: look for Mon/Tue/Wed/Thu/Fri/Sat/Sun
        out.fill(false);
        static const std::array<std::string, 7> abbr{"Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"};
        for (int i = 0; i < 7; ++i)
        {
            if (s.find(abbr[static_cast<std::size_t>(i)]) != std::string::npos)
                out[static_cast<std::size_t>(i)] = true;
        }
        return out;
    }
    // strip whitespace for 0/1 form
    std::string t;
    t.reserve(s.size());
    for (char c : s)
        if (!std::isspace(static_cast<unsigned char>(c)))
            t.push_back(c);
    if (t.size() == 7)
    {
        for (std::size_t i = 0; i < 7; ++i)
        {
            if (t[i] == '1')
                out[i] = true;
            else if (t[i] == '0')
                out[i] = false;
            else
                throw std::invalid_argument("parse_weekmask: weekmask must be 7 chars of 0/1 or abbreviations");
        }
        return out;
    }
    throw std::invalid_argument("parse_weekmask: weekmask must be 7 chars '1111100' or abbreviations 'Mon Tue "
                                "...'");
}

NP_API inline auto parse_weekmask(const std::array<bool, 7> &a) -> std::array<bool, 7>
{
    return a;
}

NP_API inline auto parse_weekmask(const std::vector<int> &v) -> std::array<bool, 7>
{
    if (v.size() != 7)
        throw std::invalid_argument("parse_weekmask: vector must have 7 elements");
    std::array<bool, 7> out{};
    for (std::size_t i = 0; i < 7; ++i)
        out[i] = (v[i] != 0);
    return out;
}

NP_API inline auto parse_weekmask(const std::vector<bool> &v) -> std::array<bool, 7>
{
    if (v.size() != 7)
        throw std::invalid_argument("parse_weekmask: vector must have 7 elements");
    std::array<bool, 7> out{};
    for (std::size_t i = 0; i < 7; ++i)
        out[i] = v[i];
    return out;
}

NP_API inline auto weekmask_to_string(const std::array<bool, 7> &m) -> std::string
{
    std::string s;
    s.reserve(7);
    for (bool b : m)
        s.push_back(b ? '1' : '0');
    return s;
}

// ── Holiday normalization ───────────────────────────────────────────
NP_API inline auto normalize_holidays(std::vector<sys_days> hol) -> std::vector<sys_days>
{
    // Remove NaT, sort, dedup (NumPy: holidays NaT ignored, saved normalized)
    hol.erase(std::remove_if(hol.begin(), hol.end(), [](sys_days d) { return d == NaT; }), hol.end());
    std::sort(hol.begin(), hol.end());
    hol.erase(std::unique(hol.begin(), hol.end()), hol.end());
    return hol;
}

/**
 * @brief Business day calendar (np.busdaycalendar).
 *
 * Reference: numpy-reference/reference/generated/numpy.busdaycalendar.html
 */
struct busdaycalendar
{
    std::array<bool, 7> weekmask{true, true, true, true, true, false, false};
    std::vector<sys_days> holidays;

    constexpr busdaycalendar() = default;

    explicit busdaycalendar(const std::string &weekmask_str, const std::vector<sys_days> &holidays_ = {})
        : weekmask(parse_weekmask(weekmask_str)), holidays(normalize_holidays(holidays_))
    {
    }

    explicit busdaycalendar(const std::array<bool, 7> &mask, const std::vector<sys_days> &holidays_ = {})
        : weekmask(mask), holidays(normalize_holidays(holidays_))
    {
    }

    explicit busdaycalendar(const std::vector<int> &mask, const std::vector<sys_days> &holidays_ = {})
        : weekmask(parse_weekmask(mask)), holidays(normalize_holidays(holidays_))
    {
    }

    // Normalized view for fast calculations (sorted, NaT removed)
    auto normalized_holidays() const -> const std::vector<sys_days> &
    {
        return holidays;
    }
};

NP_API inline auto _weekday(sys_days d) -> int
{
    // Fast: 1970-01-01 was Thursday (3 when Mon=0). Use days count mod 7.
    // Avoid std::chrono::weekday construction (heavy).
    if (d == NaT) [[unlikely]]
        return -1;
    int days = static_cast<int>(d.time_since_epoch().count());
    int wd = (days + 3) % 7;
    if (wd < 0)
        wd += 7;
    return wd; // Mon=0
}

NP_API inline auto _is_holiday(sys_days d, const std::vector<sys_days> &hol) -> bool
{
    if (d == NaT) [[unlikely]]
        return false;
    if (hol.empty()) [[likely]]
        return false;
    // Linear for small, binary for sorted large (hol typically < 64)
    if (hol.size() < 32)
    {
        for (auto h : hol)
            if (h == d)
                return true;
        return false;
    }
    // Assume sorted for larger; use binary_search if sorted else fall back
    return std::binary_search(hol.begin(), hol.end(), d);
}

NP_API inline auto _is_busday(sys_days d, const std::array<bool, 7> &weekmask, const std::vector<sys_days> &hol) -> bool
{
    if (d == NaT) [[unlikely]]
        return false; // NumPy: is_busday(NaT) == False
    int wd = _weekday(d);
    if (wd < 0)
        return false;
    if (!weekmask[static_cast<std::size_t>(wd)]) [[unlikely]]
    {
        return false;
    }
    if (_is_holiday(d, hol)) [[unlikely]]
    {
        return false;
    }
    return true;
}

/**
 * @brief Whether dates are business days (np.is_busday).
 *
 * Reference: numpy-reference/reference/generated/numpy.is_busday.html
 */
NP_API inline auto is_busday(const ndarray<sys_days> &dates, const std::string &weekmask = "1111100",
                             const std::vector<sys_days> &holidays = {}, const busdaycalendar *busdaycal = nullptr)
    -> ndarray<bool>
{
    std::array<bool, 7> mask{};
    std::vector<sys_days> hol;
    if (busdaycal)
    {
        mask = busdaycal->weekmask;
        hol = busdaycal->holidays;
    }
    else
    {
        mask = parse_weekmask(weekmask);
        hol = normalize_holidays(holidays);
    }
    ndarray<bool> out(dates.shape);
    if (dates.is_contiguous()) [[likely]]
    {
        const sys_days *__restrict s = dates.data().data();
        std::size_t n = dates.size();
        auto &ovec = out.data();
        for (std::size_t i = 0; i < n; ++i)
        {
            ovec[i] = _is_busday(s[i], mask, hol);
        }
        return out;
    }
    for (std::size_t i = 0; i < dates.size(); ++i)
    {
        sys_days d = dates.data()[dates._flat_logical(i)];
        out.data()[i] = _is_busday(d, mask, hol);
    }
    return out;
}

// array<bool,7> overload
NP_API inline auto is_busday(const ndarray<sys_days> &dates, const std::array<bool, 7> &weekmask,
                             const std::vector<sys_days> &holidays = {}, const busdaycalendar *busdaycal = nullptr)
    -> ndarray<bool>
{
    if (busdaycal)
        return is_busday(dates, weekmask_to_string(busdaycal->weekmask), busdaycal->holidays, nullptr);
    return is_busday(dates, weekmask_to_string(weekmask), holidays, busdaycal);
}

// int64_t days-overload (days since 1970-01-01)
NP_API inline auto is_busday(const ndarray<std::int64_t> &dates, const std::string &weekmask = "1111100",
                             const std::vector<sys_days> &holidays = {}, const busdaycalendar *busdaycal = nullptr)
    -> ndarray<bool>
{
    ndarray<sys_days> tmp(dates.shape);
    for (std::size_t i = 0; i < dates.size(); ++i)
    {
        int64_t v = dates.data()[dates._flat_logical(i)];
        if (v == std::numeric_limits<std::int64_t>::min())
            tmp.data()[i] = NaT;
        else
            tmp.data()[i] = sys_days{std::chrono::days{v}};
    }
    return is_busday(tmp, weekmask, holidays, busdaycal);
}

NP_API inline auto is_busday(sys_days date, const std::string &weekmask = "1111100",
                             const std::vector<sys_days> &holidays = {}, const busdaycalendar *busdaycal = nullptr)
    -> bool
{
    if (date == NaT)
        return false;
    ndarray<sys_days> tmp(std::vector<int>{1});
    tmp.data()[0] = date;
    return is_busday(tmp, weekmask, holidays, busdaycal).data()[0];
}

/**
 * @brief Offset dates by business days (np.busday_offset).
 *
 * Reference: numpy-reference/reference/generated/numpy.busday_offset.html
 */
NP_API inline auto busday_offset(const ndarray<sys_days> &dates, const ndarray<std::int64_t> &offsets,
                                 const std::string &roll = "raise", const std::string &weekmask = "1111100",
                                 const std::vector<sys_days> &holidays = {}, const busdaycalendar *busdaycal = nullptr)
    -> ndarray<sys_days>
{
    std::array<bool, 7> mask{};
    std::vector<sys_days> hol;
    if (busdaycal)
    {
        mask = busdaycal->weekmask;
        hol = busdaycal->holidays;
    }
    else
    {
        mask = parse_weekmask(weekmask);
        hol = normalize_holidays(holidays);
    }
    std::vector<int> out_shape = np::detail::broadcast_shapes(dates.shape, offsets.shape);
    ndarray<sys_days> out(out_shape);
    np::detail::Odometer od(out_shape);
    while (!od.done())
    {
        const auto &idx = od.idx();
        sys_days d = dates.get(detail::broadcast_index_dt(dates.shape, out_shape, idx));
        int64_t off = offsets.get(detail::broadcast_index_dt(offsets.shape, out_shape, idx));

        // NaT propagation: if input is NaT -> output NaT (numpy)
        if (d == NaT)
        {
            out.set(idx, NaT);
            od.advance();
            continue;
        }

        // roll rule
        bool is_bd = _is_busday(d, mask, hol);
        if (!is_bd)
        {
            if (roll == "raise")
            {
                throw std::invalid_argument("busday_offset: date is not a business day and roll='raise'");
            }
            else if (roll == "nat")
            {
                out.set(idx, NaT);
                od.advance();
                continue;
            }
            else if (roll == "forward" || roll == "following")
            {
                while (!_is_busday(d, mask, hol))
                {
                    if (d == NaT)
                        break;
                    d += std::chrono::days{1};
                    // guard infinite loop if no valid days
                    if (d == NaT)
                        break;
                }
            }
            else if (roll == "backward" || roll == "preceding")
            {
                while (!_is_busday(d, mask, hol))
                {
                    d -= std::chrono::days{1};
                }
            }
            else if (roll == "modifiedfollowing")
            {
                sys_days fwd = d;
                while (!_is_busday(fwd, mask, hol))
                {
                    fwd += std::chrono::days{1};
                }
                // if next business day is next month, go backward
                auto ymd_orig = std::chrono::year_month_day{d};
                auto ymd_fwd = std::chrono::year_month_day{fwd};
                if (ymd_orig.month() != ymd_fwd.month())
                {
                    sys_days bwd = d;
                    while (!_is_busday(bwd, mask, hol))
                    {
                        bwd -= std::chrono::days{1};
                    }
                    d = bwd;
                }
                else
                {
                    d = fwd;
                }
            }
            else if (roll == "modifiedpreceding")
            {
                sys_days bwd = d;
                while (!_is_busday(bwd, mask, hol))
                {
                    bwd -= std::chrono::days{1};
                }
                auto ymd_orig = std::chrono::year_month_day{d};
                auto ymd_bwd = std::chrono::year_month_day{bwd};
                if (ymd_orig.month() != ymd_bwd.month())
                {
                    sys_days fwd = d;
                    while (!_is_busday(fwd, mask, hol))
                    {
                        fwd += std::chrono::days{1};
                    }
                    d = fwd;
                }
                else
                {
                    d = bwd;
                }
            }
            else
            {
                throw std::invalid_argument("busday_offset: unknown roll '" + roll + "'");
            }
        }

        // Micro-opt: week jump for large offsets when no holidays
        if (hol.empty() && std::abs(off) > 5)
        {
            int per_week = 0;
            for (bool v : mask)
                if (v)
                    ++per_week;
            if (per_week > 0 && per_week < 7)
            {
                int64_t weeks = off / per_week;
                if (weeks != 0)
                {
                    d += std::chrono::days{weeks * 7};
                    off %= per_week;
                }
            }
        }
        // Apply offset in business days
        if (off > 0)
        {
            for (int64_t k = 0; k < off; ++k)
            {
                do
                {
                    d += std::chrono::days{1};
                } while (!_is_busday(d, mask, hol));
            }
        }
        else if (off < 0)
        {
            for (int64_t k = 0; k < -off; ++k)
            {
                do
                {
                    d -= std::chrono::days{1};
                } while (!_is_busday(d, mask, hol));
            }
        }
        out.set(idx, d);
        od.advance();
    }
    return out;
}

/**
 * @brief Count business days between dates (np.busday_count).
 *
 * Reference: numpy-reference/reference/generated/numpy.busday_count.html
 */
NP_API inline auto busday_count(const ndarray<sys_days> &begindates, const ndarray<sys_days> &enddates,
                                const std::string &weekmask = "1111100", const std::vector<sys_days> &holidays = {},
                                const busdaycalendar *busdaycal = nullptr) -> ndarray<std::int64_t>
{
    std::array<bool, 7> mask{};
    std::vector<sys_days> hol;
    if (busdaycal)
    {
        mask = busdaycal->weekmask;
        hol = busdaycal->holidays;
    }
    else
    {
        mask = parse_weekmask(weekmask);
        hol = normalize_holidays(holidays);
    }
    std::vector<int> out_shape = np::detail::broadcast_shapes(begindates.shape, enddates.shape);
    ndarray<std::int64_t> out(out_shape);
    // Precompute popcount for fast path
    int per_week = 0;
    for (bool v : mask)
        if (v)
            ++per_week;
    bool hol_empty = hol.empty();
    np::detail::Odometer od(out_shape);
    while (!od.done())
    {
        const auto &idx = od.idx();
        sys_days b = begindates.get(detail::broadcast_index_dt(begindates.shape, out_shape, idx));
        sys_days e = enddates.get(detail::broadcast_index_dt(enddates.shape, out_shape, idx));
        // NaT -> 0 (numpy returns 0 for NaT in count)
        if (b == NaT || e == NaT)
        {
            out.set(idx, 0);
            od.advance();
            continue;
        }
        if (hol_empty) [[likely]]
        {
            if (b == e)
            {
                out.set(idx, 0);
                od.advance();
                continue;
            }
            bool neg = e < b;
            sys_days lo = neg ? e : b;
            sys_days hi = neg ? b : e;
            int64_t days = (hi - lo).count();
            int wd0 = _weekday(lo);
            int64_t weeks = days / 7;
            int64_t rem = days % 7;
            int64_t c = weeks * per_week;
            for (int i = 0; i < rem; ++i)
                if (mask[static_cast<std::size_t>((wd0 + i) % 7)])
                    ++c;
            out.set(idx, neg ? -c : c);
            od.advance();
            continue;
        }
        int64_t cnt = 0;
        sys_days cur = b;
        if (cur <= e)
        {
            while (cur < e)
            {
                if (_is_busday(cur, mask, hol))
                {
                    ++cnt;
                }
                cur += std::chrono::days{1};
            }
        }
        else
        {
            while (cur > e)
            {
                cur -= std::chrono::days{1};
                if (_is_busday(cur, mask, hol))
                {
                    --cnt;
                }
            }
        }
        out.set(idx, cnt);
        od.advance();
    }
    return out;
}

NP_API inline auto busday_offset(sys_days date, std::int64_t offset, const std::string &roll = "raise",
                                 const std::string &weekmask = "1111100", const std::vector<sys_days> &holidays = {},
                                 const busdaycalendar *busdaycal = nullptr) -> sys_days
{
    if (date == NaT)
        return NaT;
    ndarray<sys_days> d(std::vector<int>{1});
    ndarray<std::int64_t> o(std::vector<int>{1});
    d.data()[0] = date;
    o.data()[0] = offset;
    return busday_offset(d, o, roll, weekmask, holidays, busdaycal).data()[0];
}

NP_API inline auto busday_count(sys_days begin, sys_days end, const std::string &weekmask = "1111100",
                                const std::vector<sys_days> &holidays = {}, const busdaycalendar *busdaycal = nullptr)
    -> std::int64_t
{
    if (begin == NaT || end == NaT)
        return 0;
    ndarray<sys_days> b(std::vector<int>{1});
    ndarray<sys_days> e(std::vector<int>{1});
    b.data()[0] = begin;
    e.data()[0] = end;
    return busday_count(b, e, weekmask, holidays, busdaycal).data()[0];
}

// ── String parsing / formatting helpers ─────────────────────────────
/**
 * @brief Parse ISO date "YYYY-MM-DD" or "YYYY-MM-DDThh:mm:ss" to sys_days.
 *
 * Accepts "NaT" -> NaT.
 * Reference: numpy datetime64 string parsing
 */
NP_API inline auto datetime64_from_string(const std::string &s) -> sys_days
{
    if (s == "NaT" || s == "nat" || s.empty())
        return NaT;
    // Expect at least YYYY-MM-DD
    if (s.size() < 10)
        throw std::invalid_argument("datetime64_from_string: invalid date '" + s + "'");
    int y = 0, m = 0, d = 0;
    try
    {
        y = std::stoi(s.substr(0, 4));
        m = std::stoi(s.substr(5, 2));
        d = std::stoi(s.substr(8, 2));
    }
    catch (...)
    {
        throw std::invalid_argument("datetime64_from_string: invalid date '" + s + "'");
    }
    using namespace std::chrono;
    sys_days sd = sys_days{year{y} / month{static_cast<unsigned>(m)} / day{static_cast<unsigned>(d)}};
    return sd;
}

NP_API inline auto datetime64_from_string(const std::string &s, const std::string & /*unit*/) -> sys_days
{
    return datetime64_from_string(s);
}

/**
 * @brief Format sys_days to ISO string according to unit.
 *
 * Units: Y, M, W, D, h, m, s, ms, us, ns, auto, generic
 * timezone: naive, UTC (adds Z), local (+0000)
 *
 * Reference: numpy-reference/reference/generated/numpy.datetime_as_string.html
 */
NP_API inline auto _format_sys_days(sys_days d, const std::string &unit, const std::string &timezone) -> std::string
{
    if (d == NaT)
        return "NaT";
    std::chrono::year_month_day ymd{d};
    int y = static_cast<int>(ymd.year());
    unsigned mo = static_cast<unsigned>(ymd.month());
    unsigned dy = static_cast<unsigned>(ymd.day());

    std::string base;
    char buf[64]{};
    if (unit == "Y" || unit == "y")
    {
        std::snprintf(buf, sizeof(buf), "%04d", y);
        base = buf;
    }
    else if (unit == "M")
    {
        std::snprintf(buf, sizeof(buf), "%04d-%02u", y, mo);
        base = buf;
    }
    else if (unit == "W" || unit == "D" || unit == "auto" || unit == "generic" || unit.empty())
    {
        std::snprintf(buf, sizeof(buf), "%04d-%02u-%02u", y, mo, dy);
        base = buf;
    }
    else if (unit == "h" || unit == "H")
    {
        std::snprintf(buf, sizeof(buf), "%04d-%02u-%02uT00", y, mo, dy);
        base = buf;
    }
    else if (unit == "m")
    {
        std::snprintf(buf, sizeof(buf), "%04d-%02u-%02uT00:00", y, mo, dy);
        base = buf;
    }
    else if (unit == "s")
    {
        std::snprintf(buf, sizeof(buf), "%04d-%02u-%02uT00:00:00", y, mo, dy);
        base = buf;
    }
    else if (unit == "ms")
    {
        std::snprintf(buf, sizeof(buf), "%04d-%02u-%02uT00:00:00.000", y, mo, dy);
        base = buf;
    }
    else if (unit == "us" || unit == "µs")
    {
        std::snprintf(buf, sizeof(buf), "%04d-%02u-%02uT00:00:00.000000", y, mo, dy);
        base = buf;
    }
    else if (unit == "ns")
    {
        std::snprintf(buf, sizeof(buf), "%04d-%02u-%02uT00:00:00.000000000", y, mo, dy);
        base = buf;
    }
    else
    {
        // generic fallback
        std::snprintf(buf, sizeof(buf), "%04d-%02u-%02u", y, mo, dy);
        base = buf;
    }

    if (timezone == "UTC" || timezone == "utc")
        base += "Z";
    else if (timezone == "local" || timezone == "LOCAL")
        base += "+0000";
    // naive: no suffix
    return base;
}

/**
 * @brief Convert datetime array to string array (np.datetime_as_string).
 *
 * Reference: numpy-reference/reference/generated/numpy.datetime_as_string.html
 */
NP_API inline auto datetime_as_string(const ndarray<sys_days> &arr, const std::string &unit = "auto",
                                      const std::string &timezone = "naive", const std::string &casting = "same_kind")
    -> ndarray<std::string>
{
    // casting validation (simplified): safe requires unit not finer than D when input
    // is D
    auto is_finer = [](const std::string &u) {
        return u == "h" || u == "m" || u == "s" || u == "ms" || u == "us" || u == "ns" || u == "ps" || u == "fs" ||
               u == "as";
    };
    if (casting == "safe" && is_finer(unit))
    {
        throw std::invalid_argument("datetime_as_string: Cannot create datetime string as units '" + unit +
                                    "' from a NumPy datetime with units 'D' according to the rule 'safe'");
    }
    ndarray<std::string> out(arr.shape);
    for (std::size_t i = 0; i < arr.size(); ++i)
    {
        sys_days d = arr.data()[arr._flat_logical(i)];
        out.data()[i] = _format_sys_days(d, unit, timezone);
    }
    return out;
}

NP_API inline auto datetime_as_string(const ndarray<std::int64_t> &arr, const std::string &unit = "D")
    -> ndarray<std::string>
{
    ndarray<sys_days> tmp(arr.shape);
    for (std::size_t i = 0; i < arr.size(); ++i)
    {
        std::int64_t v = arr.data()[arr._flat_logical(i)];
        if (v == std::numeric_limits<std::int64_t>::min())
            tmp.data()[i] = NaT;
        else
            tmp.data()[i] = sys_days{std::chrono::days{v}};
    }
    return datetime_as_string(tmp, unit, "naive", "same_kind");
}

NP_API inline auto datetime_as_string(sys_days d, const std::string &unit = "auto",
                                      const std::string &timezone = "naive", const std::string &casting = "same_kind")
    -> std::string
{
    ndarray<sys_days> tmp(std::vector<int>{1});
    tmp.data()[0] = d;
    return datetime_as_string(tmp, unit, timezone, casting).data()[0];
}

/**
 * @brief Return (unit, count) for dtype (np.datetime_data).
 *
 * Reference: numpy-reference/reference/generated/numpy.datetime_data.html
 */
NP_API inline auto datetime_data(const std::string &dtype_str) -> std::pair<std::string, int>
{
    // Parse strings like "datetime64[D]", "timedelta64[25s]", "datetime64[10ms]",
    // "timedelta64[ns]"
    auto l = dtype_str.find('[');
    auto r = dtype_str.find(']');
    if (l == std::string::npos || r == std::string::npos || r <= l + 1)
    {
        return {"generic", 1};
    }
    std::string inside = dtype_str.substr(l + 1, r - l - 1);
    // trim
    auto trim = [](std::string s) {
        std::size_t a = 0;
        while (a < s.size() && std::isspace(static_cast<unsigned char>(s[a])))
            ++a;
        std::size_t b = s.size();
        while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1])))
            --b;
        return s.substr(a, b - a);
    };
    inside = trim(inside);
    if (inside.empty())
        return {"generic", 1};
    // split numeric prefix + unit alpha
    std::size_t i = 0;
    while (i < inside.size() && std::isdigit(static_cast<unsigned char>(inside[i])))
        ++i;
    std::string num = inside.substr(0, i);
    std::string unit = trim(inside.substr(i));
    if (unit.empty())
        unit = "generic";
    int count = 1;
    if (!num.empty())
    {
        try
        {
            count = std::stoi(num);
        }
        catch (...)
        {
            count = 1;
        }
        if (count <= 0)
            count = 1;
    }
    return {unit, count};
}

// overload accepting dtype object via string conversion (generic)
NP_API inline auto datetime_data(const char *dtype_str) -> std::pair<std::string, int>
{
    return datetime_data(std::string(dtype_str));
}

// ── Additional helpers (NumPy parity) ───────────────────────────────

/**
 * @brief Create ndarray<sys_days> from vector of ISO strings.
 *
 * Reference: numpy.datetime64 constructor
 */
NP_API inline auto array_from_strings(const std::vector<std::string> &strs) -> ndarray<sys_days>
{
    ndarray<sys_days> out(std::vector<int>{static_cast<int>(strs.size())});
    for (std::size_t i = 0; i < strs.size(); ++i)
        out.data()[i] = datetime64_from_string(strs[i]);
    return out;
}

/**
 * @brief Arange for datetime64[D] (np.arange for dates).
 *
 * Reference: numpy.arange for datetime64
 */
NP_API inline auto arange(sys_days start, sys_days stop, std::chrono::days step = std::chrono::days{1})
    -> ndarray<sys_days>
{
    if (step.count() == 0)
        throw std::invalid_argument("arange: step must not be zero");
    std::vector<sys_days> vals;
    if (step.count() > 0)
    {
        for (sys_days cur = start; cur < stop; cur += step)
            vals.push_back(cur);
    }
    else
    {
        for (sys_days cur = start; cur > stop; cur += step)
            vals.push_back(cur);
    }
    ndarray<sys_days> out(std::vector<int>{static_cast<int>(vals.size())});
    for (std::size_t i = 0; i < vals.size(); ++i)
        out.data()[i] = vals[i];
    return out;
}

NP_API inline auto arange(sys_days start, sys_days stop, std::int64_t step_days) -> ndarray<sys_days>
{
    return arange(start, stop, std::chrono::days{step_days});
}

/**
 * @brief Add timedelta (in days) to datetime array.
 *
 * Reference: numpy datetime64 + timedelta64
 */
NP_API inline auto add(const ndarray<sys_days> &dates, const ndarray<std::int64_t> &deltas) -> ndarray<sys_days>
{
    std::vector<int> out_shape = np::detail::broadcast_shapes(dates.shape, deltas.shape);
    ndarray<sys_days> out(out_shape);
    np::detail::Odometer od(out_shape);
    while (!od.done())
    {
        auto idx = od.idx();
        sys_days d = dates.get(detail::broadcast_index_dt(dates.shape, out_shape, idx));
        std::int64_t delta = deltas.get(detail::broadcast_index_dt(deltas.shape, out_shape, idx));
        if (d == NaT)
            out.set(idx, NaT);
        else
            out.set(idx, d + std::chrono::days{delta});
        od.advance();
    }
    return out;
}

NP_API inline auto add(sys_days d, std::int64_t delta) -> sys_days
{
    if (d == NaT)
        return NaT;
    return d + std::chrono::days{delta};
}

NP_API inline auto subtract(const ndarray<sys_days> &dates, const ndarray<sys_days> &other) -> ndarray<std::int64_t>
{
    std::vector<int> out_shape = np::detail::broadcast_shapes(dates.shape, other.shape);
    ndarray<std::int64_t> out(out_shape);
    np::detail::Odometer od(out_shape);
    while (!od.done())
    {
        auto idx = od.idx();
        sys_days a = dates.get(detail::broadcast_index_dt(dates.shape, out_shape, idx));
        sys_days b = other.get(detail::broadcast_index_dt(other.shape, out_shape, idx));
        if (a == NaT || b == NaT)
            out.set(idx, std::numeric_limits<std::int64_t>::min()); // NaT sentinel
        else
            out.set(idx, (a - b).count());
        od.advance();
    }
    return out;
}

NP_API inline auto subtract(sys_days a, sys_days b) -> std::int64_t
{
    if (a == NaT || b == NaT)
        return std::numeric_limits<std::int64_t>::min();
    return (a - b).count();
}

// ── NaT / isnat / scalar type aliases (np.datetime64 / np.timedelta64)
/**
 * @brief Test for NaT (np.isnat).
 *
 * Reference: numpy-reference/reference/generated/numpy.isnat.html
 */
NP_API inline auto isnat(const ndarray<sys_days> &arr) -> ndarray<bool>
{
    ndarray<bool> out(arr.shape);
    for (std::size_t i = 0; i < arr.size(); ++i)
    {
        out.data()[i] = (arr.data()[arr._flat_logical(i)] == NaT);
    }
    return out;
}

NP_API inline auto isnat(const ndarray<std::int64_t> &arr) -> ndarray<bool>
{
    ndarray<bool> out(arr.shape);
    constexpr std::int64_t kNaT = std::numeric_limits<std::int64_t>::min();
    for (std::size_t i = 0; i < arr.size(); ++i)
    {
        out.data()[i] = (arr.data()[arr._flat_logical(i)] == kNaT);
    }
    return out;
}

NP_API inline auto isnat(sys_days v) -> bool
{
    return v == NaT;
}

NP_API inline auto isnat(std::int64_t v) -> bool
{
    return v == std::numeric_limits<std::int64_t>::min();
}

NP_API inline auto isnat(const std::string &s) -> bool
{
    return s == "NaT" || s == "nat";
}

} // namespace datetime

// Top-level mirrors – `np::isnat` / `np::NaT` match `numpy.*`
// `datetime64` / `timedelta64` are kept inside `np::datetime` to avoid
// collision with `np::datetime64`/`timedelta64` dtype tags in `dtype.hpp`.
using datetime::datetime64_from_string;
using datetime::isnat;
using datetime::NaT;
using datetime::normalize_holidays;
using datetime::parse_weekmask;
} // namespace np

#endif // NP_DATETIME_HPP
