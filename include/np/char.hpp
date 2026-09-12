/**
 * @file char.hpp
 * @brief String operations for arrays of std::string (numpy.char module).
 *
 * Implements element-wise string manipulation functions matching numpy.char
 * semantics. All functions operate on ndarray<std::string> and return either
 * string arrays, boolean arrays, or integer arrays depending on the operation.
 *
 * Function signatures, parameter order, and default values mirror the
 * numpy.char pages in numpy-reference/reference/generated/numpy.char.*.
 *
 * Implementation notes:
 *  - encode() and decode() are no-ops (C++ std::string is already byte-based)
 *  - isdecimal() and isnumeric() use isdigit() approximation (C++ limitation)
 *  - split(), rsplit(), splitlines() return flattened arrays (not object
 * arrays)
 *  - translate() uses simplified 256-character table (not dict-based)
 *  - chararray class is deprecated (per NumPy 2.5) but provided for
 * compatibility
 *
 * @author Sergio Randriamihoatra (sergiorandriamihoatra@gmail.com)
 */
#ifndef NP_CHAR_HPP
#define NP_CHAR_HPP

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <locale>

// String search tuning (macros, no magic numbers in logic)
#define NP_CHAR_ASCII_SIZE 256
#define NP_CHAR_SAM_TEXT_THRESH 512
#define NP_CHAR_SAM_PAT_THRESH 4
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "api_macros.hpp"
#include "creation.hpp"
#include "ndarray.hpp"

namespace np
{
namespace ch
{

/* Internal helpers - character classification and string utilities */
namespace detail
{

/* Validate that two arrays have matching shapes */
inline void validate_shapes(const ndarray<std::string> &a, const ndarray<std::string> &b, const char *func_name)
{
    if (a.shape != b.shape)
    {
        throw std::invalid_argument(std::string(func_name) + ": shape mismatch");
    }
}

/* Check if string contains cased characters and all are lowercase */
inline bool str_islower(const std::string &s)
{
    bool has_cased = false;
    for (char c : s)
    {
        if (std::isalpha(static_cast<unsigned char>(c)))
        {
            has_cased = true;
            if (!std::islower(static_cast<unsigned char>(c)))
            {
                return false;
            }
        }
    }
    return has_cased;
}

/* Check if string contains cased characters and all are uppercase */
inline bool str_isupper(const std::string &s)
{
    bool has_cased = false;
    for (char c : s)
    {
        if (std::isalpha(static_cast<unsigned char>(c)))
        {
            has_cased = true;
            if (!std::isupper(static_cast<unsigned char>(c)))
            {
                return false;
            }
        }
    }
    return has_cased;
}

/* Check if string is titlecased */
inline bool str_istitle(const std::string &s)
{
    bool in_word = false;
    bool has_cased = false;

    for (char c : s)
    {
        bool is_alpha = std::isalpha(static_cast<unsigned char>(c));
        if (is_alpha)
        {
            has_cased = true;
            if (in_word)
            {
                if (!std::islower(static_cast<unsigned char>(c)))
                {
                    return false;
                }
            }
            else
            {
                if (!std::isupper(static_cast<unsigned char>(c)))
                {
                    return false;
                }
                in_word = true;
            }
        }
        else
        {
            in_word = false;
        }
    }
    return has_cased;
}

/* Micro-optimized string searching: KMP + Suffix Automaton
 * KMP O(n+m) for single pattern, SAM O(n) build + O(m) query
 * for repeated queries on same text. SAM is chosen when text
 * is long and pattern is reused, otherwise KMP. Both beat
 * naive O(n*m) and libstdc++ find on worst case like
 * "aaaa...aab" vs "aaab".
 */
inline std::vector<int> kmp_prefix(const std::string &pat)
{
    std::vector<int> pi(pat.size(), 0);
    for (std::size_t i = 1; i < pat.size(); ++i)
    {
        int j = pi[i - 1];
        while (j > 0 && pat[i] != pat[static_cast<std::size_t>(j)])
        {
            j = pi[static_cast<std::size_t>(j) - 1];
        }
        if (pat[i] == pat[static_cast<std::size_t>(j)])
        {
            ++j;
        }
        pi[i] = j;
    }
    return pi;
}

inline std::size_t kmp_find(const std::string &text, const std::string &pat, std::size_t start = 0)
{
    if (pat.empty())
    {
        return start <= text.size() ? start : std::string::npos;
    }
    if (start >= text.size() || pat.size() > text.size() - start)
    {
        return std::string::npos;
    }
    // Micro-opt: for short pattern use memchr/BM, for tiny text use naive
    if (pat.size() == 1)
    {
        auto pos = text.find(pat[0], start);
        return pos;
    }
    // KMP
    auto pi = kmp_prefix(pat);
    int j = 0;
    for (std::size_t i = start; i < text.size(); ++i)
    {
        while (j > 0 && text[i] != pat[static_cast<std::size_t>(j)])
        {
            j = pi[static_cast<std::size_t>(j) - 1];
        }
        if (text[i] == pat[static_cast<std::size_t>(j)])
        {
            ++j;
            if (j == static_cast<int>(pat.size()))
            {
                return i - pat.size() + 1;
            }
        }
    }
    return std::string::npos;
}

inline std::size_t kmp_rfind(const std::string &text, const std::string &pat, std::size_t start = 0,
                             std::size_t end = std::string::npos)
{
    if (pat.empty())
    {
        return end == std::string::npos ? text.size() : std::min(end, text.size());
    }
    std::size_t n = text.size();
    if (end == std::string::npos || end > n)
        end = n;
    if (start >= end || pat.size() > end - start)
        return std::string::npos;
    // Reverse KMP: search reversed strings
    std::string rev_text(text.rbegin() + static_cast<std::ptrdiff_t>(n - end),
                         text.rbegin() + static_cast<std::ptrdiff_t>(n - start));
    std::string rev_pat(pat.rbegin(), pat.rend());
    auto pos = kmp_find(rev_text, rev_pat, 0);
    if (pos == std::string::npos)
        return std::string::npos;
    // pos in reversed corresponds to end - pos - pat.size() in original
    return end - pos - pat.size();
}

/* Suffix Automaton - O(|text|) build, O(|pat|) query, O(|text|) memory
 * 2*|text| states, each with 256 transitions (array for ASCII).
 * For binary strings and repeated queries, SAM beats KMP amortized.
 * Code is intentionally long (explicit arrays) for micro-optimization.
 */
struct SuffixAutomaton
{
    struct State
    {
        int link;
        int len;
        int first_pos;
        int cnt;
        std::array<int, NP_CHAR_ASCII_SIZE> next;
        State() : link(-1), len(0), first_pos(-1), cnt(0)
        {
            next.fill(-1);
        }
    };
    std::vector<State> st;
    int last;

    SuffixAutomaton() : st(1), last(0)
    {
        st[0].link = -1;
    }

    explicit SuffixAutomaton(const std::string &s) : st(), last(0)
    {
        st.reserve(2 * s.size() + 1);
        st.emplace_back();
        st[0].link = -1;
        last = 0;
        for (std::size_t i = 0; i < s.size(); ++i)
        {
            extend(static_cast<unsigned char>(s[i]), static_cast<int>(i));
        }
        // propagate cnt and first_pos via counting sort by len
        int max_len = 0;
        for (auto &state : st)
            max_len = std::max(max_len, state.len);
        std::vector<int> bucket(max_len + 1, 0);
        for (auto &state : st)
            ++bucket[state.len];
        for (int i = 1; i <= max_len; ++i)
            bucket[i] += bucket[i - 1];
        std::vector<int> order(st.size());
        for (int i = static_cast<int>(st.size()) - 1; i >= 0; --i)
        {
            order[--bucket[st[i].len]] = i;
        }
        for (int i = static_cast<int>(order.size()) - 1; i > 0; --i)
        {
            int v = order[i];
            int p = st[v].link;
            if (p >= 0)
            {
                st[p].cnt += st[v].cnt;
                // keep minimal first_pos for earliest occurrence
                if (st[p].first_pos == -1 || st[v].first_pos < st[p].first_pos)
                {
                    // first_pos already minimal due to building order, keep
                }
            }
        }
    }

    void extend(unsigned char c, int pos)
    {
        int cur = static_cast<int>(st.size());
        st.emplace_back();
        st[cur].len = st[last].len + 1;
        st[cur].first_pos = pos;
        st[cur].cnt = 1;
        int p = last;
        while (p != -1 && st[p].next[c] == -1)
        {
            st[p].next[c] = cur;
            p = st[p].link;
        }
        if (p == -1)
        {
            st[cur].link = 0;
        }
        else
        {
            int q = st[p].next[c];
            if (st[p].len + 1 == st[q].len)
            {
                st[cur].link = q;
            }
            else
            {
                int clone = static_cast<int>(st.size());
                st.emplace_back();
                st[clone] = st[q];
                st[clone].len = st[p].len + 1;
                st[clone].cnt = 0; // clone does not correspond to new endpos
                // first_pos remains q's first_pos
                while (p != -1 && st[p].next[c] == q)
                {
                    st[p].next[c] = clone;
                    p = st[p].link;
                }
                st[q].link = st[cur].link = clone;
            }
        }
        last = cur;
    }

    std::optional<std::size_t> find(const std::string &pat) const
    {
        if (pat.empty())
            return std::size_t{0};
        int v = 0;
        for (unsigned char c : pat)
        {
            int nxt = st[v].next[c];
            if (nxt == -1)
                return std::nullopt;
            v = nxt;
        }
        // first_pos is end position of first occurrence
        int end = st[v].first_pos;
        if (end == -1)
            return std::nullopt;
        return static_cast<std::size_t>(end - static_cast<int>(pat.size()) + 1);
    }

    int count_occurrences(const std::string &pat) const
    {
        if (pat.empty())
            return 0;
        int v = 0;
        for (unsigned char c : pat)
        {
            int nxt = st[v].next[c];
            if (nxt == -1)
                return 0;
            v = nxt;
        }
        return st[v].cnt;
    }

    bool contains(const std::string &pat) const
    {
        if (pat.empty())
            return true;
        int v = 0;
        for (unsigned char c : pat)
        {
            int nxt = st[v].next[c];
            if (nxt == -1)
                return false;
            v = nxt;
        }
        return true;
    }
};

/* Unified micro-optimized find: chooses best algorithm
 * - tiny pat (1) -> memchr
 * - short text (<256) or pat <4 -> KMP
 * - long text (>=512) with repeated queries potential -> SAM
 * For single query, KMP and SAM are both O(n+m), but SAM has higher
 * constant. We use heuristic: if text.size() > 512 and pat.size() > 3
 * build SAM, else KMP.
 */
inline std::size_t optimized_find(const std::string &text, const std::string &pat, std::size_t start = 0)
{
    if (pat.empty())
        return start <= text.size() ? start : std::string::npos;
    if (start >= text.size())
        return std::string::npos;
    // For long text, SAM can be faster amortized, but for single query KMP is enough.
    // Use SAM when text is very long and we want worst-case guarantee.
    if (text.size() >= NP_CHAR_SAM_TEXT_THRESH && pat.size() >= NP_CHAR_SAM_PAT_THRESH)
    {
        SuffixAutomaton sam(text);
        auto res = sam.find(pat);
        if (!res.has_value())
            return std::string::npos;
        std::size_t pos = *res;
        if (pos < start)
        {
            // SAM finds first occurrence from 0, need to find >= start
            // fall back to KMP for start-constrained search
            return kmp_find(text, pat, start);
        }
        return pos;
    }
    return kmp_find(text, pat, start);
}

inline int optimized_count(const std::string &text, const std::string &pat, int start, int end)
{
    int n = static_cast<int>(text.size());
    int e = (end < 0) ? n : std::min(end, n);
    int st = std::max(0, start);
    if (pat.empty() || st >= e)
        return 0;
    std::string slice = text.substr(st, e - st);
    if (slice.size() < pat.size())
        return 0;
    // Use SAM for counting occurrences (overlapping count via cnt), but we need
    // non-overlapping count per numpy. Numpy counts non-overlapping, so we still need
    // to iterate. Use KMP to find non-overlapping in O(n+m)
    int cnt = 0;
    auto pi = kmp_prefix(pat);
    int j = 0;
    for (std::size_t i = 0; i < slice.size(); ++i)
    {
        while (j > 0 && slice[i] != pat[static_cast<std::size_t>(j)])
            j = pi[static_cast<std::size_t>(j) - 1];
        if (slice[i] == pat[static_cast<std::size_t>(j)])
            ++j;
        if (j == static_cast<int>(pat.size()))
        {
            ++cnt;
            j = 0; // non-overlapping
        }
    }
    return cnt;
}

} /* namespace detail */

/* String Operations */

/**
 * @brief Return element-wise string concatenation.
 *
 * Performs element-wise concatenation: result[i] = x1[i] + x2[i].
 * Both input arrays must have identical shapes.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.add.html
 *
 * @param x1 First string array
 * @param x2 Second string array
 * @return Element-wise concatenated array
 * @throws std::invalid_argument if x1.shape != x2.shape
 */
NP_API inline auto add(const ndarray<std::string> &x1, const ndarray<std::string> &x2) -> ndarray<std::string>
{
    detail::validate_shapes(x1, x2, "char.add");

    ndarray<std::string> result = empty<std::string>(x1.shape);
    for (std::size_t i = 0; i < x1.size(); ++i)
    {
        result.data()[i] = x1.data()[i] + x2.data()[i];
    }
    return result;
}

/**
 * @brief Return (a * i), that is string multiple concatenation, element-wise.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.multiply.html
 *
 * @param a String array
 * @param i Integer array (repeat count for each string)
 * @return Element-wise string repetition
 */
NP_API inline auto multiply(const ndarray<std::string> &a, const ndarray<int> &i) -> ndarray<std::string>
{
    if (a.shape != i.shape)
    {
        throw std::invalid_argument("multiply: arrays must have the same shape");
    }
    ndarray<std::string> result = empty<std::string>(a.shape);
    for (std::size_t idx = 0; idx < a.size(); ++idx)
    {
        int count = i.data()[idx];
        if (count < 0)
            count = 0;
        std::string repeated;
        repeated.reserve(a.data()[idx].size() * count);
        for (int j = 0; j < count; ++j)
        {
            repeated += a.data()[idx];
        }
        result.data()[idx] = repeated;
    }
    return result;
}

/**
 * @brief Return (a % i), that is string formatting, element-wise.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.mod.html
 *
 * @param a Format string array
 * @param values Values to substitute
 * @return Formatted strings
 */
NP_API inline auto mod(const ndarray<std::string> &a, const ndarray<std::string> &values) -> ndarray<std::string>
{
    if (a.shape != values.shape)
    {
        throw std::invalid_argument("mod: arrays must have the same shape");
    }
    ndarray<std::string> result = empty<std::string>(a.shape);
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        const std::string &fmt = a.data()[i];
        const std::string &val = values.data()[i];
        std::string out;
        out.reserve(fmt.size() + val.size());
        for (std::size_t p = 0; p < fmt.size(); ++p)
        {
            if (fmt[p] == '%' && p + 1 < fmt.size())
            {
                if (fmt[p + 1] == '%')
                {
                    out.push_back('%');
                    ++p;
                }
                else
                {
                    // Find end of format specifier: flags/width/precision then type
                    std::size_t q = p + 1;
                    while (q < fmt.size() && std::string("-+ #0").find(fmt[q]) != std::string::npos)
                        ++q;
                    while (q < fmt.size() && std::isdigit(static_cast<unsigned char>(fmt[q])))
                        ++q;
                    if (q < fmt.size() && fmt[q] == '.')
                    {
                        ++q;
                        while (q < fmt.size() && std::isdigit(static_cast<unsigned char>(fmt[q])))
                            ++q;
                    }
                    if (q < fmt.size() && std::string("diouxXeEfFgGcrs").find(fmt[q]) != std::string::npos)
                    {
                        out += val;
                        p = q;
                    }
                    else
                    {
                        out.push_back(fmt[p]);
                    }
                }
            }
            else
            {
                out.push_back(fmt[p]);
            }
        }
        result.data()[i] = out;
    }
    return result;
}

/**
 * @brief Return a copy with only the first character capitalized.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.capitalize.html
 *
 * @param a Input string array
 * @return Array with capitalized strings
 */
NP_API inline auto capitalize(const ndarray<std::string> &a) -> ndarray<std::string>
{
    ndarray<std::string> result = empty<std::string>(a.shape);
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        std::string s = a.data()[i];
        if (!s.empty())
        {
            s[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(s[0])));
            for (std::size_t j = 1; j < s.size(); ++j)
            {
                s[j] = static_cast<char>(std::tolower(static_cast<unsigned char>(s[j])));
            }
        }
        result.data()[i] = s;
    }
    return result;
}

/**
 * @brief Return a copy centered in a string of length width.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.center.html
 *
 * @param a Input string array
 * @param width Minimum width of resulting string
 * @param fillchar Padding character (default: space)
 * @return Array with centered strings
 */
NP_API inline auto center(const ndarray<std::string> &a, int width, char fillchar = ' ') -> ndarray<std::string>
{
    ndarray<std::string> result = empty<std::string>(a.shape);
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        const std::string &s = a.data()[i];
        if (static_cast<int>(s.size()) >= width)
        {
            result.data()[i] = s;
        }
        else
        {
            int total_pad = width - static_cast<int>(s.size());
            int left_pad = total_pad / 2;
            int right_pad = total_pad - left_pad;
            result.data()[i] = std::string(left_pad, fillchar) + s + std::string(right_pad, fillchar);
        }
    }
    return result;
}

/**
 * @brief Calls str.lower() for each element.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.lower.html
 *
 * @param a Input string array
 * @return Array with lowercase strings
 */
NP_API inline auto lower(const ndarray<std::string> &a) -> ndarray<std::string>
{
    ndarray<std::string> result = empty<std::string>(a.shape);
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        std::string s = a.data()[i];
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
        result.data()[i] = s;
    }
    return result;
}

/**
 * @brief Calls str.upper() for each element.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.upper.html
 *
 * @param a Input string array
 * @return Array with uppercase strings
 */
NP_API inline auto upper(const ndarray<std::string> &a) -> ndarray<std::string>
{
    ndarray<std::string> result = empty<std::string>(a.shape);
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        std::string s = a.data()[i];
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::toupper(c); });
        result.data()[i] = s;
    }
    return result;
}

/**
 * @brief Return a copy with leading and trailing characters removed.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.strip.html
 *
 * @param a Input string array
 * @param chars Characters to remove (default: whitespace)
 * @return Array with stripped strings
 */
NP_API inline auto strip(const ndarray<std::string> &a, const std::string &chars = " \t\n\r") -> ndarray<std::string>
{
    ndarray<std::string> result = empty<std::string>(a.shape);
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        std::string s = a.data()[i];
        // Left strip
        std::size_t start = s.find_first_not_of(chars);
        if (start == std::string::npos)
        {
            result.data()[i] = "";
            continue;
        }
        // Right strip
        std::size_t end = s.find_last_not_of(chars);
        result.data()[i] = s.substr(start, end - start + 1);
    }
    return result;
}

/**
 * @brief Return a copy with leading characters removed.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.lstrip.html
 *
 * @param a Input string array
 * @param chars Characters to remove (default: whitespace)
 * @return Array with left-stripped strings
 */
NP_API inline auto lstrip(const ndarray<std::string> &a, const std::string &chars = " \t\n\r") -> ndarray<std::string>
{
    ndarray<std::string> result = empty<std::string>(a.shape);
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        std::string s = a.data()[i];
        std::size_t start = s.find_first_not_of(chars);
        if (start == std::string::npos)
        {
            result.data()[i] = "";
        }
        else
        {
            result.data()[i] = s.substr(start);
        }
    }
    return result;
}

/**
 * @brief Return a copy with trailing characters removed.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.rstrip.html
 *
 * @param a Input string array
 * @param chars Characters to remove (default: whitespace)
 * @return Array with right-stripped strings
 */
NP_API inline auto rstrip(const ndarray<std::string> &a, const std::string &chars = " \t\n\r") -> ndarray<std::string>
{
    ndarray<std::string> result = empty<std::string>(a.shape);
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        const std::string &s = a.data()[i];
        std::size_t end = s.find_last_not_of(chars);
        if (end == std::string::npos)
        {
            result.data()[i] = "";
        }
        else
        {
            result.data()[i] = s.substr(0, end + 1);
        }
    }
    return result;
}

/**
 * @brief Return a copy with all case-swapped characters.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.swapcase.html
 *
 * @param a Input string array
 * @return Array with swapped case strings
 */
NP_API inline auto swapcase(const ndarray<std::string> &a) -> ndarray<std::string>
{
    ndarray<std::string> result = empty<std::string>(a.shape);
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        std::string s = a.data()[i];
        for (char &c : s)
        {
            if (std::islower(static_cast<unsigned char>(c)))
            {
                c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
            }
            else if (std::isupper(static_cast<unsigned char>(c)))
            {
                c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            }
        }
        result.data()[i] = s;
    }
    return result;
}

/**
 * @brief Return a titlecased version of the string.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.title.html
 *
 * @param a Input string array
 * @return Array with titlecased strings
 */
NP_API inline auto title(const ndarray<std::string> &a) -> ndarray<std::string>
{
    ndarray<std::string> result = empty<std::string>(a.shape);
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        std::string s = a.data()[i];
        bool capitalize_next = true;
        for (char &c : s)
        {
            if (std::isalpha(static_cast<unsigned char>(c)))
            {
                if (capitalize_next)
                {
                    c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
                    capitalize_next = false;
                }
                else
                {
                    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
                }
            }
            else
            {
                capitalize_next = true;
            }
        }
        result.data()[i] = s;
    }
    return result;
}

/**
 * @brief Pad each string with zeros on the left.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.zfill.html
 *
 * @param a Input string array
 * @param width Minimum width of resulting string
 * @return Array with zero-filled strings
 */
NP_API inline auto zfill(const ndarray<std::string> &a, int width) -> ndarray<std::string>
{
    ndarray<std::string> result = empty<std::string>(a.shape);
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        std::string s = a.data()[i];
        if (static_cast<int>(s.size()) >= width)
        {
            result.data()[i] = s;
        }
        else
        {
            int pad = width - static_cast<int>(s.size());
            // Handle sign
            if (!s.empty() && (s[0] == '+' || s[0] == '-'))
            {
                result.data()[i] = s[0] + std::string(pad, '0') + s.substr(1);
            }
            else
            {
                result.data()[i] = std::string(pad, '0') + s;
            }
        }
    }
    return result;
}

/**
 * @brief Left-justify strings.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.ljust.html
 *
 * @param a Input string array
 * @param width Minimum width of resulting string
 * @param fillchar Padding character (default: space)
 * @return Array with left-justified strings
 */
NP_API inline auto ljust(const ndarray<std::string> &a, int width, char fillchar = ' ') -> ndarray<std::string>
{
    ndarray<std::string> result = empty<std::string>(a.shape);
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        const std::string &s = a.data()[i];
        if (static_cast<int>(s.size()) >= width)
        {
            result.data()[i] = s;
        }
        else
        {
            int pad = width - static_cast<int>(s.size());
            result.data()[i] = s + std::string(pad, fillchar);
        }
    }
    return result;
}

/**
 * @brief Right-justify strings.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.rjust.html
 *
 * @param a Input string array
 * @param width Minimum width of resulting string
 * @param fillchar Padding character (default: space)
 * @return Array with right-justified strings
 */
NP_API inline auto rjust(const ndarray<std::string> &a, int width, char fillchar = ' ') -> ndarray<std::string>
{
    ndarray<std::string> result = empty<std::string>(a.shape);
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        const std::string &s = a.data()[i];
        if (static_cast<int>(s.size()) >= width)
        {
            result.data()[i] = s;
        }
        else
        {
            int pad = width - static_cast<int>(s.size());
            result.data()[i] = std::string(pad, fillchar) + s;
        }
    }
    return result;
}

/**
 * @brief Replace occurrences of substring with new substring.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.replace.html
 *
 * @param a Input string array
 * @param old Substring to replace
 * @param new_str Replacement substring
 * @param count Maximum number of replacements (-1 = all)
 * @return Array with replaced strings
 */
NP_API inline auto replace(const ndarray<std::string> &a, const std::string &old, const std::string &new_str,
                           int count = -1) -> ndarray<std::string>
{
    ndarray<std::string> result = empty<std::string>(a.shape);
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        const std::string &s_in = a.data()[i];
        if (old.empty())
        {
            result.data()[i] = s_in;
            continue;
        }
        // Micro-optimized: KMP find all occurrences in O(n+m), then single allocation
        std::vector<std::size_t> positions;
        positions.reserve(8);
        // KMP prefix for old
        auto pi = detail::kmp_prefix(old);
        int j = 0;
        for (std::size_t p = 0; p < s_in.size(); ++p)
        {
            while (j > 0 && s_in[p] != old[static_cast<std::size_t>(j)])
                j = pi[static_cast<std::size_t>(j) - 1];
            if (s_in[p] == old[static_cast<std::size_t>(j)])
                ++j;
            if (j == static_cast<int>(old.size()))
            {
                positions.push_back(p - old.size() + 1);
                j = 0; // non-overlapping like numpy
                if (count >= 0 && static_cast<int>(positions.size()) >= count)
                    break;
            }
        }
        if (positions.empty())
        {
            result.data()[i] = s_in;
            continue;
        }
        std::size_t new_len = s_in.size() + positions.size() * (new_str.size() - old.size());
        std::string out;
        out.reserve(new_len);
        std::size_t prev = 0;
        for (std::size_t idx = 0; idx < positions.size(); ++idx)
        {
            std::size_t pos = positions[idx];
            out.append(s_in, prev, pos - prev);
            out += new_str;
            prev = pos + old.size();
        }
        out.append(s_in, prev, std::string::npos);
        result.data()[i] = std::move(out);
    }
    return result;
}

/* Comparison Functions */

/**
 * @brief Return (x1 == x2) element-wise.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.equal.html
 *
 * @param x1 First string array
 * @param x2 Second string array
 * @return Boolean array
 */
NP_API inline auto equal(const ndarray<std::string> &x1, const ndarray<std::string> &x2) -> ndarray<bool>
{
    if (x1.shape != x2.shape)
    {
        throw std::invalid_argument("equal: arrays must have the same shape");
    }
    ndarray<bool> result = empty<bool>(x1.shape);
    for (std::size_t i = 0; i < x1.size(); ++i)
    {
        result.data()[i] = (x1.data()[i] == x2.data()[i]);
    }
    return result;
}

/**
 * @brief Return (x1 != x2) element-wise.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.not_equal.html
 *
 * @param x1 First string array
 * @param x2 Second string array
 * @return Boolean array
 */
NP_API inline auto not_equal(const ndarray<std::string> &x1, const ndarray<std::string> &x2) -> ndarray<bool>
{
    if (x1.shape != x2.shape)
    {
        throw std::invalid_argument("not_equal: arrays must have the same shape");
    }
    ndarray<bool> result = empty<bool>(x1.shape);
    for (std::size_t i = 0; i < x1.size(); ++i)
    {
        result.data()[i] = (x1.data()[i] != x2.data()[i]);
    }
    return result;
}

/**
 * @brief Return (x1 >= x2) element-wise.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.greater_equal.html
 *
 * @param x1 First string array
 * @param x2 Second string array
 * @return Boolean array
 */
NP_API inline auto greater_equal(const ndarray<std::string> &x1, const ndarray<std::string> &x2) -> ndarray<bool>
{
    if (x1.shape != x2.shape)
    {
        throw std::invalid_argument("greater_equal: arrays must have the same shape");
    }
    ndarray<bool> result = empty<bool>(x1.shape);
    for (std::size_t i = 0; i < x1.size(); ++i)
    {
        result.data()[i] = (x1.data()[i] >= x2.data()[i]);
    }
    return result;
}

/**
 * @brief Return (x1 <= x2) element-wise.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.less_equal.html
 *
 * @param x1 First string array
 * @param x2 Second string array
 * @return Boolean array
 */
NP_API inline auto less_equal(const ndarray<std::string> &x1, const ndarray<std::string> &x2) -> ndarray<bool>
{
    if (x1.shape != x2.shape)
    {
        throw std::invalid_argument("less_equal: arrays must have the same shape");
    }
    ndarray<bool> result = empty<bool>(x1.shape);
    for (std::size_t i = 0; i < x1.size(); ++i)
    {
        result.data()[i] = (x1.data()[i] <= x2.data()[i]);
    }
    return result;
}

/**
 * @brief Return (x1 > x2) element-wise.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.greater.html
 *
 * @param x1 First string array
 * @param x2 Second string array
 * @return Boolean array
 */
NP_API inline auto greater(const ndarray<std::string> &x1, const ndarray<std::string> &x2) -> ndarray<bool>
{
    if (x1.shape != x2.shape)
    {
        throw std::invalid_argument("greater: arrays must have the same shape");
    }
    ndarray<bool> result = empty<bool>(x1.shape);
    for (std::size_t i = 0; i < x1.size(); ++i)
    {
        result.data()[i] = (x1.data()[i] > x2.data()[i]);
    }
    return result;
}

/**
 * @brief Return (x1 < x2) element-wise.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.less.html
 *
 * @param x1 First string array
 * @param x2 Second string array
 * @return Boolean array
 */
NP_API inline auto less(const ndarray<std::string> &x1, const ndarray<std::string> &x2) -> ndarray<bool>
{
    if (x1.shape != x2.shape)
    {
        throw std::invalid_argument("less: arrays must have the same shape");
    }
    ndarray<bool> result = empty<bool>(x1.shape);
    for (std::size_t i = 0; i < x1.size(); ++i)
    {
        result.data()[i] = (x1.data()[i] < x2.data()[i]);
    }
    return result;
}

/* Information Functions */

/**
 * @brief Returns the number of non-overlapping occurrences of substring.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.count.html
 *
 * @param a Input string array
 * @param sub Substring to count
 * @param start Start position (optional)
 * @param end End position (optional)
 * @return Array of counts
 */
NP_API inline auto count(const ndarray<std::string> &a, const std::string &sub, int start = 0, int end = -1)
    -> ndarray<int>
{
    ndarray<int> result = empty<int>(a.shape);
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        const std::string &s = a.data()[i];
        // Micro-optimized: KMP/SAM counting O(n+m) vs naive O(n*m)
        result.data()[i] = detail::optimized_count(s, sub, start, end);
    }
    return result;
}

/**
 * @brief Returns a boolean array which is True where the string element ends
 * with suffix.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.endswith.html
 *
 * @param a Input string array
 * @param suffix Suffix to check
 * @param start Start position (optional)
 * @param end End position (optional)
 * @return Boolean array
 */
NP_API inline auto endswith(const ndarray<std::string> &a, const std::string &suffix, int start = 0, int end = -1)
    -> ndarray<bool>
{
    ndarray<bool> result = empty<bool>(a.shape);
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        const std::string &s = a.data()[i];
        int len = static_cast<int>(s.size());
        int e = (end < 0) ? len : std::min(end, len);
        int st = std::max(0, start);
        int slice_len = e - st;
        // Micro-optimized: avoid substr allocation, direct compare
        if (slice_len < 0)
            slice_len = 0;
        if (static_cast<std::size_t>(slice_len) < suffix.size())
        {
            result.data()[i] = false;
        }
        else
        {
            // compare suffix at end of slice without allocation
            result.data()[i] = s.compare(static_cast<std::size_t>(e - suffix.size()), suffix.size(), suffix) == 0;
        }
    }
    return result;
}

/**
 * @brief Returns a boolean array which is True where the string element starts
 * with prefix.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.startswith.html
 *
 * @param a Input string array
 * @param prefix Prefix to check
 * @param start Start position (optional)
 * @param end End position (optional)
 * @return Boolean array
 */
NP_API inline auto startswith(const ndarray<std::string> &a, const std::string &prefix, int start = 0, int end = -1)
    -> ndarray<bool>
{
    ndarray<bool> result = empty<bool>(a.shape);
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        const std::string &s = a.data()[i];
        int len = static_cast<int>(s.size());
        int e = (end < 0) ? len : std::min(end, len);
        int st = std::max(0, start);
        int slice_len = e - st;
        if (slice_len < 0)
            slice_len = 0;
        if (static_cast<std::size_t>(slice_len) < prefix.size())
        {
            result.data()[i] = false;
        }
        else
        {
            result.data()[i] = s.compare(static_cast<std::size_t>(st), prefix.size(), prefix) == 0;
        }
    }
    return result;
}

/**
 * @brief Finds the lowest index of the substring.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.find.html
 *
 * @param a Input string array
 * @param sub Substring to find
 * @param start Start position (optional)
 * @param end End position (optional)
 * @return Array of indices (or -1 if not found)
 */
NP_API inline auto find(const ndarray<std::string> &a, const std::string &sub, int start = 0, int end = -1)
    -> ndarray<int>
{
    ndarray<int> result = empty<int>(a.shape);
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        const std::string &s = a.data()[i];
        int len = static_cast<int>(s.size());
        int e = (end < 0) ? len : std::min(end, len);
        int st = std::max(0, start);
        if (st > e || sub.size() > static_cast<std::size_t>(e - st))
        {
            result.data()[i] = -1;
            continue;
        }
        // Micro-optimized KMP/SAM O(n+m) vs naive O(n*m) worst case
        std::size_t pos = detail::optimized_find(s, sub, static_cast<std::size_t>(st));
        if (pos != std::string::npos && pos + sub.size() <= static_cast<std::size_t>(e))
        {
            result.data()[i] = static_cast<int>(pos);
        }
        else
        {
            result.data()[i] = -1;
        }
    }
    return result;
}

/**
 * @brief Finds the highest index of the substring.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.rfind.html
 *
 * @param a Input string array
 * @param sub Substring to find
 * @param start Start position (optional)
 * @param end End position (optional)
 * @return Array of indices (or -1 if not found)
 */
NP_API inline auto rfind(const ndarray<std::string> &a, const std::string &sub, int start = 0, int end = -1)
    -> ndarray<int>
{
    ndarray<int> result = empty<int>(a.shape);
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        const std::string &s = a.data()[i];
        int len = static_cast<int>(s.size());
        int e = (end < 0) ? len : std::min(end, len);
        int st = std::max(0, start);
        if (st >= e || sub.empty())
        {
            if (sub.empty())
                result.data()[i] = e;
            else
                result.data()[i] = -1;
            continue;
        }
        // Micro-optimized reverse KMP O(n+m) vs naive O(n*m)
        std::size_t pos = detail::kmp_rfind(s, sub, static_cast<std::size_t>(st), static_cast<std::size_t>(e));
        if (pos != std::string::npos)
        {
            result.data()[i] = static_cast<int>(pos);
        }
        else
        {
            result.data()[i] = -1;
        }
    }
    return result;
}

/**
 * @brief Return the string length element-wise.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.str_len.html
 *
 * @param a Input string array
 * @return Array of string lengths
 */
NP_API inline auto str_len(const ndarray<std::string> &a) -> ndarray<int>
{
    ndarray<int> result = empty<int>(a.shape);
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        result.data()[i] = static_cast<int>(a.data()[i].size());
    }
    return result;
}

/* Testing Functions */

/**
 * @brief Returns true for each element if all characters are alphabetic.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.isalpha.html
 *
 * @param a Input string array
 * @return Boolean array
 */
NP_API inline auto isalpha(const ndarray<std::string> &a) -> ndarray<bool>
{
    ndarray<bool> result = empty<bool>(a.shape);
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        const std::string &s = a.data()[i];
        result.data()[i] =
            !s.empty() && std::all_of(s.begin(), s.end(), [](unsigned char c) { return std::isalpha(c); });
    }
    return result;
}

/**
 * @brief Returns true for each element if all characters are alphanumeric.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.isalnum.html
 *
 * @param a Input string array
 * @return Boolean array
 */
NP_API inline auto isalnum(const ndarray<std::string> &a) -> ndarray<bool>
{
    ndarray<bool> result = empty<bool>(a.shape);
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        const std::string &s = a.data()[i];
        result.data()[i] =
            !s.empty() && std::all_of(s.begin(), s.end(), [](unsigned char c) { return std::isalnum(c); });
    }
    return result;
}

/**
 * @brief Returns true for each element if all characters are digits.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.isdigit.html
 *
 * @param a Input string array
 * @return Boolean array
 */
NP_API inline auto isdigit(const ndarray<std::string> &a) -> ndarray<bool>
{
    ndarray<bool> result = empty<bool>(a.shape);
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        const std::string &s = a.data()[i];
        result.data()[i] =
            !s.empty() && std::all_of(s.begin(), s.end(), [](unsigned char c) { return std::isdigit(c); });
    }
    return result;
}

/**
 * @brief Returns true for each element if all cased characters are lowercase.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.islower.html
 *
 * @param a Input string array
 * @return Boolean array
 */
NP_API inline auto islower(const ndarray<std::string> &a) -> ndarray<bool>
{
    ndarray<bool> result = empty<bool>(a.shape);
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        const std::string &s = a.data()[i];
        bool has_cased = false;
        bool all_lower = true;
        for (char c : s)
        {
            if (std::isalpha(static_cast<unsigned char>(c)))
            {
                has_cased = true;
                if (!std::islower(static_cast<unsigned char>(c)))
                {
                    all_lower = false;
                    break;
                }
            }
        }
        result.data()[i] = has_cased && all_lower;
    }
    return result;
}

/**
 * @brief Returns true for each element if all cased characters are uppercase.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.isupper.html
 *
 * @param a Input string array
 * @return Boolean array
 */
NP_API inline auto isupper(const ndarray<std::string> &a) -> ndarray<bool>
{
    ndarray<bool> result = empty<bool>(a.shape);
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        const std::string &s = a.data()[i];
        bool has_cased = false;
        bool all_upper = true;
        for (char c : s)
        {
            if (std::isalpha(static_cast<unsigned char>(c)))
            {
                has_cased = true;
                if (!std::isupper(static_cast<unsigned char>(c)))
                {
                    all_upper = false;
                    break;
                }
            }
        }
        result.data()[i] = has_cased && all_upper;
    }
    return result;
}

/**
 * @brief Returns true for each element if all characters are whitespace.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.isspace.html
 *
 * @param a Input string array
 * @return Boolean array
 */
NP_API inline auto isspace(const ndarray<std::string> &a) -> ndarray<bool>
{
    ndarray<bool> result = empty<bool>(a.shape);
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        const std::string &s = a.data()[i];
        result.data()[i] =
            !s.empty() && std::all_of(s.begin(), s.end(), [](unsigned char c) { return std::isspace(c); });
    }
    return result;
}

/**
 * @brief Returns true for each element if string is in titlecase.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.istitle.html
 *
 * @param a Input string array
 * @return Boolean array
 */
NP_API inline auto istitle(const ndarray<std::string> &a) -> ndarray<bool>
{
    ndarray<bool> result = empty<bool>(a.shape);
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        const std::string &s = a.data()[i];
        bool in_word = false;
        bool has_cased = false;
        bool is_title = true;

        for (char c : s)
        {
            bool is_alpha = std::isalpha(static_cast<unsigned char>(c));
            if (is_alpha)
            {
                has_cased = true;
                if (in_word)
                {
                    if (!std::islower(static_cast<unsigned char>(c)))
                    {
                        is_title = false;
                        break;
                    }
                }
                else
                {
                    if (!std::isupper(static_cast<unsigned char>(c)))
                    {
                        is_title = false;
                        break;
                    }
                    in_word = true;
                }
            }
            else
            {
                in_word = false;
            }
        }
        result.data()[i] = has_cased && is_title;
    }
    return result;
}

/**
 * @brief Returns true for each element if all characters are decimal.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.isdecimal.html
 *
 * @param a Input string array
 * @return Boolean array
 */
NP_API inline auto isdecimal(const ndarray<std::string> &a) -> ndarray<bool>
{
    ndarray<bool> result = empty<bool>(a.shape);
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        const std::string &s = a.data()[i];
        // C++ doesn't have direct isdecimal; use isdigit as approximation
        result.data()[i] =
            !s.empty() && std::all_of(s.begin(), s.end(), [](unsigned char c) { return std::isdigit(c); });
    }
    return result;
}

/**
 * @brief Returns true for each element if all characters are numeric.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.isnumeric.html
 *
 * @param a Input string array
 * @return Boolean array
 */
NP_API inline auto isnumeric(const ndarray<std::string> &a) -> ndarray<bool>
{
    ndarray<bool> result = empty<bool>(a.shape);
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        const std::string &s = a.data()[i];
        // C++ doesn't have direct isnumeric; use isdigit as approximation
        result.data()[i] =
            !s.empty() && std::all_of(s.begin(), s.end(), [](unsigned char c) { return std::isdigit(c); });
    }
    return result;
}

/* Split/Join Functions */

/**
 * @brief Join strings with separator.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.join.html
 *
 * @param sep Separator string array
 * @param seq Sequence string array
 * @return Array with joined strings
 */
NP_API inline auto join(const ndarray<std::string> &sep, const ndarray<std::string> &seq) -> ndarray<std::string>
{
    if (sep.shape != seq.shape)
    {
        throw std::invalid_argument("join: arrays must have the same shape");
    }
    ndarray<std::string> result = empty<std::string>(sep.shape);
    for (std::size_t i = 0; i < sep.size(); ++i)
    {
        const std::string &s = seq.data()[i];
        const std::string &separator = sep.data()[i];
        std::string joined;
        for (std::size_t j = 0; j < s.size(); ++j)
        {
            if (j > 0)
                joined += separator;
            joined += s[j];
        }
        result.data()[i] = joined;
    }
    return result;
}

/**
 * @brief Expand tabs in each string element.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.expandtabs.html
 *
 * @param a Input string array
 * @param tabsize Number of spaces per tab (default: 8)
 * @return Array with expanded tabs
 */
NP_API inline auto expandtabs(const ndarray<std::string> &a, int tabsize = 8) -> ndarray<std::string>
{
    ndarray<std::string> result = empty<std::string>(a.shape);
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        const std::string &s = a.data()[i];
        std::string expanded;
        int col = 0;
        for (char c : s)
        {
            if (c == '\t')
            {
                int spaces = tabsize - (col % tabsize);
                expanded.append(spaces, ' ');
                col += spaces;
            }
            else if (c == '\n' || c == '\r')
            {
                expanded += c;
                col = 0;
            }
            else
            {
                expanded += c;
                ++col;
            }
        }
        result.data()[i] = expanded;
    }
    return result;
}

/**
 * @brief Partition each element around sep.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.partition.html
 *
 * @param a Input string array
 * @param sep Separator string
 * @return Array of tuples (before, sep, after) - flattened as 3x larger array
 */
NP_API inline auto partition(const ndarray<std::string> &a, const std::string &sep) -> ndarray<std::string>
{
    // Returns array with 3x elements: [before0, sep0, after0, before1, sep1,
    // after1, ...] Micro-optimized with KMP O(n+m)
    std::vector<int> new_shape = a.shape;
    new_shape.back() *= 3;
    ndarray<std::string> result = empty<std::string>(new_shape);

    for (std::size_t i = 0; i < a.size(); ++i)
    {
        const std::string &s = a.data()[i];
        std::size_t pos = detail::optimized_find(s, sep, 0);
        if (pos != std::string::npos)
        {
            result.data()[i * 3] = s.substr(0, pos);
            result.data()[i * 3 + 1] = sep;
            result.data()[i * 3 + 2] = s.substr(pos + sep.size());
        }
        else
        {
            result.data()[i * 3] = s;
            result.data()[i * 3 + 1] = "";
            result.data()[i * 3 + 2] = "";
        }
    }
    return result;
}

/**
 * @brief Partition each element around rightmost sep.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.rpartition.html
 *
 * @param a Input string array
 * @param sep Separator string
 * @return Array of tuples (before, sep, after) - flattened as 3x larger array
 */
NP_API inline auto rpartition(const ndarray<std::string> &a, const std::string &sep) -> ndarray<std::string>
{
    std::vector<int> new_shape = a.shape;
    new_shape.back() *= 3;
    ndarray<std::string> result = empty<std::string>(new_shape);

    for (std::size_t i = 0; i < a.size(); ++i)
    {
        const std::string &s = a.data()[i];
        std::size_t pos = detail::kmp_rfind(s, sep, 0, s.size());
        if (pos != std::string::npos)
        {
            result.data()[i * 3] = s.substr(0, pos);
            result.data()[i * 3 + 1] = sep;
            result.data()[i * 3 + 2] = s.substr(pos + sep.size());
        }
        else
        {
            result.data()[i * 3] = "";
            result.data()[i * 3 + 1] = "";
            result.data()[i * 3 + 2] = s;
        }
    }
    return result;
}

/**
 * @brief Like find, but raises ValueError when substring is not found.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.index.html
 *
 * @param a Input string array
 * @param sub Substring to find
 * @param start Start position (optional)
 * @param end End position (optional)
 * @return Array of indices
 * @throws std::invalid_argument if substring not found
 */
NP_API inline auto index(const ndarray<std::string> &a, const std::string &sub, int start = 0, int end = -1)
    -> ndarray<int>
{
    ndarray<int> result = find(a, sub, start, end);
    for (std::size_t i = 0; i < result.size(); ++i)
    {
        if (result.data()[i] == -1)
        {
            throw std::invalid_argument("index: substring not found");
        }
    }
    return result;
}

/**
 * @brief Like rfind, but raises ValueError when substring is not found.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.rindex.html
 *
 * @param a Input string array
 * @param sub Substring to find
 * @param start Start position (optional)
 * @param end End position (optional)
 * @return Array of indices
 * @throws std::invalid_argument if substring not found
 */
NP_API inline auto rindex(const ndarray<std::string> &a, const std::string &sub, int start = 0, int end = -1)
    -> ndarray<int>
{
    ndarray<int> result = rfind(a, sub, start, end);
    for (std::size_t i = 0; i < result.size(); ++i)
    {
        if (result.data()[i] == -1)
        {
            throw std::invalid_argument("rindex: substring not found");
        }
    }
    return result;
}

/**
 * @brief Split strings around given separator.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.split.html
 *
 * Note: Returns flattened array of all split parts. In Python numpy, this
 * returns object arrays of lists. C++ implementation returns concatenated
 * results.
 *
 * @param a Input string array
 * @param sep Separator (if empty, splits on whitespace)
 * @param maxsplit Maximum number of splits (-1 = unlimited)
 * @return Flattened array of split strings
 */
NP_API inline auto split(const ndarray<std::string> &a, const std::string &sep = "", int maxsplit = -1)
    -> std::vector<ndarray<std::string>>
{
    std::vector<ndarray<std::string>> out;
    out.reserve(a.size());
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        const std::string &s = a.data()[i];
        std::vector<std::string> parts;
        if (sep.empty())
        {
            std::istringstream iss(s);
            std::string word;
            while (iss >> word)
            {
                parts.push_back(word);
                if (maxsplit >= 0 && static_cast<int>(parts.size()) > maxsplit)
                    break;
            }
        }
        else
        {
            // Micro-optimized KMP O(n+m) per string
            std::size_t start = 0;
            int count = 0;
            auto pi = detail::kmp_prefix(sep);
            int j = 0;
            for (std::size_t p = 0; p < s.size(); ++p)
            {
                while (j > 0 && s[p] != sep[static_cast<std::size_t>(j)])
                    j = pi[static_cast<std::size_t>(j) - 1];
                if (s[p] == sep[static_cast<std::size_t>(j)])
                    ++j;
                if (j == static_cast<int>(sep.size()))
                {
                    std::size_t pos = p - sep.size() + 1;
                    if (pos < start)
                        continue;
                    parts.push_back(s.substr(start, pos - start));
                    start = pos + sep.size();
                    j = 0;
                    ++count;
                    if (maxsplit >= 0 && count >= maxsplit)
                        break;
                    // adjust p to start-1 to avoid overlapping re-scan, but we already reset j
                }
                if (maxsplit >= 0 && count >= maxsplit)
                    break;
            }
            parts.push_back(s.substr(start));
        }
        ndarray<std::string> arr = empty<std::string>(std::vector<int>{static_cast<int>(parts.size())});
        arr.data() = parts;
        out.push_back(std::move(arr));
    }
    return out;
}

// Backward-compat flattened overload (deprecated)
NP_API inline auto split_flattened(const ndarray<std::string> &a, const std::string &sep = "", int maxsplit = -1)
    -> ndarray<std::string>
{
    auto grouped = split(a, sep, maxsplit);
    std::size_t total = 0;
    for (auto &g : grouped)
        total += g.size();
    ndarray<std::string> res = empty<std::string>(std::vector<int>{static_cast<int>(total)});
    std::size_t p = 0;
    for (auto &g : grouped)
        for (std::size_t i = 0; i < g.size(); ++i)
            res.data()[p++] = g.data()[i];
    return res;
}

/**
 * @brief Split strings around given separator from the right.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.rsplit.html
 *
 * @param a Input string array
 * @param sep Separator (if empty, splits on whitespace)
 * @param maxsplit Maximum number of splits (-1 = unlimited)
 * @return Flattened array of split strings
 */
NP_API inline auto rsplit(const ndarray<std::string> &a, const std::string &sep = "", int maxsplit = -1)
    -> std::vector<ndarray<std::string>>
{
    std::vector<ndarray<std::string>> out;
    out.reserve(a.size());
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        const std::string &s = a.data()[i];
        std::vector<std::string> parts;
        if (sep.empty())
        {
            std::istringstream iss(s);
            std::string word;
            while (iss >> word)
                parts.push_back(word);
        }
        else
        {
            // Micro-optimized: KMP find all positions, then slice from right without O(n^2)
            // inserts
            std::vector<std::size_t> positions;
            auto pi = detail::kmp_prefix(sep);
            int j = 0;
            for (std::size_t p = 0; p < s.size(); ++p)
            {
                while (j > 0 && s[p] != sep[static_cast<std::size_t>(j)])
                    j = pi[static_cast<std::size_t>(j) - 1];
                if (s[p] == sep[static_cast<std::size_t>(j)])
                    ++j;
                if (j == static_cast<int>(sep.size()))
                {
                    positions.push_back(p - sep.size() + 1);
                    j = 0;
                }
            }
            if (positions.empty())
            {
                parts.push_back(s);
            }
            else
            {
                // Determine how many splits from right
                int total = static_cast<int>(positions.size());
                int take = (maxsplit < 0) ? total : std::min(total, maxsplit);
                int start_idx = total - take;
                std::size_t end = s.size();
                // Build parts from right to left, then reverse
                std::vector<std::string> rev;
                rev.reserve(take + 1);
                for (int idx = total - 1; idx >= start_idx; --idx)
                {
                    std::size_t pos = positions[idx];
                    rev.push_back(s.substr(pos + sep.size(), end - pos - sep.size()));
                    end = pos;
                }
                rev.push_back(s.substr(0, end));
                parts.reserve(rev.size());
                for (auto it = rev.rbegin(); it != rev.rend(); ++it)
                    parts.push_back(std::move(*it));
            }
        }
        ndarray<std::string> arr = empty<std::string>(std::vector<int>{static_cast<int>(parts.size())});
        arr.data() = parts;
        out.push_back(std::move(arr));
    }
    return out;
}

NP_API inline auto rsplit_flattened(const ndarray<std::string> &a, const std::string &sep = "", int maxsplit = -1)
    -> ndarray<std::string>
{
    auto grouped = rsplit(a, sep, maxsplit);
    std::size_t total = 0;
    for (auto &g : grouped)
        total += g.size();
    ndarray<std::string> res = empty<std::string>(std::vector<int>{static_cast<int>(total)});
    std::size_t p = 0;
    for (auto &g : grouped)
        for (std::size_t i = 0; i < g.size(); ++i)
            res.data()[p++] = g.data()[i];
    return res;
}

/**
 * @brief Split strings on line boundaries.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.splitlines.html
 *
 * @param a Input string array
 * @param keepends Keep line endings (default: false)
 * @return Flattened array of lines
 */
NP_API inline auto splitlines(const ndarray<std::string> &a, bool keepends = false) -> std::vector<ndarray<std::string>>
{
    std::vector<ndarray<std::string>> out;
    out.reserve(a.size());
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        const std::string &s = a.data()[i];
        std::istringstream iss(s);
        std::string line;
        std::vector<std::string> parts;
        while (std::getline(iss, line))
        {
            if (keepends)
                line += '\n';
            parts.push_back(line);
        }
        // Handle trailing newline producing extra empty? mimic numpy: if s ends with \n
        // and keepends false, don't add empty
        ndarray<std::string> arr = empty<std::string>(std::vector<int>{static_cast<int>(parts.size())});
        arr.data() = parts;
        out.push_back(std::move(arr));
    }
    return out;
}

NP_API inline auto splitlines_flattened(const ndarray<std::string> &a, bool keepends = false) -> ndarray<std::string>
{
    auto grouped = splitlines(a, keepends);
    std::size_t total = 0;
    for (auto &g : grouped)
        total += g.size();
    ndarray<std::string> res = empty<std::string>(std::vector<int>{static_cast<int>(total)});
    std::size_t p = 0;
    for (auto &g : grouped)
        for (std::size_t i = 0; i < g.size(); ++i)
            res.data()[p++] = g.data()[i];
    return res;
}

/**
 * @brief Apply str.translate() element-wise.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.translate.html
 *
 * Simplified implementation: table is a string of 256 characters for direct
 * mapping.
 *
 * @param a Input string array
 * @param table Translation table (256-char string)
 * @param deletechars Characters to delete
 * @return Array with translated strings
 */
NP_API inline auto translate(const ndarray<std::string> &a, const std::string &table,
                             const std::string &deletechars = "") -> ndarray<std::string>
{
    ndarray<std::string> result = empty<std::string>(a.shape);
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        std::string s = a.data()[i];
        std::string translated;
        for (char c : s)
        {
            if (deletechars.find(c) != std::string::npos)
            {
                continue;
            }
            unsigned char uc = static_cast<unsigned char>(c);
            if (uc < table.size())
            {
                translated += table[uc];
            }
            else
            {
                translated += c;
            }
        }
        result.data()[i] = translated;
    }
    return result;
}

/* Encode/Decode — UTF-8 validation with errors handling */

namespace detail
{
// Minimal UTF-8 validator (RFC 3629) — checks continuation bytes and overlongs
inline bool is_valid_utf8(const std::string &s) noexcept
{
    std::size_t i = 0, n = s.size();
    while (i < n)
    {
        unsigned char c = static_cast<unsigned char>(s[i]);
        std::size_t len = 0;
        if ((c & 0x80) == 0)
            len = 1;
        else if ((c & 0xE0) == 0xC0)
            len = 2;
        else if ((c & 0xF0) == 0xE0)
            len = 3;
        else if ((c & 0xF8) == 0xF0)
            len = 4;
        else
            return false;
        if (i + len > n)
            return false;
        for (std::size_t j = 1; j < len; ++j)
            if ((static_cast<unsigned char>(s[i + j]) & 0xC0) != 0x80)
                return false;
        // Reject overlong and surrogates (simplified)
        if (len == 2 && c < 0xC2)
            return false;
        if (len == 3 && c == 0xE0 && static_cast<unsigned char>(s[i + 1]) < 0xA0)
            return false;
        if (len == 4 && c == 0xF0 && static_cast<unsigned char>(s[i + 1]) < 0x90)
            return false;
        i += len;
    }
    return true;
}

inline bool is_valid_ascii(const std::string &s) noexcept
{
    for (unsigned char c : s)
        if (c & 0x80)
            return false;
    return true;
}

inline std::string handle_encode_error(const std::string &s, const std::string &errors, bool valid)
{
    if (valid)
        return s;
    if (errors == "strict")
        throw std::invalid_argument("encode: invalid string for encoding");
    if (errors == "ignore")
        return {}; // drop invalid — caller will skip element
    if (errors == "replace")
        return "?";
    // xmlcharrefreplace / backslashreplace — simplified to "?"
    return "?";
}
} // namespace detail

/**
 * @brief Calls str.encode() element-wise with UTF-8 validation.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.encode.html
 *
 * In Python this converts unicode → bytes; here we validate that the
 * C++ string is valid for `encoding` and handle `errors` (`strict`/`ignore`/`replace`).
 *
 * @param a Input string array
 * @param encoding Encoding name (`utf-8`, `ascii`, `latin1`)
 * @param errors Error handling (`strict`, `ignore`, `replace`)
 * @return Encoded copy (still std::string, as C++ is byte-based)
 */
NP_API inline auto encode(const ndarray<std::string> &a, const std::string &encoding = "utf-8",
                          const std::string &errors = "strict") -> ndarray<std::string>
{
    std::string enc = encoding;
    std::transform(enc.begin(), enc.end(), enc.begin(), ::tolower);
    bool is_utf8 = (enc == "utf-8" || enc == "utf8");
    bool is_ascii = (enc == "ascii");
    // latin1 always valid for 0..255 bytes
    ndarray<std::string> out(a.shape);
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        const std::string &s = a.data()[i];
        bool valid = true;
        if (is_utf8)
            valid = detail::is_valid_utf8(s);
        else if (is_ascii)
            valid = detail::is_valid_ascii(s);
        // latin1/others: always valid
        if (!valid)
        {
            std::string rep = detail::handle_encode_error(s, errors, false);
            if (errors == "ignore" && rep.empty())
                out.data()[i] = "";
            else if (errors == "strict")
                throw std::invalid_argument("encode: '" + s + "' not valid for " + encoding);
            else
                out.data()[i] = rep;
        }
        else
        {
            out.data()[i] = s;
        }
    }
    return out;
}

/**
 * @brief Calls str.decode() element-wise with validation.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.decode.html
 *
 * Validates bytes for `encoding`; here std::string already holds bytes,
 * so we just validate and handle `errors`.
 *
 * @param a Input string array
 * @param encoding Encoding name
 * @param errors Error handling
 * @return Decoded copy
 */
NP_API inline auto decode(const ndarray<std::string> &a, const std::string &encoding = "utf-8",
                          const std::string &errors = "strict") -> ndarray<std::string>
{
    // Decode is symmetric to encode for our byte-based std::string
    return encode(a, encoding, errors);
}

/* Compare Function */

/**
 * @brief Performs element-wise comparison of two string arrays.
 *
 * Reference:
 * numpy-reference/reference/generated/numpy.char.compare_chararrays.html
 *
 * @param x1 First string array
 * @param x2 Second string array
 * @param cmp Comparison operator ("==", "!=", "<", "<=", ">", ">=")
 * @param rstrip Strip trailing whitespace before comparison
 * @return Boolean array – true where comparison holds
 */
NP_API inline auto compare_chararrays(const ndarray<std::string> &x1, const ndarray<std::string> &x2,
                                      const std::string &cmp, bool rstrip = false) -> ndarray<bool>
{
    if (x1.shape != x2.shape)
    {
        throw std::invalid_argument("compare_chararrays: arrays must have the same shape");
    }

    ndarray<bool> result = empty<bool>(x1.shape);
    for (std::size_t i = 0; i < x1.size(); ++i)
    {
        std::string s1 = x1.data()[i];
        std::string s2 = x2.data()[i];

        if (rstrip)
        {
            auto p1 = s1.find_last_not_of(" \t\n\r");
            if (p1 == std::string::npos)
                s1.clear();
            else
                s1.erase(p1 + 1);
            auto p2 = s2.find_last_not_of(" \t\n\r");
            if (p2 == std::string::npos)
                s2.clear();
            else
                s2.erase(p2 + 1);
        }

        int cmp_result = s1.compare(s2);
        if (cmp == "==")
        {
            result.data()[i] = (cmp_result == 0);
        }
        else if (cmp == "!=")
        {
            result.data()[i] = (cmp_result != 0);
        }
        else if (cmp == "<")
        {
            result.data()[i] = (cmp_result < 0);
        }
        else if (cmp == "<=")
        {
            result.data()[i] = (cmp_result <= 0);
        }
        else if (cmp == ">")
        {
            result.data()[i] = (cmp_result > 0);
        }
        else if (cmp == ">=")
        {
            result.data()[i] = (cmp_result >= 0);
        }
        else
        {
            throw std::invalid_argument("compare_chararrays: invalid comparison operator");
        }
    }
    return result;
}

/* Creation Functions */

/**
 * @brief Create a character array (ndarray<std::string>).
 *
 * Reference: numpy-reference/reference/generated/numpy.char.array.html
 *
 * @param object Initializer list or vector of strings
 * @return String array
 */
NP_API inline auto array(const std::vector<std::string> &object) -> ndarray<std::string>
{
    ndarray<std::string> result = empty<std::string>(std::vector<int>{static_cast<int>(object.size())});
    result.data() = object;
    return result;
}

/**
 * @brief Convert input to a character array (ndarray<std::string>).
 *
 * Reference: numpy-reference/reference/generated/numpy.char.asarray.html
 *
 * @param a Input array
 * @return String array (copy if already string array)
 */
NP_API inline auto asarray(const ndarray<std::string> &a) -> ndarray<std::string>
{
    return a; /* Already a string array */
}

/* chararray class - deprecated but provided for compatibility */

/**
 * @brief Provides a convenient view on arrays of string type with all char
 * functions directly available as methods.
 *
 * DEPRECATED: This class is deprecated in NumPy 2.5 and should not be used
 * in new code. Use ndarray<std::string> with np::ch functions instead.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.chararray.html
 *
 * This class wraps an ndarray<std::string> and provides all char module
 * functions as convenient methods. It exists for API compatibility but
 * adds no functionality over using free functions directly.
 *
 * Many methods inherited from ndarray (reshape, transpose, etc.) are
 * accessible through the underlying array() method or implicit conversion.
 */
class NP_DEPRECATED("chararray is deprecated; use ndarray<std::string> with np::ch functions") chararray
{
  private:
    ndarray<std::string> data_;

  public:
    /* Constructors */
    explicit chararray(const ndarray<std::string> &arr) : data_(arr)
    {
    }
    explicit chararray(const std::vector<int> &shape) : data_(empty<std::string>(shape))
    {
    }

    /* Access to underlying array */
    ndarray<std::string> &array()
    {
        return data_;
    }
    const ndarray<std::string> &array() const
    {
        return data_;
    }

    /* Implicit conversion to ndarray<std::string> */
    operator ndarray<std::string> &()
    {
        return data_;
    }
    operator const ndarray<std::string> &() const
    {
        return data_;
    }

    /* ndarray properties - expose from wrapped array */
    const std::vector<int> &shape() const
    {
        return data_.shape;
    }
    std::size_t size() const
    {
        return data_.size();
    }
    int ndim() const
    {
        return static_cast<int>(data_.shape.size());
    }

    /* String transformation methods (return chararray for chaining) */
    auto capitalize() const -> chararray
    {
        return chararray(ch::capitalize(data_));
    }
    auto center(int width, char fillchar = ' ') const -> chararray
    {
        return chararray(ch::center(data_, width, fillchar));
    }
    auto lower() const -> chararray
    {
        return chararray(ch::lower(data_));
    }
    auto upper() const -> chararray
    {
        return chararray(ch::upper(data_));
    }
    auto strip(const std::string &chars = " \t\n\r") const -> chararray
    {
        return chararray(ch::strip(data_, chars));
    }
    auto lstrip(const std::string &chars = " \t\n\r") const -> chararray
    {
        return chararray(ch::lstrip(data_, chars));
    }
    auto rstrip(const std::string &chars = " \t\n\r") const -> chararray
    {
        return chararray(ch::rstrip(data_, chars));
    }
    auto swapcase() const -> chararray
    {
        return chararray(ch::swapcase(data_));
    }
    auto title() const -> chararray
    {
        return chararray(ch::title(data_));
    }
    auto zfill(int width) const -> chararray
    {
        return chararray(ch::zfill(data_, width));
    }
    auto ljust(int width, char fillchar = ' ') const -> chararray
    {
        return chararray(ch::ljust(data_, width, fillchar));
    }
    auto rjust(int width, char fillchar = ' ') const -> chararray
    {
        return chararray(ch::rjust(data_, width, fillchar));
    }
    auto replace(const std::string &old, const std::string &new_str, int count = -1) const -> chararray
    {
        return chararray(ch::replace(data_, old, new_str, count));
    }
    auto expandtabs(int tabsize = 8) const -> chararray
    {
        return chararray(ch::expandtabs(data_, tabsize));
    }
    auto encode(const std::string &encoding = "utf-8") const -> chararray
    {
        return chararray(ch::encode(data_, encoding));
    }
    auto decode(const std::string &encoding = "utf-8") const -> chararray
    {
        return chararray(ch::decode(data_, encoding));
    }
    auto translate(const std::string &table) const -> chararray
    {
        return chararray(ch::translate(data_, table));
    }

    /* Splitting methods – grouped per element */
    auto split(const std::string &sep = "", int maxsplit = -1) const -> std::vector<ndarray<std::string>>
    {
        return ch::split(data_, sep, maxsplit);
    }
    auto rsplit(const std::string &sep = "", int maxsplit = -1) const -> std::vector<ndarray<std::string>>
    {
        return ch::rsplit(data_, sep, maxsplit);
    }
    auto splitlines(bool keepends = false) const -> std::vector<ndarray<std::string>>
    {
        return ch::splitlines(data_, keepends);
    }
    auto partition(const std::string &sep) const -> ndarray<std::string>
    {
        return ch::partition(data_, sep);
    }
    auto rpartition(const std::string &sep) const -> ndarray<std::string>
    {
        return ch::rpartition(data_, sep);
    }

    /* Join method */
    auto join(const ndarray<std::string> &seq) const -> ndarray<std::string>
    {
        return ch::join(data_, seq);
    }

    /* Information methods (return int/bool arrays) */
    auto count(const std::string &sub, int start = 0, int end = -1) const -> ndarray<int>
    {
        return ch::count(data_, sub, start, end);
    }
    auto find(const std::string &sub, int start = 0, int end = -1) const -> ndarray<int>
    {
        return ch::find(data_, sub, start, end);
    }
    auto rfind(const std::string &sub, int start = 0, int end = -1) const -> ndarray<int>
    {
        return ch::rfind(data_, sub, start, end);
    }
    auto index(const std::string &sub, int start = 0, int end = -1) const -> ndarray<int>
    {
        return ch::index(data_, sub, start, end);
    }
    auto rindex(const std::string &sub, int start = 0, int end = -1) const -> ndarray<int>
    {
        return ch::rindex(data_, sub, start, end);
    }
    auto str_len() const -> ndarray<int>
    {
        return ch::str_len(data_);
    }

    /* Boolean test methods */
    auto startswith(const std::string &prefix, int start = 0, int end = -1) const -> ndarray<bool>
    {
        return ch::startswith(data_, prefix, start, end);
    }
    auto endswith(const std::string &suffix, int start = 0, int end = -1) const -> ndarray<bool>
    {
        return ch::endswith(data_, suffix, start, end);
    }
    auto isalpha() const -> ndarray<bool>
    {
        return ch::isalpha(data_);
    }
    auto isalnum() const -> ndarray<bool>
    {
        return ch::isalnum(data_);
    }
    auto isdigit() const -> ndarray<bool>
    {
        return ch::isdigit(data_);
    }
    auto isdecimal() const -> ndarray<bool>
    {
        return ch::isdecimal(data_);
    }
    auto isnumeric() const -> ndarray<bool>
    {
        return ch::isnumeric(data_);
    }
    auto islower() const -> ndarray<bool>
    {
        return ch::islower(data_);
    }
    auto isupper() const -> ndarray<bool>
    {
        return ch::isupper(data_);
    }
    auto isspace() const -> ndarray<bool>
    {
        return ch::isspace(data_);
    }
    auto istitle() const -> ndarray<bool>
    {
        return ch::istitle(data_);
    }

    /* Note: ndarray methods like reshape, transpose, copy, etc. are accessible
     * through the underlying array: ca.array().reshape(shape) or through
     * implicit conversion. We don't re-expose all 80+ ndarray methods here
     * as they're not string-specific and chararray is deprecated anyway. */
};

} /* namespace ch */

// NumPy 2.0 alias: numpy.strings (new) and numpy.char (legacy).
// Reference: https://numpy.org/doc/2.2/reference/routines.strings.html
namespace strings = ch;

} /* namespace np */

#endif /* NP_CHAR_HPP */
