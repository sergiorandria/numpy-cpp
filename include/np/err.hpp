/**
 * @file err.hpp
 * @brief Floating-point error handling (np.seterr/geterr/errstate).
 *
 * Reference: https://numpy.org/doc/2.2/reference/routines.err.html
 *
 * Mirrors numpy's `seterr`/`geterr`/`seterrcall`/`geterrcall`/`errstate`.
 * The global state is thread-local to avoid interfering with
 * multi-threaded callers; `errstate` is an RAII guard.
 *
 * @author Sergio Randriamihoatra (sergiorandriamihoatra@gmail.com)
 */
#ifndef NP_ERR_HPP
#define NP_ERR_HPP

#include <functional>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>

#include "api_macros.hpp"
#include "exceptions.hpp"

namespace np
{

namespace err
{

/** @brief How a floating error class is handled.
 *
 * Reference: numpy-reference/reference/generated/numpy.seterr.html
 */
enum class ErrHandling
{
    ignore,
    warn,
    raise,
    call,
    print,
    log
};

NP_NODISCARD inline auto to_string(ErrHandling h) -> std::string
{
    switch (h)
    {
    case ErrHandling::ignore:
        return "ignore";
    case ErrHandling::warn:
        return "warn";
    case ErrHandling::raise:
        return "raise";
    case ErrHandling::call:
        return "call";
    case ErrHandling::print:
        return "print";
    case ErrHandling::log:
        return "log";
    }
    return "ignore";
}

NP_NODISCARD inline auto from_string(const std::string &s) -> ErrHandling
{
    if (s == "ignore")
    {
        return ErrHandling::ignore;
    }
    if (s == "warn")
    {
        return ErrHandling::warn;
    }
    if (s == "raise")
    {
        return ErrHandling::raise;
    }
    if (s == "call")
    {
        return ErrHandling::call;
    }
    if (s == "print")
    {
        return ErrHandling::print;
    }
    if (s == "log")
    {
        return ErrHandling::log;
    }
    throw std::invalid_argument("invalid err handling: " + s);
}

/** @brief Per-class error handling state.
 *
 * Keys: "divide", "over", "under", "invalid".
 */
struct ErrState
{
    ErrHandling divide = ErrHandling::warn;
    ErrHandling over = ErrHandling::warn;
    ErrHandling under = ErrHandling::ignore;
    ErrHandling invalid = ErrHandling::warn;

    auto as_map() const -> std::map<std::string, std::string>
    {
        return {{"divide", to_string(divide)},
                {"over", to_string(over)},
                {"under", to_string(under)},
                {"invalid", to_string(invalid)}};
    }
};

namespace detail
{
inline thread_local ErrState g_state{};
inline thread_local std::function<void(const std::string &, const std::string &)> g_call = nullptr;
} // namespace detail

/**
 * @brief Get current error handling (np.geterr).
 *
 * Reference: numpy-reference/reference/generated/numpy.geterr.html
 */
NP_API inline auto geterr() -> std::map<std::string, std::string>
{
    return detail::g_state.as_map();
}

NP_API inline auto geterr_state() -> ErrState
{
    return detail::g_state;
}

/**
 * @brief Set error handling (np.seterr).
 *
 * Reference: numpy-reference/reference/generated/numpy.seterr.html
 *
 * @param all If set, applies to all classes.
 * @param divide Handling for divide-by-zero.
 * @param over Handling for overflow.
 * @param under Handling for underflow.
 * @param invalid Handling for invalid.
 * @return Previous state.
 */
NP_API inline auto seterr(std::optional<std::string> all = std::nullopt,
                          std::optional<std::string> divide = std::nullopt,
                          std::optional<std::string> over = std::nullopt,
                          std::optional<std::string> under = std::nullopt,
                          std::optional<std::string> invalid = std::nullopt) -> std::map<std::string, std::string>
{
    auto old = geterr();
    if (all.has_value())
    {
        auto h = from_string(*all);
        detail::g_state.divide = h;
        detail::g_state.over = h;
        detail::g_state.under = h;
        detail::g_state.invalid = h;
    }
    if (divide.has_value())
    {
        detail::g_state.divide = from_string(*divide);
    }
    if (over.has_value())
    {
        detail::g_state.over = from_string(*over);
    }
    if (under.has_value())
    {
        detail::g_state.under = from_string(*under);
    }
    if (invalid.has_value())
    {
        detail::g_state.invalid = from_string(*invalid);
    }
    return old;
}

NP_API inline auto seterr(const std::map<std::string, std::string> &kwargs) -> std::map<std::string, std::string>
{
    auto old = geterr();
    for (auto &[k, v] : kwargs)
    {
        if (k == "all")
        {
            auto h = from_string(v);
            detail::g_state.divide = h;
            detail::g_state.over = h;
            detail::g_state.under = h;
            detail::g_state.invalid = h;
        }
        else if (k == "divide")
        {
            detail::g_state.divide = from_string(v);
        }
        else if (k == "over")
        {
            detail::g_state.over = from_string(v);
        }
        else if (k == "under")
        {
            detail::g_state.under = from_string(v);
        }
        else if (k == "invalid")
        {
            detail::g_state.invalid = from_string(v);
        }
        else
        {
            throw std::invalid_argument("seterr: unknown key " + k);
        }
    }
    return old;
}

/**
 * @brief Set callback for `call` handling (np.seterrcall).
 *
 * Reference: numpy-reference/reference/generated/numpy.seterrcall.html
 */
NP_API inline auto seterrcall(std::function<void(const std::string &, const std::string &)> func)
    -> std::function<void(const std::string &, const std::string &)>
{
    auto old = detail::g_call;
    detail::g_call = std::move(func);
    return old;
}

/**
 * @brief Get callback (np.geterrcall).
 *
 * Reference: numpy-reference/reference/generated/numpy.geterrcall.html
 */
NP_API inline auto geterrcall() -> std::function<void(const std::string &, const std::string &)>
{
    return detail::g_call;
}

/**
 * @brief RAII errstate guard (np.errstate).
 *
 * Reference: numpy-reference/reference/generated/numpy.errstate.html
 *
 * Example:
 *   {
 *     np::err::errstate guard({{"divide","raise"}});
 *     // ... floating work
 *   } // restored
 */
class errstate
{
  public:
    explicit errstate(const std::map<std::string, std::string> &kwargs) : saved_(detail::g_state)
    {
        for (auto &[k, v] : kwargs)
        {
            if (k == "all")
            {
                auto h = from_string(v);
                detail::g_state.divide = h;
                detail::g_state.over = h;
                detail::g_state.under = h;
                detail::g_state.invalid = h;
            }
            else if (k == "divide")
            {
                detail::g_state.divide = from_string(v);
            }
            else if (k == "over")
            {
                detail::g_state.over = from_string(v);
            }
            else if (k == "under")
            {
                detail::g_state.under = from_string(v);
            }
            else if (k == "invalid")
            {
                detail::g_state.invalid = from_string(v);
            }
        }
    }

    explicit errstate(std::optional<std::string> all = std::nullopt, std::optional<std::string> divide = std::nullopt,
                      std::optional<std::string> over = std::nullopt, std::optional<std::string> under = std::nullopt,
                      std::optional<std::string> invalid = std::nullopt)
        : saved_(detail::g_state)
    {
        if (all.has_value())
        {
            auto h = from_string(*all);
            detail::g_state.divide = h;
            detail::g_state.over = h;
            detail::g_state.under = h;
            detail::g_state.invalid = h;
        }
        if (divide.has_value())
        {
            detail::g_state.divide = from_string(*divide);
        }
        if (over.has_value())
        {
            detail::g_state.over = from_string(*over);
        }
        if (under.has_value())
        {
            detail::g_state.under = from_string(*under);
        }
        if (invalid.has_value())
        {
            detail::g_state.invalid = from_string(*invalid);
        }
    }

    ~errstate()
    {
        detail::g_state = saved_;
    }

    errstate(const errstate &) = delete;
    auto operator=(const errstate &) -> errstate & = delete;

  private:
    ErrState saved_;
};

// Helper to dispatch handling (warn/raise/call)
namespace detail
{
inline void handle(const std::string &cls, const std::string &msg)
{
    ErrHandling h;
    if (cls == "divide")
    {
        h = g_state.divide;
    }
    else if (cls == "over")
    {
        h = g_state.over;
    }
    else if (cls == "under")
    {
        h = g_state.under;
    }
    else if (cls == "invalid")
    {
        h = g_state.invalid;
    }
    else
    {
        return;
    }
    switch (h)
    {
    case ErrHandling::ignore:
        break;
    case ErrHandling::warn:
        // In C++ we throw RankWarning analogue; numpy warns
        throw RankWarning(cls + ": " + msg);
    case ErrHandling::raise:
        throw FloatingPointError(cls + ": " + msg);
    case ErrHandling::call:
        if (g_call)
        {
            g_call(cls, msg);
        }
        break;
    case ErrHandling::print:
        std::fprintf(stderr, "np err [%s] %s\n", cls.c_str(), msg.c_str());
        break;
    case ErrHandling::log:
        // log is print for now
        std::fprintf(stderr, "np err [%s] %s\n", cls.c_str(), msg.c_str());
        break;
    }
}
} // namespace detail

} // namespace err

// Top-level aliases matching numpy.* (not only np.err.*)
using err::errstate;
using err::geterr;
using err::geterrcall;
using err::seterr;
using err::seterrcall;

} // namespace np

#endif // NP_ERR_HPP
