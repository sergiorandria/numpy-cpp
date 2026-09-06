/**
 * @file exceptions.hpp
 * @brief NumPy-compatible exception hierarchy.
 *
 * All exceptions carry the source location (file, line, function) of the
 * throw site, formatted like a Python traceback line.
 *
 * @author Sergio Randriamihoatra (sergiorandriamihoatra@gmail.com)
 */
#ifndef NP_EXCEPTIONS_HPP
#define NP_EXCEPTIONS_HPP

#include <exception>
#include <format>
#include <source_location>
#include <string>

#include "api_macros.hpp"

namespace np::exceptions
{

namespace detail
{
// Format a message with its throw site as "file:line: function: msg".
NP_NODISCARD inline std::string format_message(const std::string &msg, const std::source_location &loc)
{
    return std::format("{}:{}: {}: {}", loc.file_name(), loc.line(), loc.function_name(), msg);
}
} // namespace detail

/** @brief Base class for all np exceptions.
 *
 * what() returns a string of the form "file:line: function: msg",
 * mirroring the information density of a Python traceback line.
 *
 * @param msg  Human-readable error message.
 * @param loc  Source location of the throw site (defaults to current).
 */
class NumpyError : public std::exception
{
  public:
    explicit NumpyError(const std::string &msg, const std::source_location &loc = std::source_location::current())
        : what_msg_(detail::format_message(msg, loc))
    {
    }

    NP_NODISCARD auto what() const noexcept -> const char * override
    {
        return what_msg_.c_str();
    }

  protected:
    std::string what_msg_;
};

/** @brief Raised when an axis parameter is out of bounds for the array.
 *
 * Reference:
 * numpy-reference/reference/generated/numpy.exceptions.AxisError.html
 *
 * @param msg  Description of the axis error.
 * @param loc  Source location of the throw site.
 */
class AxisError : public NumpyError
{
  public:
    explicit AxisError(const std::string &msg, const std::source_location &loc = std::source_location::current())
        : NumpyError(msg, loc)
    {
    }
};

/** @brief Non-fatal warning emitted for invalid axis/rank usage.
 *
 * Reference: numpy-reference/reference/generated/numpy.RankWarning.html
 *
 * @param msg  Warning message.
 * @param loc  Source location of the throw site.
 */
class RankWarning : public NumpyError
{
  public:
    explicit RankWarning(const std::string &msg, const std::source_location &loc = std::source_location::current())
        : NumpyError(msg, loc)
    {
    }
};

/** @brief Raised when matrix dimensions are incompatible for an operation.
 *
 * @param msg  Description of the dimension mismatch.
 * @param loc  Source location of the throw site.
 */
class MatrixDimError : public NumpyError
{
  public:
    explicit MatrixDimError(const std::string &msg, const std::source_location &loc = std::source_location::current())
        : NumpyError(msg, loc)
    {
    }
};

/** @brief Raised when two dtypes cannot be promoted to a common type.
 *
 * @param msg  Description of the promotion failure.
 * @param loc  Source location of the throw site.
 */
class DtypePromotionError : public NumpyError
{
  public:
    explicit DtypePromotionError(const std::string &msg,
                                 const std::source_location &loc = std::source_location::current())
        : NumpyError(msg, loc)
    {
    }
};

/** @brief Emitted when an API enters a deprecated code path.
 *
 * @param msg  Deprecation notice.
 * @param loc  Source location of the throw site.
 */
class VisibleDeprecation : public NumpyError
{
  public:
    explicit VisibleDeprecation(const std::string &msg,
                                const std::source_location &loc = std::source_location::current())
        : NumpyError(msg, loc)
    {
    }
};

/** @brief Emitted when a complex value is implicitly cast to a real type.
 *
 * @param msg  Warning message describing the cast.
 * @param loc  Source location of the throw site.
 */
class ComplexWarning : public NumpyError
{
  public:
    explicit ComplexWarning(const std::string &msg, const std::source_location &loc = std::source_location::current())
        : NumpyError(msg, loc)
    {
    }
};

/** @brief Raised by np::linalg when a decomposition fails to converge
 *        or encounters a mathematically singular problem.
 *
 * Reference: numpy-reference/reference/generated/numpy.linalg.LinAlgError.html
 *
 * @param msg  Description of the linear algebra failure.
 * @param loc  Source location of the throw site.
 */
class LinAlgError : public NumpyError
{
  public:
    explicit LinAlgError(const std::string &msg, const std::source_location &loc = std::source_location::current())
        : NumpyError(msg, loc)
    {
    }
};

/** @brief Raised when a floating-point operation has an invalid value.
 *
 * Mirrors Python's FloatingPointError used by np.seterr(..., "raise").
 *
 * Reference: numpy-reference/reference/generated/numpy.seterr.html
 */
class FloatingPointError : public NumpyError
{
  public:
    explicit FloatingPointError(const std::string &msg,
                                const std::source_location &loc = std::source_location::current())
        : NumpyError(msg, loc)
    {
    }
};

/** @brief Raised when max_work was exceeded (e.g. isclose with TooHard).
 *
 * Reference: numpy-reference/reference/generated/numpy.exceptions.TooHardError.html
 */
class TooHardError : public NumpyError
{
  public:
    explicit TooHardError(const std::string &msg, const std::source_location &loc = std::source_location::current())
        : NumpyError(msg, loc)
    {
    }
};

/** @brief Visible deprecation warning (np.exceptions.VisibleDeprecationWarning).
 *
 * Reference:
 * numpy-reference/reference/generated/numpy.exceptions.VisibleDeprecationWarning.html
 */
class VisibleDeprecationWarning : public NumpyError
{
  public:
    explicit VisibleDeprecationWarning(const std::string &msg,
                                       const std::source_location &loc = std::source_location::current())
        : NumpyError(msg, loc)
    {
    }
};

} // namespace np::exceptions

namespace np
{
using AxisError = exceptions::AxisError;
using RankWarning = exceptions::RankWarning;
using MatrixDimError = exceptions::MatrixDimError;
using DtypePromotionError = exceptions::DtypePromotionError;
using VisibleDeprecation = exceptions::VisibleDeprecation;
using VisibleDeprecationWarning = exceptions::VisibleDeprecationWarning;
using ComplexWarning = exceptions::ComplexWarning;
using LinAlgError = exceptions::LinAlgError;
using FloatingPointError = exceptions::FloatingPointError;
using TooHardError = exceptions::TooHardError;
} // namespace np

namespace np::exceptions
{
// Backward compat: numpy 1.x exposed VisibleDeprecation as VisibleDeprecationWarning
using VisibleDeprecationWarningAlias = VisibleDeprecationWarning;
} // namespace np::exceptions

#endif // NP_EXCEPTIONS_HPP
