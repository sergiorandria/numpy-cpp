/**
 * @file test_dtype.cpp
 * @brief Tests for the dtype system (np/dtype.hpp).
 */
#include <cstdint>

#include "np/np.hpp"
#include "test_util.hpp"

using namespace np;

int main()
{
    // Type aliases: np::complex128 etc. are now直接 C++ types usable as ndarray<np::complex128>
    static_assert(std::is_same_v<np::int32::type, std::int32_t>);
    static_assert(std::is_same_v<np::float64::type, double>);
    static_assert(std::is_same_v<np::complex128::type, std::complex<double>>);
    static_assert(std::is_same_v<np::bool_::type, bool>);
    test::check(dtype_of<np::int32::type> == dtype::int32, "int32 dtype_of");
    test::check(dtype_of<np::float64::type> == dtype::float64, "float64 dtype_of");
    test::check(dtype_of<np::bool_::type> == dtype::bool_, "bool_ dtype_of");

    // dtype_t mapping (type-based: dtype_t<np::tag> -> C++ type)
    static_assert(std::is_same_v<dtype_t<np::int8>, std::int8_t>);
    static_assert(std::is_same_v<dtype_t<np::uint8>, std::uint8_t>);
    static_assert(std::is_same_v<dtype_t<np::int32>, std::int32_t>);
    static_assert(std::is_same_v<dtype_t<np::float32>, float>);
    static_assert(std::is_same_v<dtype_t<np::float64>, double>);
    static_assert(std::is_same_v<dtype_t<np::complex128>, std::complex<double>>);
    static_assert(std::is_same_v<dtype_t<np::bool_>, bool>);
    // enum-based mapping kept as dtype_t_enum
    static_assert(std::is_same_v<dtype_t_enum<dtype::int8>, std::int8_t>);
    static_assert(std::is_same_v<dtype_t_enum<dtype::complex128>, std::complex<double>>);

    // dtype_of mapping
    static_assert(dtype_of<int> == dtype::int32);
    // static_assert(dtype_of<long long> == dtype::int64);
    static_assert(dtype_of<unsigned char> == dtype::uint8);
    static_assert(dtype_of<double> == dtype::float64);
    static_assert(dtype_of<float> == dtype::float32);
    static_assert(dtype_of<bool> == dtype::bool_);
    static_assert(dtype_of<std::complex<double>> == dtype::complex128);
    static_assert(dtype_of<std::complex<float>> == dtype::complex64);

    // Traits
    static_assert(is_complex_v<std::complex<double>>);
    static_assert(!is_complex_v<double>);
    static_assert(dtype_is_floating(dtype::float64));
    static_assert(!dtype_is_floating(dtype::int32));
    static_assert(dtype_is_integer(dtype::int64));
    static_assert(dtype_is_signed(dtype::int16));
    static_assert(dtype_is_unsigned(dtype::uint32));
    static_assert(dtype_is_bool(dtype::bool_));

    // Names and sizes
    test::check(std::string(dtype_name(dtype::float64)) == "float64", "dtype_name float64");
    test::check(dtype_size(dtype::int32) == 4, "dtype_size int32");
    test::check(dtype_size(dtype::complex128) == 16, "dtype_size c128");

    // _Np_dtype storage-classifier alias set: each alias binds a compile-time
    // dtype to its native storage and remains usable as a plain scalar.
    static_assert(std::is_same_v<_Np_dtype::_Np_int8::value_type, std::int8_t>);
    static_assert(std::is_same_v<_Np_dtype::_Np_uint64::value_type, std::uint64_t>);
    static_assert(std::is_same_v<_Np_dtype::_Np_float32::value_type, float>);
    static_assert(std::is_same_v<_Np_dtype::_Np_float64::value_type, double>);
    static_assert(std::is_same_v<_Np_dtype::_Np_complex128::value_type, std::complex<double>>);
    static_assert(std::is_same_v<_Np_dtype::_Np_bool_::value_type, bool>);
    static_assert(std::is_same_v<_Np_dtype::_Np_datetime64::value_type, std::int64_t>);
    static_assert(_Np_dtype::_Np_int64::type == dtype::int64);
    static_assert(_Np_dtype::_Np_float16::type == dtype::float16);
    static_assert(_Np_dtype::_Np_int8::get_type() == dtype::int8);
    static_assert(_Np_dtype::_Np_complex64::get_type() == dtype::complex64);

    // Classifier behaves like its scalar value.
    _Np_dtype::_Np_int64 a{static_cast<std::int64_t>(7)};
    static_assert(_Np_dtype::_Np_int64{static_cast<std::int64_t>(3)}.value() == static_cast<std::int64_t>(3));
    test::check(static_cast<std::int64_t>(a) == 7, "classifier convert");
    a = static_cast<std::int64_t>(9);
    test::check(a.value() == 9, "classifier assign");
    test::check(_Np_dtype::_Np_float64{1.5}.get_type() == dtype::float64, "classifier get_type");

    // Compile-time comparison between classifiers.
    static_assert(_Np_dtype::_Np_int32{} == _Np_dtype::_Np_int32{});
    static_assert(_Np_dtype::_Np_int32{} != _Np_dtype::_Np_float32{});

    // String fallback storage for the non-integral string/unicode dtypes.
    _Np_dtype::_Np_string s{"hello"};
    test::check(std::string(s.value()) == "hello", "string fallback value");
    static_assert(_Np_dtype::_Np_string::type == dtype::string_);
    static_assert(_Np_dtype::_Np_unicode::type == dtype::unicode_);
    static_assert(std::is_same_v<_Np_dtype::_Np_string::value_type, std::string>);
    static_assert(std::is_same_v<_Np_dtype::_Np_unicode::value_type, std::u32string>);
    _Np_dtype::_Np_unicode u(U"café");
    test::check(u.value() == U"café", "unicode fallback value");

    // Compile-time integral/numeric trait.
    static_assert(is_integral_dtype_v<dtype::int8>);
    static_assert(is_integral_dtype_v<dtype::uint64>);
    static_assert(!is_integral_dtype_v<dtype::float32>);
    static_assert(!is_integral_dtype_v<dtype::string_>);
    static_assert(!is_integral_dtype_v<dtype::bool_>);
    static_assert(is_numeric_dtype_v<dtype::int32>);
    static_assert(is_numeric_dtype_v<dtype::float64>);
    static_assert(is_numeric_dtype_v<dtype::complex128>);
    static_assert(is_numeric_dtype_v<dtype::datetime64>);
    static_assert(!is_numeric_dtype_v<dtype::string_>);
    static_assert(!is_numeric_dtype_v<dtype::object_>);
    static_assert(is_integral_dtype<dtype::int64>::value);
    static_assert(std::is_base_of_v<std::true_type, is_integral_dtype<dtype::int64>>);

    return test::failures() ? 1 : 0;
}
