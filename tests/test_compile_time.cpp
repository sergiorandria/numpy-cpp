/**
 * @file test_compile_time.cpp
 * @brief Negative compile-time guarantees of the fixed-shape path.
 *
 *        Every invalid use below (shape mismatch, unbroadcastable operands,
 *        element-count-preserving reshape violations, mismatched joins,
 *        out-of-range axes, wrong indexing arity) is expressed as a
 *        requires-expression over the public API. The static_asserts prove
 *        the library REJECTS these programs at compile time instead of
 *        silently truncating / misaligning data (NumPy parity: those are
 *        runtime errors there, but the fixed path turns them into
 *        compile-time errors).
 */
#include "np/np.hpp"

namespace
{

using np::ndarray;
using np::ndarrayf;

template <typename L, typename R>
concept addable = requires(const L &l, const R &r) { l + r; };

template <typename A, typename B>
concept concatable = requires(const A &a, const B &b) { np::concatenate(a, b); };

template <typename A, typename B>
concept stackable = requires(const A &a, const B &b) { np::stack<0>(a, b); };

template <typename A, int... E>
concept reshapable = requires(const A &a) { a.template reshape<E...>(); };

template <typename A, int Axis>
concept squeezable = requires(const A &a) { a.template squeeze<Axis>(); };

template <typename A, int Axis>
concept expandable = requires(const A &a) { a.template expand_dims<Axis>(); };

template <typename A, int Axis>
concept summable = requires(const A &a) { a.template sum<Axis>(); };

template <typename A>
concept three_index = requires(const A &a) { a(0, 1, 2); };

template <typename A, typename B>
concept dottable = requires(const A &a, const B &b) { np::linalg::dot(a, b); };

template <typename A, typename B>
concept matmulable = requires(const A &a, const B &b) { np::linalg::matmul(a, b); };

// --- the detector itself is sound: valid uses are accepted -----------------
static_assert(addable<ndarrayf<int, 2, 3>, ndarrayf<int, 2, 3>>);
static_assert(addable<ndarrayf<int, 2, 3>, ndarrayf<int, 3>>);
static_assert(addable<ndarrayf<int, 2, 3>, ndarrayf<int, 2, 1>>);
static_assert(addable<ndarrayf<int, 2, 3>, ndarrayf<int, 1, 3>>);
static_assert(addable<ndarrayf<int, 2, 3>, int>);
static_assert(addable<int, ndarrayf<int, 2, 3>>);
static_assert(reshapable<ndarrayf<int, 2, 3>, 3, 2>);
static_assert(reshapable<ndarrayf<int, 2, 3>, 1, 6>);
static_assert(reshapable<ndarrayf<int, 2, 3>, 2, 3>);
static_assert(concatable<ndarrayf<int, 3>, ndarrayf<int, 2>>);
static_assert(concatable<ndarrayf<int, 2, 3>, ndarrayf<int, 4, 3>>);
static_assert(stackable<ndarrayf<int, 2, 2>, ndarrayf<int, 2, 2>>);
static_assert(squeezable<ndarrayf<int, 1, 3, 1>, 0>);
static_assert(expandable<ndarrayf<int, 2, 3>, 0>);
static_assert(expandable<ndarrayf<int, 2, 3>, 2>);
static_assert(summable<ndarrayf<int, 2, 3>, 1>);
static_assert(three_index<ndarrayf<int, 2, 3, 4>>);
static_assert(dottable<ndarrayf<int, 3>, ndarrayf<int, 3>>);
static_assert(dottable<ndarrayf<int, 2, 3>, ndarrayf<int, 3>>);
static_assert(dottable<ndarrayf<int, 3>, ndarrayf<int, 3, 2>>);
static_assert(dottable<ndarrayf<int, 2, 3>, ndarrayf<int, 3, 2>>);
static_assert(matmulable<ndarrayf<int, 2, 3>, ndarrayf<int, 3, 2>>);

// --- shape mismatches are rejected at compile time -------------------------
static_assert(!addable<ndarrayf<int, 2, 3>, ndarrayf<int, 2, 2>>);
static_assert(!addable<ndarrayf<int, 2, 3>, ndarrayf<int, 4>>);
static_assert(!addable<ndarrayf<int, 2, 3>, ndarrayf<int, 3, 2>>);
static_assert(!addable<ndarrayf<int, 2, 3>, ndarrayf<int, 5, 4, 3>>);

// --- reshape must preserve the element count -------------------------------
static_assert(!reshapable<ndarrayf<int, 2, 3>, 2, 2>);
static_assert(!reshapable<ndarrayf<int, 2, 3>, 7>);
static_assert(!reshapable<ndarrayf<int, 2, 3>, 0, 5>);
static_assert(!reshapable<ndarrayf<int, 2, 3>, 2, 2, 1>);

// --- joins must be shape-consistent ----------------------------------------
static_assert(!concatable<ndarrayf<int, 2, 3>, ndarrayf<int, 4>>);
static_assert(!concatable<ndarrayf<int, 3>, ndarrayf<int, 2, 2>>);
static_assert(!stackable<ndarrayf<int, 2, 3>, ndarrayf<int, 3>>);
static_assert(!stackable<ndarrayf<int, 2, 2>, ndarrayf<int, 3, 2>>);

// --- axes must be in range --------------------------------------------------
static_assert(!squeezable<ndarrayf<int, 2, 3>, 5>);
static_assert(!expandable<ndarrayf<int, 2, 3>, 7>);
static_assert(!summable<ndarrayf<int, 2, 3>, 5>);
static_assert(!summable<ndarrayf<int, 2, 3>, -2>);

// --- indexing arity must match the rank ------------------------------------
static_assert(!three_index<ndarrayf<int, 2, 3>>);
static_assert(!three_index<ndarrayf<int, 4>>);

// --- linalg contraction dimension is checked at compile time ---------------
static_assert(!dottable<ndarrayf<int, 2, 3>, ndarrayf<int, 2, 3>>);
static_assert(!dottable<ndarrayf<int, 3, 2>, ndarrayf<int, 3>>);
static_assert(!matmulable<ndarrayf<int, 2, 3>, ndarrayf<int, 2, 4>>);
static_assert(!matmulable<ndarrayf<int, 2, 2>, ndarrayf<int, 3, 2>>);

// --- decompositions reject complex element types at compile time ------------
using CplxArr = np::ndarray<std::complex<double>>;
using RealArr = np::ndarray<double>;

template <typename A>
concept svdable = requires(const A &a) { np::linalg::svd(a); };

template <typename A>
concept qrable = requires(const A &a) { np::linalg::qr(a); };

template <typename A>
concept eigable = requires(const A &a) { np::linalg::eig(a); };

template <typename A>
concept detable = requires(const A &a) { np::linalg::det(a); };

template <typename A>
concept inversible = requires(const A &a) { np::linalg::inv(a); };

template <typename A>
concept solvable = requires(const A &a, const A &b) { np::linalg::solve(a, b); };

template <typename A>
concept powerable = requires(const A &a) { np::linalg::matrix_power(a, 2); };

template <typename A>
concept choleskyable = requires(const A &a) { np::linalg::cholesky(a); };

template <typename A>
concept normable = requires(const A &a) { np::linalg::norm(a); };

template <typename A>
concept rankable = requires(const A &a) { np::linalg::matrix_rank(a); };

template <typename A>
concept pinvable = requires(const A &a) { np::linalg::pinv(a); };

template <typename A>
concept condable = requires(const A &a) { np::linalg::cond(a); };

template <typename A>
concept eighable = requires(const A &a) { np::linalg::eigh(a); };

template <typename A>
concept lstsqable = requires(const A &a, const A &b) { np::linalg::lstsq(a, b); };

template <typename A>
concept tensordotable = requires(const A &a, const A &b) { np::linalg::tensordot(a, b); };

template <typename A>
concept crossable = requires(const A &a, const A &b) { np::linalg::cross(a, b); };

template <typename A>
concept diagonable = requires(const A &a) { np::linalg::diagonal(a); };

template <typename A>
concept transposeable = requires(const A &a) { np::linalg::matrix_transpose(a); };

template <typename A>
concept matrix_normable = requires(const A &a) { np::linalg::matrix_norm(a); };

template <typename A>
concept tensorinvable = requires(const A &a) { np::linalg::tensorinv(a); };

template <typename A>
concept tensorsolvable = requires(const A &a, const A &b) { np::linalg::tensorsolve(a, b); };

template <typename A>
concept vecdotable = requires(const A &a, const A &b) { np::linalg::vecdot(a, b); };

template <typename A>
concept vector_normable = requires(const A &a) { np::linalg::vector_norm(a); };

static_assert(svdable<RealArr>);
static_assert(qrable<RealArr>);
static_assert(eigable<RealArr>);
static_assert(svdable<CplxArr>);
static_assert(qrable<CplxArr>);
static_assert(eigable<CplxArr>);

static_assert(detable<RealArr>);
static_assert(inversible<RealArr>);
static_assert(solvable<RealArr>);
static_assert(powerable<RealArr>);
static_assert(choleskyable<RealArr>);
static_assert(normable<RealArr>);
static_assert(rankable<RealArr>);
static_assert(pinvable<RealArr>);
static_assert(condable<RealArr>);
static_assert(eighable<RealArr>);
static_assert(detable<CplxArr>);
static_assert(inversible<CplxArr>);
static_assert(solvable<CplxArr>);
static_assert(powerable<CplxArr>);
static_assert(choleskyable<CplxArr>);
static_assert(normable<CplxArr>);
static_assert(rankable<CplxArr>);
static_assert(pinvable<CplxArr>);
static_assert(condable<CplxArr>);
static_assert(eighable<CplxArr>);

static_assert(lstsqable<RealArr>);
static_assert(tensordotable<RealArr>);
static_assert(crossable<RealArr>);
static_assert(lstsqable<CplxArr>);
static_assert(tensordotable<CplxArr>);
static_assert(crossable<CplxArr>);

// --- fixed path: shapes are part of the type and complex is rejected --------
using FRealArr = np::ndarrayf<double, 2, 2>;
using FRealVec2 = np::ndarrayf<double, 2>;
using FVec3 = np::ndarrayf<double, 3>;
using FVec4 = np::ndarrayf<double, 4>;
using FNonSquare = np::ndarrayf<double, 2, 3>;
using FCplxArr = np::ndarrayf<std::complex<double>, 2, 2>;

template <typename A, typename B>
concept fixed_solvable = requires(const A &a, const B &b) { np::linalg::solve(a, b); };

template <typename A, typename B>
concept fixed_lstsqable = requires(const A &a, const B &b) { np::linalg::lstsq(a, b); };

static_assert(svdable<FRealArr>);
static_assert(svdable<FNonSquare>);
static_assert(qrable<FRealArr>);
static_assert(qrable<FNonSquare>);
static_assert(detable<FRealArr>);
static_assert(inversible<FRealArr>);
static_assert(powerable<FRealArr>);
static_assert(choleskyable<FRealArr>);
static_assert(normable<FRealArr>);
static_assert(rankable<FRealArr>);
static_assert(pinvable<FRealArr>);
static_assert(condable<FRealArr>);
static_assert(eighable<FRealArr>);
static_assert(fixed_solvable<FRealArr, FRealVec2>);
static_assert(crossable<FVec3>);
static_assert(!svdable<FCplxArr>);
static_assert(!qrable<FCplxArr>);
static_assert(!detable<FCplxArr>);
static_assert(!inversible<FCplxArr>);
static_assert(!powerable<FCplxArr>);
static_assert(!choleskyable<FCplxArr>);
static_assert(!normable<FCplxArr>);
static_assert(!rankable<FCplxArr>);
static_assert(!pinvable<FCplxArr>);
static_assert(!condable<FCplxArr>);
static_assert(!eighable<FCplxArr>);
static_assert(!detable<FNonSquare>);
static_assert(!inversible<FNonSquare>);
static_assert(!choleskyable<FNonSquare>);
static_assert(!eighable<FNonSquare>);
static_assert(!condable<FNonSquare>);
static_assert(!fixed_solvable<FRealArr, FVec3>);
static_assert(!fixed_solvable<FNonSquare, FRealVec2>);
static_assert(!fixed_lstsqable<FRealArr, FVec3>);
static_assert(!fixed_lstsqable<FNonSquare, FVec3>);
static_assert(!fixed_lstsqable<FRealArr, FNonSquare>);
static_assert(!crossable<FVec4>);
static_assert(!crossable<FRealArr>);

static_assert(diagonable<RealArr>);
static_assert(transposeable<RealArr>);
static_assert(matrix_normable<RealArr>);
static_assert(tensorinvable<RealArr>);
static_assert(tensorsolvable<RealArr>);
static_assert(vecdotable<RealArr>);
static_assert(vector_normable<RealArr>);
static_assert(diagonable<CplxArr>);
static_assert(transposeable<CplxArr>);
static_assert(matrix_normable<CplxArr>);
static_assert(tensorinvable<CplxArr>);
static_assert(tensorsolvable<CplxArr>);
static_assert(vecdotable<CplxArr>);
static_assert(vector_normable<CplxArr>);

} // namespace

int main()
{
    return 0;
}
