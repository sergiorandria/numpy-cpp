/**
 * @file np.hpp
 * @brief Umbrella header for the entire NumPy-like C++ library.
 *
 * Include this single header to get the whole API:
 *   #include <np/np.hpp>
 *
 * @author Sergio Randriamihoatra (sergiorandriamihoatra@gmail.com)
 */
#ifndef NP_NP_HPP
#define NP_NP_HPP

#include "api_macros.hpp"
#include "bigint.hpp"
#include "bitwise.hpp"
#include "bundle.hpp"
#include "char.hpp"
#include "cohomology.hpp"
#include "concatenate.hpp"
#include "constants.hpp"
#include "creation.hpp"
#include "creation_fixed.hpp"
#include "datetime.hpp"
#include "differential.hpp"
#include "dtype.hpp"
#include "emath.hpp"
#include "err.hpp"
#include "exceptions.hpp"
#include "fft.hpp"
#include "functional.hpp"
#include "gpu.hpp"
#include "half.hpp"
#include "homology.hpp"
#include "homotopy.hpp"
#include "indexing.hpp"
#include "io.hpp"
#include "lattice.hpp"
#include "linalg.hpp"
#include "linalg_fixed.hpp"
#include "accelerator.hpp"
#include "logic.hpp"
#include "manifold.hpp"
#include "manipulation.hpp"
#include "masked_array.hpp"
#include "math.hpp"
#include "matrix.hpp"
#include "memory.hpp"
#include "memristor.hpp"
#include "modular.hpp"
#include "ndarray.hpp"
#include "ndarray_fixed.hpp"
#include "neuromorphic.hpp"
#include "other.hpp"
#include "padic.hpp"
#include "persistent.hpp"
#include "photonics.hpp"
#include "physics.hpp"
#include "polynomial.hpp"
#include "powerful.hpp"
#include "pqc.hpp"
#include "quantum.hpp"
#include "random.hpp"
#include "simd.hpp"
#include "sorting.hpp"
#include "spectral.hpp"
#include "statistics.hpp"
#include "tensor_core.hpp"
#include "testing.hpp"
#include "threadpool.hpp"
#include "variety.hpp"
#include "window.hpp"

// NOTE: No blanket -Wbraced-scalar-init suppression here. That warning is
// scoped with push/pop directly around NDProxy in ndarray.hpp; an unscoped
// pragma at the bottom of this umbrella header would miss the already-
// included library code and instead silence the warning for all user code
// following the include.

#endif // NP_NP_HPP
