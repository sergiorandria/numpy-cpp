#API — dev(760 + routines)

Umbrella `include/np/np.hpp:13` (28 includes; all integrated). Every `np::` has Doxygen `Reference: numpy-reference/...` or Bott–Tu/Hatcher.

## Module table (36 groups)

| Group | Header | Key symbols (file:line) | NumPy ref |
|-------|--------|--------------------------|-----------|
| **Constants** | `constants.hpp` | `pi, e, euler_gamma, inf, nan, newaxis, NINF/PINF` | `constants.html` |
| **Creation** | `creation.hpp:65` / `creation_fixed.hpp` | `zeros/ones/full/empty, arange, linspace:logspace, geomspace, eye, meshgrid, asanyarray:946, fromiter:1023, rec.*:1497` | `routines.array-creation.html` |
| **Manipulation** | `manipulation.hpp:64` | `copyto:1980, ndim/shape/size, reshape, ravel, transpose, permute_dims, matrix_transpose, moveaxis, broadcast_to, as_strided:2395, pad:2395, split:703, block:1832, atleast_*:1606` | `routines.array-manipulation.html` |
| **Concatenate** | `concatenate.hpp` | `concatenate, stack, vstack/hstack/dstack, column_stack, concat` | same |
| **Bitwise** | `bitwise.hpp:165` | `bitwise_and/or/xor, invert, left/right_shift, bitwise_count, packbits` | `routines.bitwise.html` |
| **Math** | `math.hpp:2968` / `emath.hpp` | 112 ufuncs `sin/cos, exp/log, floor/ceil, nextafter, trapz:2765` | `routines.math.html` |
| **Strings** | `char.hpp` | `np::ch`/`strings`, 40+ `add, center, encode, is*` | `routines.char.html` |
| **Logic** | `logic.hpp:590` | `isfinite/isinf/isnan, logical_and/or, all/any, isclose:590, isin:590, intersect1d:669, unique_all:730` | `routines.logic.html` |
| **Functional** | `functional.hpp` | `apply_along_axis, vectorize, piecewise` | `routines.functional.html` |
| **Datetime** | `datetime.hpp:99` | `busdaycalendar:71, is_busday:192, busday_offset:341, busday_count:390, datetime_as_string:445, NaT:513` | `routines.datetime.html` |
| **Dtype** | `dtype.hpp:1522` | `can_cast, promote_types, finfo/iinfo, isdtype, rec.format_parser` | `routines.dtype.html` |
| **Masked** | `masked_array.hpp:916` | `MaskedArray, masked_where, anom, allequal` +36 | `routines.ma.html` |
| **Indexing** | `indexing.hpp:42` | `Slice, IndexExp, c_/r_/s_, ix_:338, nditer:576, flatiter:616, Arrayterator:762, take:820` | `routines.indexing.html` |
| **Sorting** | `sorting.hpp:426` | `sort, argsort, lexsort, partition, searchsorted, nonzero` | `routines.sort.html` |
| **Statistics** | `statistics.hpp:1955` | `mean/var/std, median, quantile:2291, histogramdd` | `routines.statistics.html` |
| **Linalg** | `linalg.hpp:2669` | `dot, matmul, tensordot, einsum:3837, cross:3322, qr, eig, cholesky, norm:1630, einsum_path:3845, matvec:3990` | `routines.linalg.html` |
| **Linalg fixed** | `linalg_fixed.hpp` | `ndarrayf` compile-time `det, svd` | same |
| **FFT** | `fft/fft_core.hpp:244` | `fft, rfft, fftn, fftshift` 18 ops | `routines.fft.html` |
| **Random** | `random.hpp:64` | `Generator:42` 50 dists, `SeedSequence:1311`, `PCG64` | `random/index.html` |
| **I/O** | `io.hpp:1131` | `load, savez, NpzFile, savetxt, DataSource` | `routines.io.html` |
| **Polynomial** | `polynomial.hpp:604` | `Polynomial, Chebyshev, polyfit, polyutils` | `routines.polynomials.html` |
| **SIMD** | `simd.hpp:983` | `add/mul WASM/RVV, Features::has_*` | `simd/index.html` |
| **Err** | `err.hpp` | `seterr, errstate` | `routines.err.html` |
| **Exceptions** | `exceptions.hpp` | `LinAlgError, AxisError` | `routines.exceptions.html` |
| **Window** | `window.hpp` | `bartlett, kaiser` | `routines.window.html` |
| **Testing** | `testing.hpp:108` | `assert_equal, Tester:442` | `routines.testing.html` |
| **Other** | `other.hpp:119` | `who, byte_bounds, einsum_path_stub` | `routines.other.html` |
| **Threadpool** | `threadpool.hpp:236` | `ThreadPool::global().parallel_for` | `threadpool` |
| **BigInt** | `bigint.hpp:304` | `bigint (cpp_int/GMP), make_bigint, _mpz` | `—` |
| **Homology** | `homology.hpp:539` | `SimplicialComplex, betti_numbers, homology_groups, smith_normal_form, exact_rank` | `Hatcher` |
| **Homotopy** | `homotopy.hpp:236` | `Whitehead, aspherical, homotopy_groups` | `Hatcher` |
| **Modular** | `modular.hpp:232` | `ModularForm, Hecke` | `Diamond–Shurman` |
| **Manifold** | `manifold.hpp:583` | `AbstractManifold, Sphere/Torus/Projective/Klein, de_rham, simplicial` | `Lee` |
| **Variety** | `variety.hpp` | `deprecated alias of manifold.hpp` | `—` |
| **Differential** | `differential.hpp:438` | `VM, ScalarField, KForm, exterior_derivative, wedge, pullback, kernel::gradient/hessian/laplacian` | `Bott–Tu` |
| **Lattice** | `lattice.hpp:143` | `Lattice, PosetLattice, meet/join, dual, lll/bkz, gram, volume, shortest/closest, LatticeFactory, Builder, Strategy, Visitor, Observer, Decorator` | `Micciancio–Goldwasser, Lenstra–Lenstra–Lovász` |
| **Padic** | `padic.hpp:135` | `Padic, PadicLattice, Hensel/Newton, valuation/norm/expansion/teichmuller, PadicFactory, Builder, Strategy, Visitor, Observer, Decorator, to_padic_lattice` | `Gouvea, Koblitz, Serre` |
| **Neuromorphic** | `neuromorphic.hpp:1` | `Event/EventArray, SpikeEncoder (rate/temporal), LIF/Izhikevich, STDP (standalone), INeuromorphicBackend (CPU/LIF-sim), NeuromorphicFactory, EventBuilder, SpikeVisitor, QuantizedEventArray` | `Gerstner (algorithmic only; no Loihi2/SpiNNaker hardware)` |
| **Memory** | `memory.hpp:1` | `TaggedArray<T,S>` (ordinary host storage), `HbmHintArray/CxlHintArray/DeviceHintArray` placement hints, `tag_hbm_hint/tag_device_hint`, `migrate_to_host`, `zeros_hinted(shape, space)` | `HBM3/CXL3.0/GH200 (hints only; no device migration)` |
| **Tensor** | `tensor_core.hpp:1` | `TensorBackend (CPU/GPU-FP32/CPU-blocked), TensorFactory, QuantizedTensor, quantize, matmul_fp8` | `cuBLAS FP32 / blocked CPU` |
| **Analog** | `memristor.hpp:1` | `Crossbar (VMM, program, outer-product), MemristorCell (ion-drift/Simmons/TEAM/VTEAM/Yakopcic/Stanford), WindowFunction, MappingScheme, IMemristorBackend (Sim/Noisy/HW/Serial), DifferentialCrossbar, TiledCrossbar, CrossbarBuilder, ReRAMFactory (mythic/dmatrix presets)` | `Strukov/Kvatinsky/Yakopcic, Mythic/d-Matrix` |
| **Photonics** | `photonics.hpp:1` | `MachZehnderMesh (unitary), PhotonicsFactory::identity, apply (optical matmul)` | `Lightmatter/Luminous` |
| **Quantum** | `quantum.hpp:1` | `StateVector (2^n), QuantumFactory::zero/plus_state, prob` | `IBM Heron/Quantinuum` |
| **Accelerator** | `accelerator.hpp:1` | `IAccelerator (CPU/GPU/ReRAM-sim/auto_select), AcceleratorFactory::cpu/gpu/reram/auto_select/powerful` | `Heterogeneous (CPU/GPU/sim; no Loihi hardware)` |
| **Cohomology** | `cohomology.hpp:191` | `cohomology_groups, cohomology_ring, cup_product, poincare_pairing, intersection_form, kunneth` | `Hatcher Ch.3` |
| **Bundle** | `bundle.hpp:103` | `VectorBundle, tangent/cotangent, chern/stiefel/euler/pontryagin, whitney_sum, HodgeStar` | `Milnor–Stasheff` |
| **Persistent** | `persistent.hpp:94` | `FilteredSimplex, Filtration, persistence_barcode, bottleneck_distance, vietoris_rips` | `Edelsbrunner–Harer` |
| **Spectral** | `spectral.hpp:129` | `MayerVietoris, SpectralSequence, leray_serre (Hopf), ahss, total_betti` | `McCleary` |

Count `712` base + ~50 higher-math (homology/bundle/persistent/spectral) + aliases. `other.hpp` parity shims (`who/disp/info/source/lookfor/deprecate/show_config`, `einsum_path_stub`) and default PQC KEM/signature wrappers are intentional documented stubs.

## Quick reference

```cpp
// ndarray
auto a = np::zeros<double>({3,4}); // creation.hpp
a.shape;
a.strides;
a.offset;
a.data(); // ndarray.hpp
a.sum();
a.mean(axis);
a.sort();                           // ndarray + sorting
np::copyto(dst, src);               // manipulation.hpp:1980
np::is_busday(dates);               // datetime.hpp:192
np::isin(a, b);                     // logic.hpp:590
np::linalg::dot(a, b);              // linalg.hpp:2669
np::fft::fft(in);                   // fft/fft_core.hpp:244
np::testing::assert_allclose(a, b); // testing.hpp
```

See `../README.md` quick start and `numpy-reference/` for Python semantics.
