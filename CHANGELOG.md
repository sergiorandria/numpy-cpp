# Changelog — numpy-cpp

All notable changes to `numpy-cpp` will be documented here. Format based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), versioning follows [SemVer](https://semver.org/spec/v2.0.0.html).

## [Unreleased] — honesty pass + 50/50 suites

### Added
- **Leech lattice** — `LatticeFactory::leech()` (was a documented stub): Λ24 via extended binary Golay-code lifting (12 cyclic-shift code lifts + 11 pair vectors + verified odd row, scaled by 1/√8), with `detail::golay24_*` helpers. Provenance verified in `test_lattice.cpp`: Golay weight enumerator {0:1, 8:759, 12:2576, 16:759, 24:1}, exact integer-preimage determinant 2^36 (⇒ unimodular), even Gram matrix, GF(2) row space {0, all-ones}, exhaustive no-roots search (170016 candidates), all 1104 type-(±4,±4) and sampled octad type-(±2⁸) minimal vectors in span — hence Λ24 by Niemeier uniqueness.
- **Vector math library** — new `vecmath.hpp`: dependency-free SLEEF/SVML-class SIMD kernels (`exp`, `expm1`, `exp2`, `log`, `log10`, `log2`, `log1p`, `sin`, `cos`, `sincos`, `tan`, `asin`, `acos`, `atan`, `atan2`, `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`, `sqrt`, `cbrt`, `pow`, `hypot`, `floor`, `ceil`, `trunc`, `rint`, `fabs` for `float`/`double`). Single-source portable abstraction over AVX-512/AVX2/SSE2/NEON/scalar tiers, Cody-Waite range reduction, ≤2 ULP exp/log, ≤4 ULP trig/hyperbolic, exact small-`pow` exponents, IEEE edge cases bit-compatible with libm. `simd.hpp` dispatches to it (SLEEF still preferred for `sin/cos/exp/log` when `NP_ENABLE_SLEEF`), `math.hpp` gained contiguous fast paths for every transcendental ufunc (`tan`, `arcsin`/`arccos`/`arctan`/`arctan2`, `sinh`/`cosh`/`tanh`, `arcsinh`/`arccosh`/`arctanh`, `expm1`/`exp2`, `log10`/`log2`/`log1p`, `sqrt`/`cbrt`, `power`/`hypot`, `floor`/`ceil`/`trunc`/`rint`, `absolute`). New `tests/test_vecmath` (ULP sweeps + specials + integration); `np::exp`/`sin`/`log` measure 6.4x/7.1x/2.4x vs the scalar ufunc path.

### Fixed
- **E8 lattice basis** — `LatticeFactory::e8()` spanned an index-2 sublattice (det −2), not E8: with A7 simple roots fixed, any 8th E8 vector gives det = |Σv| ≥ 2. Replaced with Bourbaki simple roots (det −1, all 240 roots verified in span by `test_lattice.cpp`).
- **SIMD detection** — `simd.hpp` used an `elif` chain that defined only the top ISA macro, silently disabling all SSE2 kernels on standard `-msse4.2`/AVX builds; detection is now cumulative (also fixes `Features::has_sse2` reporting).

### Fixed
- **Docs honesty** — `README.md` no longer claims `0 stubs` or bare `Loihi2/SpiNNaker`, `HBM/CXL`, `Hopper/AMX` backends: `neuromorphic` is CPU LIF simulation, `memory` is host storage with `HbmHintArray`/`CxlHintArray` placement hints (`memory.hpp:104-176`), `tensor` is blocked-CPU/FP32-GPU dispatch with quantize-around-FP32 (`tensor_core.hpp:22`), `accelerator` is CPU/GPU/ReRAM-sim (`accelerator.hpp:226-242`). Documented `other.hpp` parity shims and default PQC wrappers as intentional stubs; test count corrected to `50/50` across `README.md`, `docs/` and CI step name.
- **CI enforcement** — `lint` job is now blocking (`clang-tidy` warnings fail the build), new `clang-format --check` job enforces `.clang-format` (`ColumnLimit: 120`, 4-space), TSan keeps `NP_WERROR=ON` (warnings-as-errors stay on where races hide).
- **Error handling (AGENTS.md §4)** — narrowed `datetime64_from_string` catch to `const std::exception&` with rethrow as `invalid_argument`; documented the `datetime_data` count fallback and the `cuda.hpp` `noexcept` cache fallback so no `catch (...)` silently swallows.

## [1.0.0] — 2025-09-03 — Production Ready

First stable **1.0** — header-only C++20 NumPy 2.2 — 712 routines, zero Python runtime.

### Added
- **SIMD** — `simd.hpp` `AVX512`/`AVX2`/`SSE2`/`NEON`/`RVV`/`WASM`/`SVE`/`VSX` + `SLEEF 3.7.0` for `sin/cos` via `FetchContent` (`NP_ENABLE_SLEEF`), dispatched in `math.hpp`, `linalg` `fma`, `tensor_core` `quantize`, `ndarray` elementwise/`sum`, `polynomial` Horner batch (`-DNP_ENABLE_AVX2`).
- **Tensor** — `tensor_core.hpp` `Strassen 7`, `Winograd` 15-add, `AlphaEvolve 4×4` rank-48 (vs 49/64) tiled, `Laderman 3×3` 23, `CW` `2.375`, `Hybrid-Auto` (`pow2`/`4×4`/`GPU`/`AMX`), `FP8/FP4` quant, `einsum` (`-DNP_ENABLE_POWERFUL`).
- **GPU** — `gpu.hpp` `dlopen` `libcuda`/`libcudart` (no hard link), `is_available()`/`device_count()`, `try_matmul`/`cpu_matmul`/`sharded`/`batch` + `managed_alloc` (`cudaMallocManaged` dlopen), `pinned_alloc` `madvise(HUGEPAGE)`, `fft` `cuFFT` probe, streams.
- **PQC** — `pqc.hpp` `secure_allocator` (`mlock`/`MADV_DONTDUMP`/`secure_zero`), `secure_buffer` isolation, `ct_barrier`/`ct_select` constant-time.
- **Security** — `ndarray` `secure_zero`/`secure_clear`/`secure_fill`/`secure_at`, `matrix` `secure_matmul`, `polynomial` `secure_polyval/roots`, `creation` `NP_USE_SECURE_IMPL` (`secure_buffer` + `secure_zero` for `zeros`), `linalg`/`fft`/`io` `secure` namespaces (`ct_barrier`).
- **Half** — `half.hpp` `half` (`_Float16`/`float16_t`/`float` fallback) vs `dtype::float16` tag, `is_half`, `quantize_half`/`dequantize_half`.
- **Differential** — `differential.hpp` `LLVMJit` ORC `LLJIT` content-hash cache, `getPtrTy`, `getOrInsertDeclaration`, `ExecutorAddr`, `VM::try_create` `expected` (C++23), `mdspan` view for `ndarray_fixed` (C++23).
- **Python** — `python/numpy_cpp.cpp` zero-copy `buffer_protocol` (`py::array_t` + `capsule`), `bool` specialization, `to_ndarray`/`to_pyarray`.
- **Isabelle** — `Hardware`/`Differential`/`Padic`/`Dual`/`Lattice`/`Spectral` sessions with `migrate_length`, `dot_row`, `SVD` lemmas (`isabelle build -D isabelle`).

### Changed
- **Performance** — `ndarray` `is_contiguous`/`_flat_logical`/`_for_each_logical` `[[likely]]` + `__restrict`, `linalg` `BLOCK=128` (`float` L3) + `fma` + `OpenMP` sharding, `memory` `madvise(HUGEPAGE)` + `TaggedArray` DRY, `polynomial` Horner `simd` batch, `window` `bartlett` etc., `bench_hardware` `0.00 ms` HBM zero-copy, `512×512` `13.73 ms` `→` `13.67 ms` (Auto), `1024×1024` `144 ms` (see `docs/PERFORMANCE.md`).
- **Build** — `CMakeLists.txt` `NP_ENABLE_SLEEF`/`POWERFUL` (`AVX2`+`GPU`+`OpenMP`+`LTO`+`Native`), `LLVM` `-std=c++17` filter, `numpy-cpp-llvm` OBJECT lib, `NP_WERROR` with allowlist.
- **CI** — `.github/workflows/ci.yml` matrix `asan`/`ubsan`/`tsan` (`ASAN_OPTIONS`/`UBSAN_OPTIONS`/`TSAN_OPTIONS`), `lint` `warnings-as-errors` (not `continue-on-error`), `benchmark` job (`bench_hardware`/`bench_math` + `PERFORMANCE.md` freshness), `isabelle` + `examples`.
- **Docs** — `README` `712` routines, `PERFORMANCE.md` real `GCC 14.2 -O3 -mavx2` numbers (`2025-09-03`), `ARCHITECTURE.md` file:line map.

### Fixed
- `half` tag clash (`half.hpp:19` vs `dtype.hpp:612`), `tensor_core` `matmul_fp16` manual loop (`uint16_t` mismatch), `linalg` `fma` `is_same_v<R,T,U>` gate, `cohomology` `(int)`→`static_cast<int>`, `bundle` `binom` cast, `threadpool` `raw new→make_unique` + `jthread` + `stop_token`, `photonics` `lock.unlock()` intentional, `io` `secure::save/load` `expected`, `spectral` `static_cast` etc.

### Security
- `NP_USE_SECURE_IMPL` hardened `zeros` + `secure_buffer` `mlock` + `DONTDUMP` + slack wipe, `secure_*` namespaces for `ndarray`/`matrix`/`polynomial`/`linalg`/`fft`/`io`.

## [2.2.0] — Previous dev (pre-1.0)
- Dev branch with 712 routines, header-only, `isabelle` proofs, `python` bridge.
