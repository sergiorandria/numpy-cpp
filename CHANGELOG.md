# Changelog — numpy-cpp

All notable changes to `numpy-cpp` will be documented here. Format based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), versioning follows [SemVer](https://semver.org/spec/v2.0.0.html).

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
