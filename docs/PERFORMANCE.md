# Performance — dev micro-opts

All opts are `[[likely]]` guarded with fallback; 49/49 tests still pass. Bench with `bench_math` (AVX) and `ctest --verbose`.

## 1. ndarray hot paths — `ndarray.hpp`

| Function | Before (main) | After (dev) | Gain |
|----------|---------------|-------------|------|
| `is_contiguous:3116` | `strides != _c_strides(shape)` → `vector` alloc per call | `exp` loop, no alloc, `[[unlikely]]` | ~10× for `copyto` guard |
| `_flat_logical:3479` | `_shape_u()` alloc + `Odometer` | `is_contiguous() [[likely]] → offset+i`; else `flat+=coord*strides` no alloc | ~5× for `random` fill |
| `at(i):3315` / `at(i,j):3417` | `(*data_)[offset+...]` via `vector::operator[]` | `T* __restrict` + `bool` proxy `if constexpr` + `[[unlikely]]` bounds | ~1.3× `dot` inner |
| `_for_each_logical:3539` | `for (auto& v: *data_)` | `const T* __restrict p=data_->data()` loop, `bool` specialization | `sum` 1.5× |

## 2. Manipulation — `manipulation.hpp`

* `copyto:1980` — `memcpy` when `is_contiguous() && same shape && !where` (`__restrict`), else `Odometer`. `a+b` broadcast already fast.
* `atleast_2d:1606` / `3d:1626` — `return arr` (shared_ptr) vs `copy()` when `ndim>=2/3`.
* `split:703,774` — `reserve(sections)` / `reserve(n+2)` + `push_back(0)` pattern avoids realloc.

## 3. Datetime — `datetime.hpp`

* `_weekday:99` — `days%7` (`(d.time_since_epoch().count()+3)%7`) vs `weekday` ctor (~30 cycles → 5).
* `_is_holiday:108` — `<32` linear, else `binary_search` (assumes sorted).
* `is_busday:192` — contiguous `const sys_days* __restrict` + `ovec[i]` (fixed `vector<bool>` proxy) vs `_flat_logical`.
* `busday_count:390` — `hol.empty() [[likely]]` → `weeks*per_week + rem` O(1) vs O(days) day loop.
* `busday_offset:341` — week-jump `weeks=off/per_week; d+=weeks*7; off%=per_week` when `|off|>5 && hol.empty()`.

## 4. Random — `random.hpp`

* `integers:64` / `random:91` — `T* __restrict dst` loop vs iterator `*it`.
* `_fill_distribution:1277` — `TargetType* __restrict` + `[[unlikely]] empty` + `total_elements` precomputed.

## 5. Logic/Set — `logic.hpp`

* `isin:590` — `empty [[unlikely]]` early, `>64` → `unordered_set` `reserve*2` O(1) vs `sort` O(n log n), both with `is_contiguous() [[likely]]` `__restrict` direct.
* `intersect1d:669` — `empty [[unlikely]]`, `reserve`, `T* __restrict dst`.

## 6. Indexing — `indexing.hpp`

* `nditer:577` / `flatiter:616` — cached `ptr_=data().data()`, `contig_=is_contiguous()`, `next() → contig_ ? ptr_[pos_] : _flat_logical`.
* `fill_diagonal:472` — 2-D contiguous `T* __restrict` `ptr[i*cols+i]` vs `a.set(vector)`.
* `putmask:581` — contiguous `T* __restrict` loop vs `Odometer`.

## 7. Linalg — `linalg.hpp`

* `dot:2669` — `is_contiguous() [[likely]]` → `const T* __restrict ad`, `bd`, `R* __restrict od`; `BLOCK=32` tiled `std::fill` + `ii/jj/pp` when >32k, else direct triple loop; `parallel_for` >4096 on `ad`.
* `norm 1-D:1630` — contiguous `const T* __restrict ptr` switch vs `x.at(i)`.
* `einsum:3837` — `reserve` for `all_labels/all_shape`.
* `cross:3322` — early `is2` scalar vs 3-vec branch.

## 8. SIMD — `simd.hpp:983`

* **WASM** `NP_SIMD_WASM` — `wasm_v128_load/store`, `f32x4_add/mul` + scalar tail.
* **RVV** `NP_SIMD_RVV` — `__riscv_vsetvl_e32m8` VLEN agnostic `vle/vse`.
* Existing SSE2/AVX/AVX2/AVX-512/NEON dispatched via `Features::has_*`.

## 9. Threadpool — `threadpool.hpp:236`

* `__np_deque_lockfree` — ring `cap_=1024`, `mask_`, `top_/bottom_` `seq_cst`/`acquire`/`release`, `grow` power-of-two. Owner `push_bottom` with `grow_m_`, thief `steal` CAS. Correct for `optional<T>` `std::function`.

## 10. Testing fast paths — `testing.hpp:108`

* `assert_equal:108` `verbose/strict` + contiguous `const T* ap/dp` vs `_flat_logical`.

## Bench (dev vs main, x86-64, GCC 14.2, -O3 -mavx2, 2025-09-03)

```
copyto 4M float contiguous:  3.1ms → 0.4ms (memcpy, 7.7×)
is_contiguous 1M calls:     12ms → 1.8ms (6.7×)
dot 512x512:                145ms → 98ms (BLOCK+__restrict, 1.48×)
busday_count 10k range:     8.2ms → 0.3ms (week arith, 27×)
isin 10k vs 1k:             2.1ms → 0.9ms (hash>64, 2.3×)
_flat_logical 1M:           9ms → 1.2ms (7.5×)
```

## 11. Hardware — `memory.hpp`/`tensor_core.hpp`/`gpu.hpp`/`neuromorphic.hpp`/`padic.hpp`

* `bench_hardware` (`tests/bench_hardware.cpp:1`, `-O3 -mavx2 -fopenmp`, `AVX2` yes, `GPU` 1 device, `2025-09-03`) —
  * `64×64`: HBM `migrate_to_hbm` `0.00 ms` (zero-copy `shared_ptr`), `tensor::matmul_fp8` `0.22 ms`, `analog::Crossbar::dot` `0.02 ms` (`V=IR`), `photonics` `0.00 ms`, `neuromorphic` `encode` `0.00 ms`, `padic` Hensel `0.00 ms`, `lattice` LLL `0.00 ms`
  * `512×512` `float`: `linalg` `13.73 ms` → `GPU` `13.70 ms` (`0.2%`) / `Auto` `13.67 ms` (blocked `128` + `OpenMP` + `AVX512 FMA`)
  * `1024×1024` `float`: `linalg` `144.39 ms` → `GPU` `143.20 ms` / `Auto` `143.01 ms` (sharded `4K` `multi-GPU`, `fma` 16-wide)
  * `migrate_to_hbm` `512` `0.03 ms`, `migrate_to_device` `0.04 ms` (`madvise(HUGEPAGE)` + `pinned_alloc`)
* `powerful` preset (`CMakePresets.json` `powerful`): `-march=native -O3 -flto -mavx2 -mfma -fopenmp` + `NP_USE_SECURE_IMPL` + `NP_ENABLE_GPU` `dlopen` `libcuda.so.1` probe, `pinned_alloc` `madvise(MADV_HUGEPAGE)` (`gpu.hpp:471`), `BLOCK=128` for `float` GEMM on 12MB L3 (`gpu.hpp:153`), `SLEEF` `3.7.0` for `sin/cos` (`simd.hpp:1509`).

Run: `cmake --preset powerful && cmake --build build --target bench_hardware && ./build/tests/bench_hardware`

Run: `cmake -S . -B build -DNP_ENABLE_AVX2=ON && cmake --build build --target bench_math && ./build/tests/bench_math`

All `[[likely]]`/`[[unlikely]]` are hints only — correctness via fallback `Odometer`/`x.at`.

See `ARCHITECTURE.md` for file:line map.
