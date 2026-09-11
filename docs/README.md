# Docs — dev branch

This folder is the **rewritten documentation for `dev`** (header-only, 760+ routines, 49/49 tests). `main`’s README is the stable user guide; here we document internals, micro-opts, and benchmarks introduced in `f7b2653..cf8f4a4`.

## Index

| Doc | Purpose | Key file:line |
|-----|---------|---------------|
| [Architecture](ARCHITECTURE.md) | Dual engines, views, strides, dtype, threadpool | `ndarray.hpp:3116`, `linalg.hpp:2669`, `threadpool.hpp:236` |
| [Performance](PERFORMANCE.md) | `is_contiguous` fast, `copyto` memcpy, `isin` hash, blocked GEMM, week arithmetic, WASM/RVV | `datetime.hpp:99`, `logic.hpp:590`, `simd.hpp:983` |
| [API](API.md) | Per-module table 36 groups, 760+ routines, file:line | `np.hpp:13` umbrella |
| [Contributing](CONTRIBUTING.md) | Dev workflow, `feat(module):` commits, `clang-format`, `ctest` | `AGENTS.md`, `.clang-format` |

Start with `../README.md` (dev quick start) → `ARCHITECTURE.md` → `PERFORMANCE.md` for the micro-opt story.

## Quick links

* **Build:** `cmake -S . -B build && cmake --build build -j8 && ctest --test-dir build` — see `../README.md#testing`
* **Single-header:** `g++ -std=c++20 -I include main.cpp -o main` (`np.hpp` umbrella, `random.hpp` explicit)
* **NumPy ref:** `numpy-reference/reference/generated/numpy.<func>.html` — every `np::` Doxygen has `Reference:` link
* **Bench:** `cmake --build build --target bench_math && ./build/tests/bench_math` (AVX, not in ctest)

## Dev vs main

| Aspect | `main` | `dev` (this branch) |
|--------|--------|---------------------|
| `is_contiguous` | `strides != _c_strides` alloc | `exp` loop, no alloc, `[[unlikely]]` |
| `_flat_logical` | `_shape_u()` vector alloc + `Odometer` | `is_contiguous() [[likely]] → offset+i` |
| `copyto` | `Odometer` per element | `memcpy` `__restrict` fast |
| `isin` | `sort` only | `>64` → `unordered_set` |
| `dot` | `a.get` per inner | `BLOCK=32` + `__restrict` + `parallel_for` |
| `SIMD` | SSE2/AVX/NEON | + WASM `v128` + RVV `__riscv_vsetvl` |
| `Threadpool` | mutex `dq_` | Chase-Lev ring `top/bottom` CAS |

All 49 tests still pass — micro-opts are `[[likely]]` guarded with fallback.

## How to read

* `file:line` references are greppable: `grep -n "is_contiguous" include/np/ndarray.hpp`
* `dev` branch log: `git log dev --oneline -10` → `e2f9e3e`, `f7b2653`, `cf8f4a4`
* To diff vs `main`: `git diff main..dev --stat` (7 files, ~1200 lines micro-opts + docs rewrite)

---

*Generated for `dev` at 2025-08-31 — header-only, C++20, `shared_ptr` views.*
