#numpy - cpp

> **NumPy 2.2 in C++20. Header-only. Zero Python. Native speed.**

[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg?style=flat-square&logo=c%2B%2B)](https://en.cppreference.com/w/cpp/20)
[![CMake](https://img.shields.io/badge/CMake-3.20%2B-064F8C.svg?style=flat-square&logo=cmake)](https://cmake.org)
[![Header-only](https://img.shields.io/badge/header--only-Yes-brightgreen?style=flat-square)](include/np/np.hpp)
[![NumPy](https://img.shields.io/badge/NumPy-2.2-013243.svg?style=flat-square&logo=numpy)](https://numpy.org/doc/stable/)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-green?style=flat-square)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-49%2F49-brightgreen?style=flat-square)](#testing)
[![SIMD](https://img.shields.io/badge/SIMD-SSE4.2%20%7C%20AVX2%20%7C%20AVX--512%20%7C%20NEON%20%7C%20WASM%20%7C%20RVV-orange?style=flat-square)](#performance)

**numpy-cpp** is a complete, header-only C++20 reimplementation of the NumPy 2.2 API — **760+ routines** across **36 modules**, with NumPy-identical semantics. Include one header, get the whole scientific stack at compiled speed.

> Parity shims: a handful of `numpy.distutils` / `ctypeslib` compat helpers in `other.hpp` are intentional thin stubs (documented in [Known Divergences](#known-divergences)), and optional PQC KEM/signature wrappers default to stubs unless `NP_PQC_ALG` is set. Everything in the NumPy surface above has a real implementation plus scalar fallback.

```cpp
#include <np/np.hpp> // now fully integrated (random + concatenate included)

int main() {
  auto a = np::arange<double>(0, 10, 0.5); // [0, 0.5, …, 9.5]
  auto b = np::linspace<double>(0, 2 * M_PI, 100);
  auto y = np::sin(b); // ufunc, SIMD-dispatched

  auto M = np::eye<double>(3);
  auto N = np::ones<double>({3, 3});
  auto P = np::matmul(M, N); // blocked GEMM, parallel

  auto rng = np::random::Generator(42);
  auto s = rng.standard_normal<double>({1000}); // PCG64 + Box-Muller
  auto m = np::mean(s);
}
```

No linking. No Python runtime. No code generation. Just `#include <np/np.hpp>`.

---

## Table of Contents

- [Why numpy-cpp?](#why-numpy-cpp)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Coverage](#api-coverage)
- [Performance](#performance)
- [Architecture](#architecture)
- [Project Layout](#project-layout)
- [Testing & Benchmarks](#testing--benchmarks)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [Known Divergences](#known-divergences)
- [License](#license)

---

## Why numpy-cpp?

| You know NumPy… | …so you already know numpy-cpp |
|-----------------|-------------------------------|
| `np.zeros((3,4))` | `np::zeros<double>({3,4})` |
| `np.linalg.norm(a)` | `np::linalg::norm(a)` |
| `np.fft.fft(x)` | `np::fft::fft(x)` |
| `np.random.default_rng(0)` | `np::random::Generator(0)` |
| `np.is_busday(d)` | `np::is_busday(d)` |

* **Familiar** — every routine matches the [NumPy reference](https://numpy.org/doc/stable/reference/) signature and carries a Doxygen `Reference: numpy-reference/...` link.
* **Zero-overhead** — header-only `INTERFACE` library (`cmake --install` just copies headers). Views are `shared_ptr` aliases, not copies. Contiguous fast paths use `memcpy` / direct `T* __restrict`.
* **Portable SIMD** — auto-detected: SSE4.2 / AVX2 / AVX-512 on x86-64, NEON on ARM64, WASM SIMD128, RISC-V Vector, POWER VSX. Scalar fallback always correct.
* **Two array engines** — `ndarray<T>` (dynamic, heap) + `ndarrayf<T, Extents...>` (fixed, stack, `constexpr`-foldable).
* **Production-ready** — lock-free Chase-Lev threadpool, `49/49` CTest suites, `clang-format` enforced, BSD-3-Clause.

> If you embed scientific computing in C++ — games, robotics, trading, edge inference — numpy-cpp lets you keep NumPy semantics without shipping Python.

---

## Features

- **Creation** — `zeros`, `ones`, `empty`, `full`, `arange`, `linspace`, `logspace`, `geomspace`, `eye`, `meshgrid`, `fromiter`, …
- **Manipulation** — `reshape`, `transpose`, `broadcast_to`, `pad`, `split`, `block`, `copyto` (memcpy fast path), …
- **Math** — 112+ ufuncs (`sin`, `exp`, `floor`, `nextafter`, `trapz`, …) + `emath` + `window` (Bartlett, Kaiser, …)
- **Logic & Set** — `isfinite`, `isclose`, `isin` (hash `>64`), `intersect1d`, `unique`, …
- **Sorting** — `sort`, `argsort`, `lexsort`, `partition`, `searchsorted`, `nonzero`
- **Statistics** — `mean`, `var`, `std`, `median`, `quantile`, `histogram`, `histogramdd`
- **Linear algebra** — `dot` (blocked `32`), `matmul`, `tensordot`, `einsum` + `einsum_path`, `qr`, `eig`, `cholesky`, `norm`, `cross`
- **FFT** — `fft`, `rfft`, `fftn`, `fftshift` (18 ops, Bluestein for arbitrary `n`)
- **Random** — `Generator` (PCG64) with 50+ distributions + `SeedSequence`
- **Datetime** — `is_busday`, `busday_count` (O(1) week arithmetic), `busday_offset`, `datetime_as_string`
- **Strings** — `np::ch` / `np::strings` — 40+ `add`, `center`, `encode`, `is*`
- **I/O** — `load`, `save`, `savez`, `NpzFile`, `savetxt`, `DataSource`
- **Polynomial** — `Polynomial`, `Chebyshev`, `polyfit`, `polyutils`
- **Dtype & Masked** — `can_cast`, `promote_types`, `finfo`/`iinfo`, `MaskedArray`
- **Extras** — `bigint` (Boost `cpp_int` / GMP), `pqc` constant-time hardening (KEM/signature wrappers are opt-in stubs unless `NP_PQC_ALG` is set), `differential` LLVM JIT (optional, interpreter fallback), `homology`/`homotopy`/`manifold`/`variety`, `lattice`/`padic`, `neuromorphic` (CPU LIF simulation, no Loihi2/SpiNNaker hardware), `memory` (host storage with HBM/CXL placement *hints*, no device migration), `tensor` (blocked-CPU / FP32-GPU dispatch, FP8 is quantize-around-FP32, no Hopper/AMX tile path), `analog` (ReRAM crossbar simulation), `photonics` (Mach-Zehnder unitary math + simulation), `quantum` (in-house state-vector simulation), `accelerator` (heterogeneous CPU/GPU/sim dispatch)

---

## Installation

**Requires:** C++20 (GCC 14+ or Clang 15+), CMake 3.20+ (3.28 recommended).

### 1 — Copy (header-only)

```bash
cp -r include/np /usr/local/include/
#then
g++ -std=c++20 -O3 -I include main.cpp -o main
```

### 2 — CMake `add_subdirectory` / `FetchContent` (recommended)

```cmake
#CMakeLists.txt
cmake_minimum_required(VERSION 3.20)
project(my_app CXX)
set(CMAKE_CXX_STANDARD 20)

#Option A : local checkout
add_subdirectory(numpy-cpp)
#Option B : FetchContent
include(FetchContent)
FetchContent_Declare(numpy-cpp GIT_REPOSITORY https://github.com/sergiorandria/numpy-cpp.git GIT_TAG dev)
FetchContent_MakeAvailable(numpy-cpp)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE numpy-cpp::numpy-cpp)
#legacy alias also works : np::np
```

```cpp
// main.cpp
#include <np/np.hpp>
#include <np/random.hpp>
int main() {
  auto a = np::arange<double>(0, 10, 0.5);
  auto s = a.sum();
}
```

### 3 — CMake install

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local
cmake --build build && cmake --install build
#then in downstream:
find_package(numpy-cpp CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE numpy-cpp::numpy-cpp)
```

#### SIMD / Tuning Options

| Option | Default | Effect |
|--------|---------|--------|
| `NP_ENABLE_SIMD` | `ON` | SSE4.2 / NEON auto |
| `NP_ENABLE_AVX2` | `OFF` | `-mavx2 -mfma` / `/arch:AVX2` |
| `NP_ENABLE_AVX512` | `OFF` | `-mavx512f -mavx512dq` |
| `NP_ENABLE_SVE` | `OFF` | ARM SVE (`-march=armv8-a+sve`) |
| `NP_ENABLE_RVV` | `OFF` | RISC-V Vector (`rv64gcv`) |
| `NP_ENABLE_WASM_SIMD` | `OFF` | WASM SIMD128 (`-msimd128`) |
| `NP_ENABLE_VSX` | `OFF` | POWER VSX (`-mcpu=power9`) |
| `NP_ENABLE_LTO` | `OFF` | IPO/LTCG |
| `NP_ENABLE_NATIVE` | `OFF` | `-march=native` / `/Ot` |
| `NP_ENABLE_FAST_MATH` | `OFF` | `-ffast-math` / `/fp:fast` |
| `NP_USE_THREADING` | `OFF` | Chase-Lev threadpool for sort/linalg |
| `NP_ENABLE_BIGINT` | `ON` | `boost::multiprecision::cpp_int` |
| `NP_ENABLE_GMP` | `OFF` | GMP `mpz` backend |
| `NP_ENABLE_PQC` | `ON` | Constant-time hardening |

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DNP_ENABLE_AVX2=ON -DNP_ENABLE_LTO=ON -DNP_USE_THREADING=ON
```

---

## Quick Start

```cpp
#include <np/np.hpp>
#include <np/random.hpp>
#include <np/linalg.hpp> // explicit for linalg if needed
#include <iostream>

int main() {
  // — Creation & arithmetic (broadcasting) —
  auto a = np::zeros<double>({3, 4});
  auto b = np::ones<double>({3, 4});
  auto c = a + b * 2.0; // element-wise, broadcast

  // — Reductions & views —
  auto v = np::arange<int>(0, 12).reshape({3, 4});
  auto row = v[1]; // view, shared storage
  std::cout << v.sum() << " " << np::mean(v) << "\n";

  // — Linalg —
  auto A = np::eye<double>(4);
  auto x = np::arange<double>(0, 4);
  auto y = np::matmul(A, x.reshape({4, 1}));
  auto n = np::linalg::norm(y);

  // — FFT —
  auto t = np::linspace<double>(0, 1, 64);
  auto sig = np::sin(t * (2 * M_PI * 5));
  auto F = np::fft::rfft(sig);

  // — Datetime (week arithmetic, O(1)) —
  using namespace std::chrono;
  auto d = np::datetime::sys_days{days{18500}};
  bool workday = np::is_busday(d);

  // — Random —
  auto rng = np::random::Generator(0);
  auto z = rng.standard_normal<double>({2, 3});

  (void)c;
  (void)row;
  (void)n;
  (void)F;
  (void)workday;
  (void)z;
}
```

Compile:

```bash
g++ -std=c++20 -O3 -msse4.2 -I include main.cpp -o main && ./main
#WASM
clang++ --target=wasm32 -mwasm-simd128 -DNP_SIMD_WASM -I include main.cpp -o main.wasm
#RISC - V
clang++ -march=rv64gcv -DNP_SIMD_RVV -I include main.cpp -o main.rvv
```

---

## API Coverage

> 760+ routines, 36 groups — every `np::` carries `Reference: numpy-reference/...` or Hatcher/Bott–Tu.

| Module | Header | Highlights |
|--------|--------|------------|
| **Constants** | `constants.hpp` | `pi`, `e`, `euler_gamma`, `inf`, `nan`, `newaxis` |
| **Creation** | `creation.hpp` / `creation_fixed.hpp` | `zeros`, `ones`, `full`, `arange`, `linspace`, `eye`, `meshgrid` |
| **Manipulation** | `manipulation.hpp:1980` | `copyto` (memcpy), `reshape`, `transpose`, `pad`, `block` |
| **Concatenate** | `concatenate.hpp` | `concatenate`, `stack`, `vstack`/`hstack`/`dstack` |
| **Bitwise** | `bitwise.hpp` | `bitwise_and`/`or`/`xor`, `invert`, `packbits` |
| **Math** | `math.hpp` / `emath.hpp` | 112 ufuncs, `trapz` |
| **Strings** | `char.hpp` | `np::ch` — 40+ string ops |
| **Logic** | `logic.hpp:590` | `isclose`, `isin` (hash), `intersect1d` |
| **Functional** | `functional.hpp` | `apply_along_axis`, `vectorize`, `piecewise` |
| **Datetime** | `datetime.hpp:99` | `is_busday`, `busday_count`, `busday_offset` |
| **Dtype** | `dtype.hpp` | `can_cast`, `promote_types`, `finfo`/`iinfo` |
| **Masked** | `masked_array.hpp` | `MaskedArray`, `masked_where` |
| **Indexing** | `indexing.hpp:576` | `Slice`, `nditer`, `flatiter`, `ix_` |
| **Sorting** | `sorting.hpp` | `sort`, `argsort`, `lexsort`, `searchsorted` |
| **Statistics** | `statistics.hpp` | `mean`/`var`/`std`, `quantile`, `histogramdd` |
| **Linalg** | `linalg.hpp:2669` | `dot` (BLOCK=32), `matmul`, `einsum`, `norm`, `qr`/`eig`/`cholesky` |
| **FFT** | `fft/fft_core.hpp:244` | `fft`, `rfft`, `fftn` (18 ops) |
| **Random** | `random.hpp:64` | `Generator`, `SeedSequence`, PCG64, 50 dists |
| **I/O** | `io.hpp` | `load`, `savez`, `NpzFile`, `savetxt` |
| **Polynomial** | `polynomial.hpp` | `Polynomial`, `Chebyshev`, `polyfit` |
| **SIMD** | `simd.hpp:983` | WASM `v128` + RVV `__riscv_vsetvl` |
| **Window** | `window.hpp` | `bartlett`, `kaiser`, `hamming`, … |
| **Testing** | `testing.hpp:108` | `assert_equal`, `assert_allclose`, `Tester` |
| **Threadpool** | `threadpool.hpp:236` | Chase-Lev work-stealing (`SPAA 2005`) |

Full per-symbol table: [`docs/API.md`](docs/API.md).

---

## Performance

All fast paths are `[[likely]]`-guarded with exact fallback — correctness first, speed second.

| Hot path | Optimization | Where |
|----------|--------------|-------|
| `is_contiguous()` | no `_c_strides` alloc, `exp` loop | `ndarray.hpp:3116` |
| `_flat_logical` / `_for_each_logical` | `is_contiguous → offset+i` direct `T*` | `ndarray.hpp:3479`, `3539` |
| `copyto`, `block`, `broadcast` | `memcpy` when contiguous + `__restrict` | `manipulation.hpp:1980` |
| `isin` | `size>64 → unordered_set` else `sort` (empirical crossover) | `logic.hpp:590` |
| `dot` / `matmul` | blocked GEMM `BLOCK=32` + `parallel_for` `>4096` | `linalg.hpp:2669` |
| `norm` | contiguous `ptr` fast path | `linalg.hpp:1630` |
| `busday_count` | `weeks*5 + remainder` O(1) when `hol.empty()` | `datetime.hpp:390` |
| `SIMD` | SSE4.2/AVX2/AVX-512/NEON/WASM/RVV dispatched ufuncs | `simd.hpp:983` |
| `Threadpool` | Chase-Lev deque, CAS `top`/`bottom`, power-of-two grow | `threadpool.hpp:236` |

Build for max throughput:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DNP_ENABLE_AVX2=ON -DNP_ENABLE_LTO=ON -DNP_ENABLE_NATIVE=ON -DNP_USE_THREADING=ON
cmake --build build -j && ./build/tests/bench_math  # AVX micro-bench (not in ctest)
```

See [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md) for micro-benchmarks and [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for engine internals.

---

## Architecture

```
ndarray<T>  ── shared_ptr<vector<T>> + shape/strides/offset ──► views alias storage
                 is_contiguous() [[likely]] ──► ptr[i] else _flat()
                 _for_each_logical ──► direct T* when contiguous

ndarrayf<T, Extents...> ── std::array<T,N> on stack, constexpr shape
                 expr.hpp ── lazy fused element-wise trees, broadcast-checked at compile time

dispatch ── NP_SIMD_* macros ──► simd.hpp (SSE/AVX/NEON/WASM/RVV) ──► scalar tail
threadpool ── Chase-Lev deque ──► parallel_for over contiguous spans
```

Two engines, one API: `ndarray<T>` for runtime shapes, `ndarrayf<T, Extents...>` for compile-time shapes and `constexpr` folding (`tests/test_constexpr.cpp`).

---

## Project Layout

```
include/np/
  np.hpp              umbrella (28 includes; fully integrated)
  ndarray.hpp         6.5k LOC — core array, views, strides
  linalg.hpp          4.0k LOC — BLAS-like, blocked GEMM, decompositions
  manipulation.hpp    2.6k LOC — reshape, copyto, block, pad
  random.hpp          1.8k LOC — PCG64 Generator, 50 distributions
  math.hpp / emath.hpp  3k LOC — 112 ufuncs
  simd.hpp            1.1k LOC — portable SIMD kernels
  threadpool.hpp      1.0k LOC — lock-free work stealing
  fft/                18 ops — Cooley-Tukey + Bluestein
  ... 26 headers total
docs/
  README.md           docs index
  ARCHITECTURE.md     dual engines, views, strides
  PERFORMANCE.md      bench & fast-path catalogue
  API.md              per-module file:line table
  CONTRIBUTING.md     workflow & style
  MATH_PROOFS.md      correctness proofs vs NumPy ref
tests/                49 CTest suites + bench_math / bench_hardware (AVX, manual)
```

---

## Testing & Benchmarks

```bash
cmake -S . -B build && cmake --build build -j8
ctest --test-dir build --output-on-failure   # 49/49

#single suite verbose
./build/tests/test_ndarray --verbose
#header - only smoke test
g++ -std=c++20 -I include tests/test_math.cpp -o /tmp/t && /tmp/t

#micro - benchmark(AVX, not in ctest)
cmake --build build --target bench_math && ./build/tests/bench_math
```

CI target is `49/49` green. Every fast path has a scalar fallback exercised by tests.

---

## Documentation

| Doc | What |
|-----|------|
| [`docs/README.md`](docs/README.md) | Docs index |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Engines, views, strides, threadpool |
| [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md) | Fast paths, thresholds, bench |
| [`docs/API.md`](docs/API.md) | 760+ routines, file:line |
| [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md) | Branching, `clang-format`, `feat()` commits |
| [`docs/MATH_PROOFS.md`](docs/MATH_PROOFS.md) | Proofs vs `numpy-reference/` |

Doxygen per function: `grep -n "Reference:" include/np/*.hpp`.

---

## Contributing

1. Branch from `dev`: `git checkout -b feat/my-opt dev`
2. Match NumPy signature exactly — check `numpy-reference/reference/generated/numpy.<func>.html`.
3. Implement in `include/np/<module>.hpp` with Doxygen `Reference:` link.
4. Format: `clang-format -i include/np/*.hpp` (`.clang-format`: 4-space, Allman-style custom braces, `ColumnLimit: 120`).
5. Add `tests/test_<module>.cpp` using `tests/test_util.hpp` (`test::check`, `test::approx`).
6. Register in `tests/CMakeLists.txt` `NP_TESTS`.
7. `cmake --build build && ctest --output-on-failure` — commit `feat(module): ...` with `file:line`.

See [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md) and `AGENTS.md`.

---

## Known Divergences

* `operator[](i,j)` is C++23 — use `arr(i,j)` or `arr[i][j]` proxy (`ndarray.hpp:3315`).
* Complex `linalg` is real-only (`is_complex_v` static-assert) — dispatches to real `double`.
* `ndarray<bool>` uses proxy reference (`vector<bool>` bitset); `is_contiguous()` aware.
* `numpy.distutils` / `ctypeslib` are thin `other.hpp` stubs (`who`, `disp`, `info`, `source`, `lookfor`, `deprecate`, `show_config`, buffer-size helpers) plus `einsum_path_stub` (real path logic lives in `linalg.hpp`). PQC KEM/signature wrappers are opt-in stubs unless `NP_PQC_ALG` selects a backend.

---

## License & References

BSD-3-Clause (same as NumPy). See [`LICENSE`](LICENSE).

Independent reimplementation — not affiliated with NumPy. NumPy docs: <https://numpy.org/doc/stable/>. C++20 standard: <https://en.cppreference.com/w/cpp/20>.

**Author:** Sergio Randriamihoatra — `dev` branch micro-opts `f7b2653..cf8f4a4`.

> If numpy-cpp saves you a Python dependency, give it a star — it helps the project stay header-only and dependency-free.
