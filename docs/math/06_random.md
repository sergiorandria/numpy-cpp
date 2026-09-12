# 06 — Random (50 distributions) — `random.hpp:42`

## Model

`Generator` wraps `mt19937_64 engine_` seeded via `SeedSequence:1311` (splitmix64). Each `Generator::normal` etc is `std::distribution<T>(params)(engine_)` → `ndarray`.

## Claim

`integers(low,high,size)` → `n=_numel()` samples `uniform_int_distribution(low,high-1)` same as `numpy.random.Generator.integers`. `random` → `uniform_real(0,1)`, `normal` → `normal_distribution`, `beta` → `gamma` ratio, `dirichlet` via `gamma` normalized, `multivariate_normal` via `cholesky`.

## Optimization

**Dev:** `integers:64` / `random:91` `T* __restrict dst = result.data().data(); for i<n dst[i]=dist(engine_)` vs `*it` iterator — same `dist(engine_)` sequence, direct `__restrict` enables vectorization.

`_fill_distribution:1277` `TargetType* __restrict` loop, `[[unlikely]] empty` early, `total_elements` precomputed.

`shuffle:161` 1-D `is_contiguous() [[likely]]` `T* __restrict p` + `std::shuffle(p, p+n)` vs `begin()`; `indices.reserve(n0)` + `push_back`.

`permutation`, `choice`, `spawn` via `engine_` fork.

## Correctness

`SeedSequence` splitmix ensures `spawn` independent streams, matches NumPy `SeedSequence`. `permuted` etc via `shuffle`.

Complexity O(n) per sample, `dirichlet` O(n*k).

