# Contributing — dev

Branch `dev` is the integration branch for micro-opts. `main` is stable (712 routines). PRs target `dev`.

## Workflow

1. **Branch** `git checkout -b feat/my-opt dev` (not `main`).
2. **Check ref** `numpy-reference/reference/generated/numpy.<func>.html` — match Python signature exactly (see `AGENTS.md`).
3. **Implement** in `include/np/<module>.hpp` with Doxygen `Reference:` link and `NP_API`.
4. **Test** `tests/test_<module>.cpp` using `tests/test_util.hpp` (`test::check`, `approx`).
5. **Format** `clang-format -i include/np/*.hpp` — `.clang-format`: 4-space, custom Allman-style braces, `ColumnLimit: 120`, `UseTab: Never`.
6. **Build** `cmake -S . -B build && cmake --build build -j8 && ctest --test-dir build --output-on-failure` — must be **49/49**.
7. **Commit** `feat(module): ...` with `file:line` (e.g. `ndarray.hpp:3116`). One logical task per commit, no `build/` artifacts (`CMakeCache.txt`, `build/` are in `.gitignore`).
8. **PR** to `dev` — include bench delta if perf-related (see `PERFORMANCE.md`).

## Micro-opt guidelines (dev)

* Guard fast path with `if (is_contiguous()) [[likely]]` + fallback `Odometer`/`_flat_logical` for views.
* Use `T* __restrict` for `vector<T>::data()` (except `vector<bool>` proxy — `if constexpr` fallback).
* `reserve` vectors before `push_back`; `[[unlikely]]` on throw paths.
* `std::memcpy` only when same shape + `is_contiguous()` + `offset==0`.
* Hash threshold `64` for `isin` (empirical); `BLOCK=32` for `dot`.
* Threadpool via `NP_USE_THREADING` — `parallel_for` threshold `>4096`.
* Add `[[nodiscard]]`, `noexcept` where appropriate.

## Style

* `snake_case` funcs, `PascalCase` types, Allman braces, `NP_API`/`NP_NODISCARD`.
* Include guards `#ifndef NP_<NAME>_HPP`.
* Namespace `np` (sub `np::linalg`, `np::datetime`, `np::testing`).

## Docs

* Update `README.md` (dev badge) and `docs/*.md` when adding routines.
* Keep `file:line` in doc tables greppable.
* Run `cmake --build build --target bench_math` for perf docs.

See `AGENTS.md` and `ARCHITECTURE.md` for layout (`include/np/detail/*` for `proxy.hpp`, `expr.hpp`).

## Release

`dev` → `main` squash after 49/49 + `clang-format` clean. Tag `vX.Y-dev` for bench.
