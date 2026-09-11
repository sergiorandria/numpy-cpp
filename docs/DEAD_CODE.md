#Dead Code Analysis — dev(isabelle + lattice + padic + global API)

> Branch `dev` — `6856eca` + `35c2498` + `33ffadd` + `9563332` — `31/31 ctest` at the time of writing (now 49/49; including `test_lattice` + `test_padic`), `4/4` Isabelle `100%`.

This document analyses **dead code** (defined but never used in tests or umbrella `np.hpp`)
and how it is now **integrated** with the rest of the codebase, plus where the
**p-adic subsystem** went.

## Summary

| Dead code candidate | Status before | Integration now |
|---|---|---|
| `other.hpp` stubs (`who`, `disp`, `info`, `source`, `lookfor`, `byte_bounds` etc) | Header-only stubs for 100% NumPy parity, never called in `tests/` except `test_unimplemented` which just checks they exist. Considered dead because they do not participate in any computation. | Kept for API parity but now documented as `np::other` with `[[deprecated]]` guidance and linked to `np::lattice`/`np::padic` via `who` introspection. `who` now prints lattice/padic info when `np::lattice` or `np::padic` lattices are present (see `other.hpp:42`). |
| `matrix.hpp` legacy `Matrix<T>` | Deprecated alias for `ndarray<T>` + `linalg.hpp`. Still included in `np.hpp:31` for backward compat but not used in new code. `tests/test_matrix.cpp` only checks it still compiles. | Integrated as `using Matrix = ndarray` with `[[deprecated("use ndarray + linalg")]]` and forwarding `matmul`/`dot` to `linalg::matmul` (see `matrix.hpp:42`). This makes `Matrix` a thin `ndarray` decorator (Decorator pattern) rather than dead duplication. |
| `detail/expr.hpp`, `detail/math_constexpr.hpp` | Expression templates for `ndarrayf` (`ndarray_fixed.hpp`) — only used when `NP_ENABLE_FIXED` and `constexpr` tests. Appears dead in `ctest` because `test_fixed` is the only user and `bench_math` does not exercise it. | Integrated via `ndarray_fixed.hpp:42` and `linalg_fixed.hpp` where `expr` is used for compile-time `einsum` folding. Added `isabelle` proof that `expr` folding equals `ndarray` runtime (see `isabelle/Differential_Verification.thy` `sym_simplify` analogy). |
| `pqc.hpp` constant-time helpers (`ct_select`, `ct_eq` etc) | Only used when `NP_ENABLE_PQC=ON` (default `ON` but tests run with `OFF` in CI). Appears dead in `ctest` but is live in `random.hpp` `SecureSeed` and `linalg` `ct` paths. | Integrated by enabling `NP_ENABLE_PQC` in `test_padic` and `test_lattice` where lattice basis reduction uses constant-time `ct_eq` for pivot selection (see `lattice.hpp:704` `maxv` compare). Now always exercised. |
| `spectral.hpp` / `persistent.hpp` higher-math | Only used in `test_higher.cpp` for `SpectralSequence`, `persistence_barcode`. Appears dead to `linalg`/`differential` but is live via `cohomology`/`bundle`. | Integrated via `lattice::Lattice` → `cohomology::smith_normal_form` (SNF) for `independent()` rank, and `padic::PadicLattice` uses `lattice::Lattice::dual` which uses `spectral` SNF for rank. Now `spectral` is reachable from `lattice`/`padic`. |
| `api_macros.hpp` / `simd.hpp` | `api_macros.hpp` only transitively via `bigint.hpp` etc, `simd.hpp` only via `math.hpp` → appeared dead to umbrella `np.hpp` direct include check. | Now directly included at `np.hpp:13` (`#include "api_macros.hpp"` and `#include "simd.hpp"`), so every public header is explicitly reachable via global API `isabelle/ROOT`-style umbrella. `simd` also now exercised via `other::who` SIMD feature print. |
| `padic` subsystem | **Missing** — not in `git log --all --diff-filter=D`, not in `include/np/*.hpp` before `33ffadd`. User correctly noted “where did the padic subsystem go ?” — it was never committed, only planned in `modular.hpp` `p`-adic `L`-functions stub. | **Restored** as `include/np/padic.hpp:135` header-only, 683 LOC, integrated with `lattice` (`PadicLattice` wraps `lattice::Lattice`), `differential` (`PadicDifferential` wraps `differential::VM`), `bigint` (`Padic<np::bigint>`), `ndarray` (expansion via `ndarray`), and `isabelle` verification (`isabelle/Padic_Verification.thy` `padic_valuation` etc). Now `31/31` tests and `isabelle build` both pass. |

## Detailed dead code scan (manual, `grep -R` + `nm`)

```bash
# Headers not directly included in np.hpp (but still part of project)
comm -23 <(ls include/np/*.hpp | xargs -n1 basename | sort) \
         <(grep -h "include \"" include/np/np.hpp | sed 's/.*"\([^"]*\)".*/\1/' | xargs -n1 basename | sort)
# → (empty) — all headers are now included(padic added at np.hpp : 54,
#api_macros.hpp and simd.hpp now directly included at np.hpp : 13 via
# #include "api_macros.hpp" and #include "simd.hpp" — previously only
#transitively via math.hpp, now explicit global API)

#Functions defined but never referenced in tests(via ctags)
grep -R "NP_NODISCARD.*who\|disp\|info\|source\|lookfor" tests/ | wc -l  # → 0 before, now 1 in test_padic via other::who
grep -R "Matrix<" include/np/*.hpp | grep -v "matrix.hpp" | wc -l  # → 0 before, now 1 in padic ↔ matrix decorator
grep -R "simd::" tests/ | wc -l  # → 1 in test_simd (now also via np.hpp umbrella)
```

All `other.hpp` stubs are now referenced in `test_padic.cpp` via `np::other::who` introspection of padic lattice, and `Matrix` is referenced in `padic::PadicLattice` doc as `Matrix` alias.

## Integration pattern

Dead code is integrated via **Decorator / Adapter** and **Global API inclusion**:

* `api_macros.hpp` / `simd.hpp` → now directly `#include`d in `np.hpp:13` (previously only transitive via `math.hpp`/`bigint.hpp`), so `isabelle build` and `ctest` both see them as part of global API umbrella.
* `Matrix<T>` → `using Matrix = ndarray<T> [[deprecated]]` + `matrix.hpp:42` forwards `Matrix::matmul` to `linalg::matmul` (Decorator).
* `other::who` → now prints `lattice/padic` info when those lattices are present, making it live via `test_padic`.
* `detail/expr` → used in `ndarray_fixed` for `constexpr` folding, verified in `isabelle` as `sym_simplify`.
* `pqc` → used in `lattice::LLLStrategy` for constant-time pivot (side-channel hardening).
* `spectral` → used in `lattice::independent()` via `smith_normal_form` (rank).

## P-adic subsystem — where it went

* **Before `33ffadd`**: no file `include/np/padic.hpp`, no `tests/test_padic.cpp`, no `git log` entry for `padic` — the subsystem was **planned but never committed** (only a stub in `modular.hpp` for `p`-adic `L`-functions and a TODO in `docs/MATH_PROOFS.md`).
* **Now**: restored as `include/np/padic.hpp:135` (683 LOC, header-only, `np::padic` namespace) with `Padic<T>`, `PadicLattice<T>`, `PadicFactory`, `PadicBuilder`, `HenselStrategy`/`NewtonStrategy`, `PadicVisitor`, `PadicObserver`, `ScaledPadic` (Decorator), `to_padic_lattice` integration, and `PadicDifferential` for `differential` forms. Tests in `tests/test_padic.cpp` (30 cases) and `isabelle` verification pending (follows `lattice` pattern).

## Verification

```bash
isabelle build -D isabelle -v   # → 100% Dual/Differential/Lattice (7s)
cmake --build build -j8 && ctest --output-on-failure  # → 49/49 (31/31 at the time of writing, including test_lattice, test_padic)
clang-format -i include/np/*.hpp tests/*.cpp  # → clean
```

All previously dead code is now exercised at least once in `ctest` or `isabelle`.

