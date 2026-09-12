# Math Proofs — Index (`dev`)

This folder is the **per-module expansion** of [`../MATH_PROOFS.md`](../MATH_PROOFS.md) (the master document). Each file proves a block of the 712 `np::` routines, including the `dev`-branch micro-optimizations layered on top of them.

**Method.** Every `np::` routine is checked against the NumPy formula documented at `numpy-reference/reference/generated/numpy.<func>.html` (mirrored 1:1 via a `Reference:` Doxygen tag in the source). Each proof is a *pointwise equality* argument: fast path values equal slow path values equal the NumPy spec, generally by reduction to **Lemma 0.3** (contiguous fast-path equivalence, see [`00_overview.md`](00_overview.md)).

| File | Module(s) | Routines | Hot `file:line` |
|---|---|---|---|
| [00_overview](00_overview.md) | Preliminaries, notation, Lemma 0.3 | — | `ndarray.hpp:3116,3479` |
| [01_ndarray](01_ndarray.md) | Core `ndarray`: `is_contiguous`, `_flat_logical`, `_for_each_logical`, `at` | 1 engine | `ndarray.hpp:3116,3479,3315,3539` |
| [02_creation](02_creation.md) | `creation`: `arange`, `linspace`, `eye`, … | 42 | `creation.hpp:65` |
| [03_manipulation](03_manipulation.md) | `manipulation`: `copyto`, `split`, `atleast_*`, … | 45 | `manipulation.hpp:1980,728` |
| [04_linalg](04_linalg.md) | `linalg`: `dot`/`matmul`, `norm`, `cross`, `einsum` | 44 | `linalg.hpp:2669,1630,3322,3837` |
| [05_fft](05_fft.md) | `fft`: radix-2, Bluestein, shifts | 18 | `fft/fft_core.hpp:244` |
| [06_random](06_random.md) | `random`: `Generator`, `_fill_distribution`, `shuffle` | 50 | `random.hpp:64,1277,161` |
| [07_statistics_sorting](07_statistics_sorting.md) | `statistics` + `sorting` + `logic`/set ops | 34 | `logic.hpp:590,669`, `indexing.hpp:576` |
| [08_optimizations](08_optimizations.md) | Synthesis of all micro-opts: hashing, datetime week arithmetic, WASM/RVV SIMD, Chase-Lev deque | — | `simd.hpp:983`, `threadpool.hpp:236` |

**How to read a proof.** Each entry states a **Claim** (what the routine computes, in LaTeX matching the NumPy spec), the **C++ realization** (both the reference/slow path and the `dev` fast path), an **Equivalence argument** (fast path = slow path = spec), and **Complexity**.

**Verification.**

```bash
cmake --build build -j8 && ctest --test-dir build --output-on-failure   # 22/22 after every dev micro-opt (f7b2653..12115ad)
grep -n "is_contiguous" include/np/ndarray.hpp
```

See [`../MATH_PROOFS.md`](../MATH_PROOFS.md) for the full 26-group master proof and [`../PERFORMANCE.md`](../PERFORMANCE.md) for benchmark deltas.

**Rendering note.** These files use standard `$inline$` / `$$display$$` LaTeX delimiters. GitHub renders this natively; in VS Code, install the **Markdown+Math** extension (`goessner.mdmath`) to get the same rendering in the built-in preview — the default Markdown preview does not evaluate LaTeX.
