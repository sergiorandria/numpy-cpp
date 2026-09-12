# 07 — Statistics (19) + Sorting (15) + Logic/Set

## Statistics — `statistics.hpp:1955`, delegates to `ndarray::mean/var`

**Claim:** `mean` `Σ a[i]/n` via `_for_each_logical` fast `__restrict` (already proven). `var` `Σ (x-mean)²/(n-ddof)`, `std` `sqrt(var)`, `median` `sort` + middle, `quantile:2291` `method linear/inverted_cdf` via `sort` + `lerp`.

**Optimization:** Indirect via `ndarray` fast paths — already `is_contiguous` `__restrict`.

## Sorting — `sorting.hpp:426`, `ndarray: sort`

**Claim:** `sort` `std::sort` (or `parallel_for` when `NP_USE_THREADING` >4096) equals `numpy.sort` values (order for equal keys may differ, allowed). `argsort` `iota` + `sort` indices, `searchsorted` `lower_bound`, `nonzero` `vector per dim`.

**Optimization:** `__restrict` on `data()` loops not needed — `std::sort` is already optimized.

## Logic/Set — `logic.hpp:590`

**Claim:** `isin:590` `hash>64` vs `sort` both compute `∈`. `intersect1d:669` `sort+unique+set_intersection` equals `a∩b` sorted unique; `union` `set_union`, `setdiff` `set_difference`, `setxor` `symmetric_difference`.

**Optimization:** `isin` `empty [[unlikely]]`, `>64` `unordered_set reserve*2` O(1) vs `binary_search` O(log n) with `is_contiguous() [[likely]]` `__restrict`; `intersect` `empty [[unlikely]]` + `reserve(min)` + `__restrict dst`.

## Indexing — `indexing.hpp:576` (also part of sorting)

`nditer:576`/`flatiter:616` cached `ptr_/contig_` → `contig_ ? ptr_[pos_] : _flat_logical` proven via Lemma 0.3. `fill_diagonal:472` 2-D `ptr[i*cols+i]` vs `a.set`.

All O(n log n) for sort, O(n) for scan.
