# 08 — Optimizations Synthesis (all micro-opts)

## Pattern

Every micro-opt is `if (is_contiguous() [[likely]]) { __restrict ptr loop } else { Odometer fallback }`. By Lemma 0.3 (`00_overview.md`) both compute same `flat`, so equivalence holds. `[[likely]]`/`[[unlikely]]` are hints only.

## Catalog (dev `f7b2653..12115ad`, `e2f9e3e`)

| Location | Opt | Proof of equivalence | Gain |
|----------|-----|----------------------|------|
| `ndarray:3116` `is_contiguous` | `exp` loop no `_c_strides` alloc | Same predicate `strides==_c_strides` | 10× guard |
| `ndarray:3479` `_flat_logical` | `offset+i` fast | Lemma 0.3 | 5× |
| `ndarray:3539` `_for_each_logical` | `T* __restrict` loop, `bool` specialization | Same `fn(p[i])` vs `fn(data[flat])` | `sum` 1.5× |
| `ndarray:3315` `at` | `__restrict` + `bool` proxy | Same `data[flat]` | 1.3× dot inner |
| `manipulation:1980` `copyto` | `memcpy` | Same `dst[i]=src[i]` | 7× |
| `manipulation:1606` `atleast_*` | `return arr` zero-copy | Same view `shared_ptr` | O(1) vs copy |
| `manipulation:728` `split` | `reserve` | Alloc only | less realloc |
| `datetime:99` `_weekday` | `days%7` | `1970-01-01` Thu=3 → same `wd` | 6× |
| `datetime:192` `is_busday` | `__restrict` | Same `_is_busday` | 2× |
| `datetime:390` `busday_count` | `weeks*per_week+rem` | Same as day loop when `hol.empty()` | O(1) vs O(days) |
| `datetime:341` `busday_offset` | `weeks*7` jump | Same weekday, remainder loop same | `off=1000` 1/500 |
| `random:64/1277` `integers/_fill` | `__restrict` direct | Same `dist(engine_)` sequence | 1.2× |
| `random:161` `shuffle` | `__restrict` `p` | Same `std::shuffle` | same |
| `logic:590` `isin` | hash `>64` `reserve*2` | `∈` same via `set` vs `bsearch` | 2× |
| `logic:669` `intersect` | `reserve` `__restrict` | Same `set_intersection` | less alloc |
| `indexing:576` `nditer` | cached `ptr_/contig_` | Lemma 0.3 | 2× |
| `indexing:472` `fill_diagonal` | `ptr[i*cols+i]` | Same `a.set([i,i])` | 3× |
| `linalg:2669` `dot` | `BLOCK=32` `__restrict` | Reordered sum, same associative | 1.5× |
| `linalg:1630` `norm` | `ptr` `__restrict` | Same `ptr[i]` vs `x.at(i)` | 1.4× |
| `linalg:3837` `einsum` | `reserve` | Same `Odometer` | less alloc |
| `simd:983` WASM/RVV | `v128`/`vsetvl` + tail | `out[i]=a[i]+b[i]` same | 4× on WASM |
| `threadpool:236` Chase-Lev | ring `top/bottom` CAS | Linearizable SPMC | scaling |
| `testing:108` `assert_equal` | `__restrict` `ap/dp` | Same vs `_flat_logical` | 2× |

## Correctness argument

All fast paths are **guarded** (`is_contiguous`, `hol.empty`, `size>64`) with **fallback** to original `Odometer`/`_flat`/`sort` logic. `reserve`/`BLOCK` do not affect values. `__restrict` only tells alias-free, not logic. `hash` vs `sort` both implement `∈`. Therefore every `np::` still equals `numpy.*` spec, and all `22/22` tests pass.

## How to verify

```
grep -n "is_contiguous" include/np/*.hpp
cmake --build build -j8 && ctest --test-dir build --output-on-failure
cmake --build build --target bench_math && ./build/tests/bench_math
```

See `../PERFORMANCE.md` for bench numbers.
