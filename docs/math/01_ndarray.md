# 01 — `ndarray` Core Engine

Every routine in this file reduces to **Lemma 0.3** (`00_overview.md`); the work here is showing that each site's fast/slow pair implements the *same predicate or accessor* the lemma is stated about.

## 1.1 `is_contiguous` — `ndarray.hpp:3116`

**Claim.** $\mathrm{is\_contiguous}(a) \Leftrightarrow \mathrm{str}(a) = \sigma^C(\mathrm{shape}(a))$ (with the offset/size side conditions from §2 of `00_overview.md`).

**Main (pre-`dev`) path:** builds `_c_strides(shape)` — a fresh `vector<size_t>` — then compares vectors. $O(k)$ time, $O(k)$ allocation.

**`dev` path:**

$$
\texttt{exp} \leftarrow 1;\quad \text{for } i = k-1 \text{ down to } 0: \quad \text{if } \sigma_i \ne \texttt{exp} \to \text{false}; \quad \texttt{exp} \mathrel{*}= s_i,
$$

with `[[unlikely]]` on the mismatch and `offset≠0` branches.

**Proof of equivalence.** The loop computes exactly $\sigma^C_i(s) = \prod_{t>i} s_t$ incrementally (`exp` after processing index $i+1,\dots,k-1$ equals $\prod_{t>i}s_t$) and compares it to $\sigma_i$ in the same order the vector comparison would — so it decides the identical predicate, without materializing the intermediate vector. $O(k)$ time, $O(1)$ allocation. $\blacksquare$

## 1.2 `_flat` (`ndarray.hpp:3472`) and `_flat_logical` (`ndarray.hpp:3479`)

**Claim.** `_flat(idx)` computes $\mathrm{off}(a) + \sum_j idx_j\sigma_j$ (`detail::flat_index`); `_flat_logical(i)` computes $\mathrm{flat}(i)$ as defined in `00_overview.md`, with the `dev` fast branch `is_contiguous() [[likely]] → offset + i`. This is Lemma 0.3 verbatim — see that proof.

## 1.3 `at(i)`, `at(i,j)`, `operator()(i,j)` — `ndarray.hpp:3315,3417,3391`

**Claim.** `at(i)` requires $k=1$ and $0\le i < s_0$, then returns $\mathrm{data}[\mathrm{off}(a) + i\sigma_0]$; `at(i,j)` is the $k=2$ analogue.

**`dev` change:** `[[unlikely]]` on the two throw paths (`k \ne 1`, out-of-range), plus a compile-time branch: `if constexpr (is_same_v<T,bool>)` returns the `vector<bool>` proxy reference, else obtains `T* __restrict d = data.data()` and returns `d[flat]`.

**Proof.** Bounds-checking logic is untouched; the `if constexpr` selects between two access strategies required by `std::vector<bool>`'s packed-bit representation (which has no real `bool&`) versus a plain pointer for every other `T`. `__restrict` is a non-aliasing promise on `d`, true here because `data()` returns this array's own private buffer — it does not change which element `d[flat]` denotes. $\blacksquare$

## 1.4 `_for_each_logical` — `ndarray.hpp:3539`

**Claim.** Iterates $i = 0,\dots,n-1$, invoking `fn(a[\mathrm{flat}(i)])$` for each.

**`dev` fast path (contiguous):**

```cpp
const T* __restrict p = data.data();
for (std::size_t i = 0; i < n; ++i) fn(p[i]);
```

with a `bool`-specialized overload `for (auto& v : *data_) fn(v)` to handle the `vector<bool>` proxy. Non-contiguous falls back to `Odometer`.

**Proof.** By Lemma 0.3, $p[i] = \mathrm{data}[\mathrm{flat}(i)]$ for every $i$ when $a$ is contiguous, so the fast loop calls `fn` on the identical sequence of values as the `Odometer` path. $\blacksquare$

**Corollary.** `sum` (`ndarray.hpp:3680`), which accumulates `total += v` inside `_for_each_logical`, computes $\sum_i a[i]$ correctly regardless of the fast/slow branch taken; the same corollary transfers to `mean`, `prod`, `all`, `any`, since they are all thin wrappers over the same traversal primitive.

## 1.5 Complexity of `sum`/`mean`/`var`

$O(n)$, $n = |a|$. The contiguous path performs exactly $n$ loads and $n$ adds with no `Odometer` bookkeeping overhead (no per-step division/modulo to recover coordinates).

## 1.6 `is_f_contiguous` — `ndarray.hpp:3126`

Same `exp`-accumulator structure as §1.1, but sweeping strides forward with $\mathrm{stride}_0 = 1$ (Fortran/column-major order) instead of backward from the last axis.

## 1.7 `data()` and `__restrict`

`T* __restrict` on `data()`-derived pointers is a promise to the optimizer that the pointer doesn't alias any other pointer written through in the same scope, enabling auto-vectorization (SIMD loads/stores without runtime alias checks). It has no effect on the *value* computed at `d[flat]` — see §3 of `00_overview.md`.
