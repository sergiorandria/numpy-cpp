# 03 — Manipulation (45 routines) — `manipulation.hpp`

## 3.1 `copyto` — `manipulation.hpp:1980`

**Spec.**

$$
\mathrm{dst}[idx] \leftarrow \mathrm{src}_{\mathrm{bcast}}[idx] \quad \text{for every } idx, \text{ subject to } \mathrm{where}[idx] \text{ (or unconditionally if no mask)}.
$$

**Slow path.** An `Odometer` enumerates $\mathrm{dst.shape}$, computing each element's physical offset via `detail::flat_index` on both sides.

**`dev` fast path.**

```cpp
if (same_shape && dst.is_contiguous() && src.is_contiguous() && !where) [[likely]]
    memcpy(dst.data(), src.data(), n * sizeof(T));   // __restrict on both sides
```

with an elementwise masked fallback `if (mask[i]) ap[i] = vp[i]` when both operands are contiguous.

**Proof.** By Lemma 0.3, for a contiguous array the sequence $(\mathrm{data}[\mathrm{flat}(0)],\dots,\mathrm{data}[\mathrm{flat}(n-1)])$ equals $(\mathrm{data}[\mathrm{off}],\dots,\mathrm{data}[\mathrm{off}+n-1])$ — a contiguous physical run. `memcpy` over that run therefore copies exactly the same $n$ logical elements, in the same order, as the per-element `Odometer` loop. The masked variant is pointwise identical to the unmasked one restricted to `mask[i]=true` positions when both sides are contiguous (so `ap[i]`/`vp[i]` address the same logical elements as the general `where`-guarded loop). $\blacksquare$

**Complexity.** $O(n)$ either way; the fast path replaces $n$ divisions/modulos (coordinate recovery) with a single bulk memory copy.

## 3.2 `atleast_1d` / `atleast_2d` / `atleast_3d` — `manipulation.hpp:1606`

**Spec.** `atleast_2d(a)` $= a$ if $\mathrm{ndim}(a)\ge 2$, else $a$ reshaped to $(1, N)$ (analogously for 1-D/3-D).

**Proof.** The reshaped/passthrough result shares `data_` with the parent (a *view*, `shared_ptr`-aliased), differing only in `(shape, strides)` metadata — this is exactly `reshape`'s documented view semantics for a contiguous source, so `atleast_*` never copies when NumPy wouldn't either. $O(1)$ vs. the naive $O(n)$ copy it replaces.

## 3.3 `split` (`manipulation.hpp:703`), `array_split` (`:852`), `block` (`:1832`)

**Spec.** `split` cuts at $\mathrm{split\_points} = \{0\} \cup \mathrm{indices} \cup \{n\}$, producing slices $a[\mathrm{start}:\mathrm{end}]$ per adjacent pair, enumerated via `Odometer`.

**`dev` change:** `reserve(sections)` / `reserve(n+2)` ahead of the slicing loop — a pure allocation-count optimization; the `src_idx`/`dst_idx` index-mapping loops that actually produce slice contents are unchanged, so equivalence is immediate (`reserve` never affects logical contents, only avoids reallocation during `push_back`).

## 3.4 `pad`, `transpose`, `moveaxis`, `broadcast_to`

**Proof.** All enumerate `out_shape` via `Odometer`, mapping each output index to a source index using NumPy's broadcast/transpose/pad rules — structurally the same traversal NumPy performs internally. `pad`'s constant-fill mode `memcpy`s the interior region when the source is contiguous (same equivalence argument as §3.1); border cells are filled directly with the constant value, matching `numpy.pad(mode="constant")`.

## 3.5 `unique`

Delegates entirely to `logic::unique_all` (`logic.hpp:730`); proven in `07_statistics_sorting.md`.

## 3.6 `as_strided`, `sliding_window_view`

**Spec.** `as_strided` (`manipulation.hpp:2395`) constructs a new view sharing `data_` with arbitrary caller-supplied `(shape, strides)` — an intentionally unsafe escape hatch identical to `numpy.lib.stride_tricks.as_strided`. `sliding_window_view` is realized as an `Odometer`-driven copy over the windowed index space, matching `numpy.lib.stride_tricks.sliding_window_view` element-for-element.

**Complexity.** `copyto`: $O(n)$ `memcpy` vs. $O(n)$ `Odometer`. `split`: $O(n)$ total across all output slices.
