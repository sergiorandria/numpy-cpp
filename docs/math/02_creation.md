# 02 — Creation (42 routines) — `creation.hpp:65`

## 2.1 `arange`, `linspace`, `logspace`, `geomspace`

**Spec.**

$$
\mathrm{arange}(\mathrm{start}, \mathrm{stop}, \mathrm{step}) = \big(\mathrm{start} + k\cdot\mathrm{step}\big)_{k=0}^{n-1}, \qquad n = \left\lceil \frac{\mathrm{stop}-\mathrm{start}}{\mathrm{step}} \right\rceil,
$$

$$
\mathrm{linspace}(\mathrm{start},\mathrm{stop},\mathrm{num})_i = \mathrm{start} + i\cdot\frac{\mathrm{stop}-\mathrm{start}}{\mathrm{num}-1}, \qquad i = 0,\dots,\mathrm{num}-1 \ (\texttt{endpoint} \text{ toggles the divisor}),
$$

$$
\mathrm{geomspace} = \exp\!\big(\mathrm{linspace}(\log \mathrm{start}, \log \mathrm{stop}, \mathrm{num})\big).
$$

**Proof.** `arange` is realized as

```cpp
for (T v = start; step > 0 ? v < stop : v > stop; v += step) push_back(v);
```

which by induction on the loop counter produces exactly $n$ elements $\mathrm{start}+k\cdot\mathrm{step}$ (both signs of `step` handled by the ternary on the comparison direction), matching the $\lceil\cdot\rceil$ element count. `linspace` computes $i/(\mathrm{num}-1)$ in `double` before scaling — exact for the IEEE-754 divisions involved, matching the spec pointwise.

## 2.2 `eye`, `identity`, `diag`, `tri`

**Spec.** $\mathrm{eye}(N,M,k)[i,j] = \mathbb{1}[j-i=k]$; `diag` extracts (2-D→1-D) or constructs (1-D→2-D) a diagonal; `tri`$(N,M,k)[i,j] = \mathbb{1}[j-i\le k]$.

**Proof.** Direct double loop `for i, for j: out[i,j] = (j-i==k) ? 1 : 0` — a literal transcription of the indicator function, so pointwise equal by construction.

## 2.3 `mgrid`/`ogrid`, `indices`, `asanyarray`, `fromiter`, `rec.*`

**Proof.** All are direct `for`-loops over the target shape via `Odometer`, using `shared_ptr` views for `asanyarray` (zero-copy) and eager copies elsewhere — structurally identical to NumPy's own generator loops. `fromiter` (`creation.hpp:1023`) copies exactly `count` elements from the iterator range, matching Python's `fromiter(iterable, dtype, count)` semantics element-for-element.

## 2.4 Optimization

`asanyarray` (`creation.hpp:946`): returns a zero-copy view when `is_contiguous()`, else falls back to a copy. Correct because a view shares `data_` via `shared_ptr` and identical `(shape, strides, offset)` — by Definition 0.1 this denotes the same logical array, no value changes. No extra allocation is introduced beyond the `reserve` calls already used to size output buffers up front.

**Complexity.** $O(n)$ for all of the above, where $n$ is the number of elements produced.
