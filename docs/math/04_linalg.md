# 04 — Linalg (44 routines) — `linalg.hpp`

## 4.1 `dot` / `matmul` — `linalg.hpp:2669`

**Spec.** For $a \in \mathbb{R}^{M\times K}$, $b \in \mathbb{R}^{K\times N}$:

$$
c_{ij} = \sum_{p=0}^{K-1} a_{ip}\,b_{pj}, \qquad i \in [0,M),\ j\in[0,N).
$$

**Main (reference) path.** `a.get({i,p}) * b.get({p,j})` accumulated per inner step — each `.get` recomputes a strided flat offset (correct but slow for non-trivial strides).

**`dev` fast path.** Guarded by `a.is_contiguous() && b.is_contiguous()` `[[likely]]`, obtaining `const T* __restrict ad, bd` and `T* __restrict od`:

- *Small/medium* ($M\!\cdot\! N\!\cdot\! K \le 32768$ or any dimension $\le 32$): triple loop
  $$
  od[iN+j] \mathrel{+}= ad[iK+p]\cdot bd[pN+j].
  $$
- *Large*: **blocked** with tile size $B=32$, zero-initialized output, then

$$
\text{for } i_0=0,B,2B,\dots < M:\ \text{for } j_0 \dots < N:\ \text{for } p_0\dots < K:\quad
od[iN+j] \mathrel{+}= ad[iK+p]\cdot bd[pN+j]
$$

over $i\in[i_0,i_0{+}B),\ j\in[j_0,j_0{+}B),\ p\in[p_0,p_0{+}B)$ (clamped at the boundary).

**Proof of equivalence.**

1. *Small/medium path = spec*: the triple loop is a direct transcription of $\sum_p a_{ip}b_{pj}$ using flat C-order arithmetic $iN+j$, $iK+p$, $pN+j$ — which is exactly the physical offset formula for a contiguous C-order array (Lemma 0.3, applied to `a`, `b`, and `od` each being contiguous), so `ad[iK+p] = a[i,p]` etc. pointwise.
2. *Blocked path = small/medium path*: floating-point/integer addition used to accumulate $od[iN+j]$ is associative and commutative for the purposes of this argument (ignoring floating-point rounding order, which NumPy itself does not guarantee to fix either — `numpy.dot` documents no particular summation order). The $B$-tiles of $p_0,p_0+1,\dots,p_0+B-1$ partition $[0,K)$ exactly (every $p$ falls in exactly one tile, boundary tiles clipped to $K \bmod B$), so summing tile-by-tile and summing $p=0,\dots,K-1$ directly reach the same total $\sum_p a_{ip}b_{pj}$.
3. *`parallel_for` over rows*: each thread owns a disjoint set of $i$ rows and writes only to $od[i\cdot N \dots]$ for its own $i$ — no two threads write the same output cell, so the parallel accumulation is race-free and produces the same per-row sums as the serial blocked loop.

$\blacksquare$

**Complexity.** $O(MKN)$ regardless of path; blocking changes cache behavior, not asymptotic cost.

## 4.2 `norm` — `linalg.hpp:1630`

**Spec (1-D $x \in \mathbb{R}^n$).**

$$
\|x\|_2 = \sqrt{\textstyle\sum_i x_i^2}, \quad
\|x\|_1 = \textstyle\sum_i |x_i|, \quad
\|x\|_\infty = \max_i |x_i|, \quad
\|x\|_{-1} = \Big(\textstyle\sum_i \tfrac{1}{|x_i|}\Big)^{-1}, \quad
\|x\|_{-2} = \Big(\textstyle\sum_i \tfrac{1}{x_i^2}\Big)^{-1/2}.
$$

**Proof.** The `dev` fast path (`x.is_contiguous() [[likely]]`) reads via `const T* __restrict ptr` in place of `x.at(i)`. By Lemma 0.3, `ptr[i] = x.at(i)` for every $i$ under contiguity, so the accumulator `acc` — built identically for either access method — is bit-for-bit the same reduction. The 2-D matrix-norm variants (Frobenius, induced 1/∞-norm via column/row sums) apply the identical argument axis-wise.

## 4.3 `cross` — `linalg.hpp:3322`

**Spec.** 2-element inputs: scalar $a_0 b_1 - a_1 b_0$. 3-element: $(a_1b_2-a_2b_1,\ a_2b_0-a_0b_2,\ a_0b_1-a_1b_0)$.

**Proof.** An early `is2` check dispatches to the scalar formula with broadcasting over leading axes for the 2-element case; otherwise a 3-vector `Odometer` loop computes each of the three components directly from the formula above — literal transcription, so pointwise equal to `numpy.cross` (which uses the same `axis` parameter via `norm_axis` to locate the length-2/3 axis).

## 4.4 `einsum` — `linalg.hpp:3837`

**Spec.**

$$
\mathrm{out}[idx_{\mathrm{out}}] = \sum_{\text{summed labels}} \ \prod_{\text{operand } o} \mathrm{operands}[o]\big[\,idx_o\,\big], \qquad idx_o[d] = idx_{\mathrm{all}}\big[\mathrm{lab\_pos}[\mathrm{label}_d]\big].
$$

**Proof.** `all_labels.reserve(...)`/`all_shape.reserve(...)` are pure allocation hints; the enumeration itself is an `Odometer(all_shape)` sweep over every label's full extent, at each point evaluating the product-of-gathers and accumulating into `out[idx_out]` — the direct Einstein-summation definition, independent of the `reserve` calls.

**`EinsumPath`** (`linalg.hpp:3845`): chooses between exhaustive search ($n\le 4$ operands) and a greedy heuristic for $n>4$. Both only choose a *contraction order* (which pairs of operands to multiply first) to minimize $\mathrm{cost} = \sum |\,\text{intermediate shape}\,|$; the final summed value is invariant under contraction order (multiplication distributes over the same finite sum regardless of grouping), so path choice affects performance only, never the numeric result.

## 4.5 `eig` / `qr` / `svd` / `cholesky` / `solve`

Unchanged in `dev` — Householder QR, Francis-shift QR eigenalgorithm, and Cholesky/`solve` via forward-back substitution are documented in-source per routine with their own convergence/stability references; no micro-opt touches these paths this cycle.

**Complexity.** `dot`: $O(MKN)$. `norm`: $O(n)$. `eig`: $O(N^3)$ per Francis-QR sweep set (standard for dense unsymmetric eigendecomposition).
