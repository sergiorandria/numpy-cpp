# 00 — Overview: Array Model, Notation, Lemma 0.3

## 1. Notation

Let $a$ be an `ndarray<T>` with:

- shape $\mathrm{shape}(a) = (s_0, \dots, s_{k-1})$, so $n = |a| = \prod_{j=0}^{k-1} s_j$ (`_numel()`),
- a flat backing store $\mathrm{data} : \mathrm{Vec}\langle T\rangle$,
- strides $\mathrm{str}(a) = (\sigma_0, \dots, \sigma_{k-1})$ (in elements, not bytes),
- an element offset $\mathrm{off}(a) \in \mathbb{N}$.

Element access is affine in the multi-index:

$$
a[i_0, \dots, i_{k-1}] \;=\; \mathrm{data}\Big[\mathrm{off}(a) + \sum_{j=0}^{k-1} i_j\,\sigma_j\Big].
$$

The canonical **C-order (row-major)** strides for a shape $s$ are

$$
\sigma^C_j(s) = \prod_{t=j+1}^{k-1} s_t, \qquad \sigma^C_{k-1}(s) = 1,
$$

computed by `_c_strides(shape)`.

For a linear index $i \in [0, n)$, the **mixed-radix expansion** into coordinates is

$$
\mathrm{coord}_j(i) = \left\lfloor \frac{i}{\sigma^C_j(s)} \right\rfloor \bmod s_j, \qquad j = 0, \dots, k-1,
$$

which is exactly how `flat(i)` (via `_flat_logical`) turns a linear index into an element.

## 2. Lemma 0.3 — Contiguous flat equivalence

**Definition (Contiguous), `ndarray.hpp:3116`.**

$$
\mathrm{is\_contiguous}(a) \;\Longleftrightarrow\; \mathrm{off}(a) = 0 \;\wedge\; \mathrm{str}(a) = \sigma^C(\mathrm{shape}(a)) \;\wedge\; |\mathrm{data}| \ge n.
$$

The `dev` implementation computes this via a single reverse pass (`exp` accumulator, no `_c_strides` allocation) and marks the mismatch/`offset≠0` branches `[[unlikely]]` — a performance hint, not a change to the predicate.

**Definition (`_flat_logical`, `ndarray.hpp:3479`).**

$$
\mathrm{flat}(i) =
\begin{cases}
\mathrm{off}(a) & \text{if } \mathrm{shape}(a) = () \text{ or } i = 0 \\[4pt]
\mathrm{off}(a) + i & \text{if } \mathrm{is\_contiguous}(a) \;[[\text{likely}]] \\[4pt]
\mathrm{off}(a) + \displaystyle\sum_{j} \mathrm{coord}_j(i)\cdot \sigma_j & \text{otherwise}
\end{cases}
$$

The old implementation always took the third branch through an `Odometer` (materializing a coordinate vector on the heap per call); `dev` short-circuits to `offset + i` when contiguous.

**Lemma 0.3.** For contiguous $a$ and any linear index $i \in [0, n)$,

$$
\mathrm{data}[\mathrm{flat}(i)] \;=\; \mathrm{data}[\mathrm{off}(a) + i].
$$

*Proof.* Since $a$ is contiguous, $\sigma_j = \sigma^C_j(s)$ for every $j$. The mixed-radix expansion of $i$ against C-order strides is exactly the standard base-$(s_0,\dots,s_{k-1})$ positional representation, so

$$
i = \sum_{j=0}^{k-1} \mathrm{coord}_j(i)\cdot \sigma^C_j(s).
$$

Substituting into the non-contiguous branch's formula gives

$$
\mathrm{off}(a) + \sum_j \mathrm{coord}_j(i)\cdot\sigma_j \;=\; \mathrm{off}(a) + \sum_j \mathrm{coord}_j(i)\cdot\sigma^C_j(s) \;=\; \mathrm{off}(a) + i,
$$

which is precisely what the contiguous fast branch returns directly, without computing coordinates. The two branches therefore agree on every $i$, and both equal $\mathrm{off}(a)+i$. $\blacksquare$

**Corollary (fast-path correctness).** Any loop of the shape

```cpp
if (is_contiguous()) [[likely]] {
    const T* p = data.data();
    for (std::size_t i = 0; i < n; ++i) { /* use p[i] */ }
} else {
    Odometer od(shape);
    while (!od.done()) { /* use data[_flat(od.idx())] */ od.advance(); }
}
```

produces the identical sequence of `data[flat(i)]` values as the slow, always-`Odometer` version, for every `i`. Hence any `np::` routine that delegates iteration to `_for_each_logical` — `sum` (`ndarray.hpp:3680`), `mean`, `prod`, `all`, `any`, and by extension every reduction/ufunc built on top of it — is correct under the micro-optimization. This is the **single proof obligation** discharged once here and reused by every contiguous fast path in `01`–`08`.

## 3. What Lemma 0.3 does *not* need to prove

Three classes of `dev` change are correctness-irrelevant by construction, and are noted rather than re-proven per site:

1. **`[[likely]]` / `[[unlikely]]`** — branch-prediction hints only; they do not change control flow or values.
2. **`reserve(n)` / block-size constants (e.g. `BLOCK=32` in `dot`)** — affect allocation count/cache behavior only, never the values written.
3. **`__restrict`-qualified pointers** — an alias-freedom promise to the optimizer; changes only what the compiler is permitted to assume, not what the code computes. It is the programmer's obligation (verified here per-site) that the promise is actually true — e.g. `dot`'s `a`, `b`, and output buffers must not alias, which holds because the output is freshly allocated.

Every subsequent proof in this series either reduces to Lemma 0.3, or is a self-contained algorithmic argument (Cooley–Tukey induction for FFT, blocked-sum associativity for `dot`, Chase-Lev linearizability for the thread pool, etc.).
