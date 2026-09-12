# Architecture — dev

## 1. Two engines

### Dynamic `ndarray<T>` — `ndarray.hpp:516,6485 LOC`

```cpp
template <typename T>
class ndarray {
  std::shared_ptr<std::vector<T>> data_; // heap, shared for zero-copy views
  std::vector<int> shape;                // e.g. {3,4}
  std::vector<std::size_t> strides;      // C-order: {4,1} for {3,4}
  std::size_t offset = 0;                // view offset
  dtype type;
};
```

* **Views** share `data_` + `offset`/`strides` — `reshape`, `transpose`, `atleast_2d` (`manipulation.hpp:1606` now `return arr` not `copy()`) are O(1).
* **Contiguity** is hot: `is_contiguous:3116` now `exp` loop without ` _c_strides` alloc; `is_f_contiguous:3126` similar. Used as `[[likely]]` guard in `copyto`, `is_busday`, `dot`, `norm`, `nditer`.
* **Indexing** `arr(i,j)` → `data_[offset + i*strides[0]+j*strides[1]]` (`at:3315` `__restrict` + `bool` proxy). `_flat:3472` → `detail::flat_index`; `_flat_logical:3479` fast `offset+i` when contiguous else `flat+=coord*strides` (no `_shape_u` alloc).
* **Iteration** `_for_each_logical:3539` contiguous `const T* __restrict p=data_->data()` loop vs `Odometer` fallback; `bool` specialization via `for (auto& v: *data_)`.

### Fixed `ndarrayf<T, Extents...>` — `ndarray_fixed.hpp:1411 LOC`

* `std::array<T, (Extents*...)>` stack, `static_assert(Extents>0)`, `constexpr` `sum/mean` via `detail/expr.hpp` lazy trees. No `shared_ptr`; views copy.

### Bridge

* `dtype.hpp:621` `timedelta64` vs `datetime.hpp:516` `timedelta64 = chrono::days` — kept separate (`np::datetime::timedelta64`).
* `simd.hpp` autodetects `__AVX512F__/__AVX2__/__SSE4_2__/__ARM_NEON` + new `NP_SIMD_WASM/RVV:983`.

## 2. Umbrella and modules

`np.hpp:13` (58 LOC) includes 28 headers in dependency order (now fully integrated; `random.hpp:64` + `concatenate.hpp:106` after `ADL` fix via `is_ndarrayf`/`NP_CONCATENATE_HPP` guards).

```
np → ndarray → creation → manipulation → linalg → fft → ...
   → logic (isin) → datetime (weekday) → random (_fill) → testing
```

* `creation.hpp:65 NP_API` — `arange`, `linspace`, `asanyarray:946`, `fromiter:1023`, `mgrid:1499`.
* `manipulation.hpp:64 NP_API` — `copyto:1980` now `memcpy`, `split:703` `reserve`, `block:1832` 1-D.
* `linalg.hpp:2669` — `dot` blocked, `norm:1630` contiguous, `EinsumPath:3845`.
* `indexing.hpp:576` — `nditer`/`flatiter` cached `ptr_/contig_`.
* `logic.hpp:590` — `isin` hash/sort dual.
* `datetime.hpp:99` — `_weekday` days%7, `busday_*` week arithmetic.

## 3. Memory and threads

* **SIMD** `simd.hpp:1232` — `add_f32_wasm:983` `v128_load`, `add_f32_rvv` `__riscv_vsetvl`. Scalar tail always.
* **Threadpool** `threadpool.hpp:236` — `__np_deque_lockfree` ring `cap_=1024`, `mask_=cap_-1`, `top_/bottom_` atomics, `grow` power-of-two. `ThreadPool::global().parallel_for` used in `linalg::dot` >4096, `statistics`, `sorting`. Opt-in `cmake -DNP_USE_THREADING=ON`.
* **Views** `shared_ptr` gives zero-copy `transpose`/`reshape`; `is_contiguous()` decides `memcpy` vs `Odometer`.

## 4. Error handling

`exceptions.hpp:251` hierarchy (`AxisError`, `LinAlgError` at `linalg.hpp:165`), `err.hpp:403` thread-local `ErrState` (`seterr/geterr`).

## 5. Testing

`tests/test_util.hpp:48` — `test::check`, `approx`, `approx_c`. `tests/CMakeLists.txt` 29 targets + `bench_math` (AVX, not in ctest). All 29 pass on `dev` (incl. `test_higher` for cohomology/bundle/persistent/spectral).

## 6. File:line map (hot)

| Hot path | File:line | Change in dev |
|----------|-----------|---------------|
| `is_contiguous` | `ndarray.hpp:3116` | no alloc |
| `_flat_logical` | `ndarray.hpp:3479` | `offset+i` |
| `_for_each_logical` | `ndarray.hpp:3539` | `__restrict` |
| `copyto` | `manipulation.hpp:1980` | `memcpy` |
| `dot` | `linalg.hpp:2669` | BLOCK=32 |
| `is_busday` | `datetime.hpp:192` | `__restrict` |
| `isin` | `logic.hpp:590` | hash>64 |

See `PERFORMANCE.md` for bench deltas.
