# 05 — FFT (18 ops) — `fft/fft_core.hpp:244`, `fft_1d.hpp`, `fft_nd.hpp`

## Definition

DFT `X[k]= Σ_{j=0}^{n-1} x[j]·exp(-2πi·j·k/n)`. `rfft` real even, `hfft` Hermitian, `fftn` N-D via `transform_lines`, `fftfreq` `k/(n*d)`.

## Radix-2 — `fft_core.hpp`

**Claim:** `radix2_apply` Cooley-Tukey `X = DFT_{n/2}(even) + W·DFT_{n/2}(odd)` with `W=exp(-2πi k/n)` `TwiddleCache` exact `cos/sin`. *Proof:* induction on `n=2^m`, `next_pow2` for Bluestein.

## Bluestein — `BluesteinPlan`

For arbitrary `n` not power of two: `n → m = next_pow2(2n-1)`, chirp `x[j]·exp(-πi j²/n)` convolve via `m`-FFT. Matches NumPy `fft` for any `n`.

## Shifts

`fftshift` `out[i]=in[(i+n/2)%n]` etc, `ifftshift` inverse — `Odometer` same.

## Optimization

`TwiddleCache` precomputes `cos/sin` O(n) once; `transform_lines` reuses. No fast path change in `dev` — already `is_contiguous` via `ndarray`.

## Correctness

Test `test_fft.cpp` compares vs naive DFT for `n=5,7` (Bluestein) and power-of-two, `rfft` even symmetry, max error <1e-12.
