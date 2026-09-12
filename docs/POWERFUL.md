# Powerful workstation + GPU — tuning guide

This guide is for **very powerful computer + graphics card** (e.g. i7-9750H 12-thread, 32 GiB, GTX 1650 4 GiB, or RTX 4090 + 64-core Threadripper).

## Quick start

```bash
cmake --preset powerful          # AVX2+GPU+OpenMP+Threading+LTO+native
cmake --build --preset powerful -j$(nproc)
ctest --preset powerful
./build/tests/bench_hardware      # 64 vs 512/1024 GEMM CPU vs GPU
./build/examples/powerful_demo
```

`powerful` enables: `-mavx2 -mfma`, `-fopenmp`, `-march=native`, `-flto`, `NP_ENABLE_GPU`, `NP_USE_THREADING`, `NP_ENABLE_O3`.

For `fast-math` (may affect NaN/inf): `cmake --preset powerful-fastmath`.

Manual:

```bash
cmake -S . -B build -DNP_ENABLE_POWERFUL=ON
# or granular:
cmake -S . -B build -DNP_ENABLE_AVX2=ON -DNP_ENABLE_GPU=ON -DNP_ENABLE_OPENMP=ON -DNP_USE_THREADING=ON -DNP_ENABLE_LTO=ON -DNP_ENABLE_NATIVE=ON
```

## What is optimized

| Layer | Powerful optimization | Fallback |
|-------|----------------------|----------|
| **linalg GEMM** | `gpu::try_matmul` (OpenMP target / CUDA driver `dlopen`) for `M*N*K > tune::gpu_threshold()` (1M → 4M on 32+ threads), else `gpu::cpu_matmul` blocked `tune::optimal_block` (128 f32 / 96 f64 for 12 MiB L3, AVX2 `FMA` 8-wide, `madvise HUGEPAGE`) | ThreadPool `parallel_for` >4096, scalar triple loop |
| **GPU abstraction** `gpu.hpp` | `dlopen libcuda.so.1` `cuInit` + `omp_get_num_devices()` probe, `try_matmul` OpenMP `target teams distribute parallel for collapse(2)`, pinned `cudaMallocHost` / `aligned_alloc 64` | CPU blocked |
| **Accelerator** | `GPUAccelerator` → `gpu::matmul`, `AutoAccelerator` micro-benchmarks 128² | CPU |
| **Tensor** | `GpuFp32Backend` → `gpu::try_matmul` (>1M), `CpuBlockedBackend` → `gpu::cpu_matmul` | `linalg::matmul` |
| **Memory** | `GpuArray`/`PinnedArray` (`madvise HUGEPAGE`), `migrate_to_device/pinned`, `HBMArray` | `Host` |
| **Tune** `powerful.hpp` | `l3_cache_bytes()` via `sysconf`, `optimal_block_f32()` (`sqrt(L3/24)`), `gpu_threshold_flops()` (1M/2M/4M by threads) | static 32 |

## Verify

```bash
nvidia-smi  # driver 580+, CUDA 13
lscpu | grep -E "cache|AVX"
./build/tests/bench_hardware  # expect GPU 64 0.05ms < CPU 0.19ms, 1024 CPU~60ms GPU~63ms (offload overhead)
```

No CUDA toolkit required — driver `dlopen` fallback ensures header-only build on CI. With toolkit (`-DNP_ENABLE_CUDA=ON`) links `CUDA::cudart`/`cublas`.

## When GPU helps

GTX 1650: ~2.9 TFLOPS FP32, PCIe 8 GB/s → transfer dominates <256. Threshold `tune::gpu_threshold` ensures GPU only for >1M FLOPs. RTX 4090: larger threshold, use `AutoAccelerator` which benchmarks.

## Troubleshooting

- `omp_get_num_devices()==0` → install `libgomp-plugin-nvptx` or set `NP_ENABLE_GPU=OFF`
- `madvise` only on Linux; macOS/Windows falls back to `aligned_alloc`
- `-march=native` may not be portable; distribute with `-mavx2 -mfma` instead
