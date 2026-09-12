# Examples — hardware-aware (neuromorphic, HBM, tensor, padic, quantum)

Build: `cmake -S . -B build && cmake --build build -j8 && ./build/examples/neuromorphic_snn`

## Neuromorphic SNN (`examples/neuromorphic_snn.cpp`)
```cpp
auto spikes = np::spike::encode_rate(img, 100, 100);
np::neuromorphic::LIFNeuron lif; lif.step(2.0);
np::neuromorphic::LifSimBackend sim(10.0, 1.0);
auto out = sim.process(ea); // real per-channel LIF simulation
```
Uses `np::event::EventArray` (COO, `shared_ptr`+`span`), `np::spike::encode_rate/temporal`, `LIF`/`Izhikevich`, standalone `STDP` primitive, `INeuromorphicBackend` Strategy (CPU pass-through harness / LIF-sim), `QuantizedEventArray` Decorator. Software simulation only — no Loihi/SpiNNaker hardware.

## HBM / Tensor (`examples/hbm_matmul.cpp`)
```cpp
auto ha = np::mem::migrate_to_hbm(a); // HBMArray
auto c = np::tensor::matmul_fp8(a,b,1.0f,1.0f); // simulated FP8 (quantize/dequantize around FP32)
auto acc = np::accelerator::AcceleratorFactory::gpu(); acc->matmul(a,b);
```
`np::mem::HBMArray`/`CXLArray` zero-copy `shared_ptr` alias, `np::tensor::GpuFp32Backend`/`CpuBlockedBackend` Strategy.

## p-adic Hensel (`examples/padic_hensel.cpp`)
```cpp
np::padic::Padic<int64_t> x0(7,3,6); // 3^2=2 mod7
auto root = np::padic::HenselStrategy<int64_t>(10).lift(x0,
  [](auto &x){ return Padic(x.p, x.value*x.value-2, x.prec); },
  [](auto &x){ return Padic(x.p, 2*x.value, x.prec); });
auto pl = np::padic::to_padic_lattice(np::lattice::LatticeFactory::cubic<int64_t>(2),7,10);
```
`Padic`/`PadicLattice`/`PadicDifferential` with `Hensel`/`Newton` Strategy, `PadicBuilder`, `Teichmuller` — verified in `isabelle/Padic_Verification.thy`.

## Quantum / Photonics / Analog (`examples/quantum_photonics.cpp`)
```cpp
auto s = np::quantum::QuantumFactory::plus_state(2); // 2^n StateVector
auto y = np::photonics::PhotonicsFactory::identity(2).apply(x); // MachZehnderMesh
np::analog::Crossbar cb(eye<float>(2)); cb.dot(xv); // ReRAM V=IR
```

All examples are header-only, `g++ -std=c++20 -I include examples/*.cpp -o /tmp/ex`.

See `isabelle/README.md` for proofs (`4/4` theories `100%`).

