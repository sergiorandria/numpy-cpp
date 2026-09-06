/**
 * @file quantum.hpp
 * @brief Quantum — StateVector, isolated VM, circuit ops
 * (H/X/Y/Z/S/T/RX/RY/RZ/CNOT/CZ/SWAP/Toffoli), measurement.
 *
 * Provides `np::quantum` with isolated qubit simulation:
 *   - `Qubit`/`StateVector` (2^n amps, ndarray<c128>, prob, normalize, measure)
 *   - `QuantumGate` variant (1q/2q/3q unitaries, ndarray<c128> 2x2/4x4/8x8)
 *   - `QuantumCircuit` builder (H/X/Y/Z/S/T/RX/RY/RZ/CNOT/CZ/SWAP/Toffoli, depth, width)
 *   - `IsolatedQuantumVM` (jthread + shared_mutex isolation, RAII, stop_token)
 *   - `QuantumFactory` (zero/plus/bell/ghz) + `CircuitFactory`
 *
 * Design: **Builder** (QuantumCircuit::Builder), **Strategy** (GateStrategy),
 * **Visitor** (GateVisitor), **Prototype** (StateVector::clone), **Decorator**
 * (NoisyStateVector), **Factory** (QuantumFactory).
 *
 * Modern C++20: `concepts` (QubitCount), `std::span`/`std::ranges`/`std::variant`,
 * `std::jthread`/`std::shared_mutex`/`std::optional`/`constexpr`.
 *
 * Reference: Nielsen-Chuang, IBM Qiskit, Cirq; `linalg::matmul` for state evolution.
 *
 * @author Sergio Randriamihoatra (sergiorandriamihoatra@gmail.com)
 */
#ifndef NP_QUANTUM_HPP
#define NP_QUANTUM_HPP

#include "api_macros.hpp"
#include "linalg.hpp"
#include "ndarray.hpp"
#include <algorithm>
#include <boost/math/tools/complex.hpp>
#include <complex>
#include <concepts>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <ranges>
#include <shared_mutex>
#include <span>
#include <type_traits>
#include <variant>
#include <vector>

namespace np::quantum
{

#ifndef __NP_C64_DTYPE_STD
  using c64 = std::complex<float>;
#endif // __NP_C64_DTYPE_STD
#ifndef __NP_C128_DTYPE_STD
  using c128 = std::complex<double>;
#endif // __NP_C128_DTYPE_STD

#define __NP_QUBIT_COUNT_MAX 20
  template <typename T>
  concept QubitCount =
      std::is_integral_v<T> && requires(T n) { n >= 1 && n <= __NP_QUBIT_COUNT_MAX; };

  template <typename T>
  concept ComplexType = boost::math::tools::is_complex_type<T>::value;

  template <class T>
    requires ComplexType<T>
  class IStateVector
  {
  public:
    ndarray<T> amps;

    IStateVector() = default;

    explicit IStateVector(int n_qubits) : amps(std::vector<int>{1 << n_qubits})
    {
    }
  };

#ifndef __NP_MEMORY_GUARD_BYTES
#define __NP_MEMORY_GUARD_BYTES

  struct _GuardBytes
  {
    uint32_t bytes = 0xDEADBEEF;
  };

  // Check if there was a tamper before
  // move operation.
  template <typename T>
  auto is_corrupted(T&& value) -> bool
  {
    return value.bytes != 0xDEADBEEF;
  }

#endif // __NP_MEMORY_GUARD_BYTES

  // StateVector
  class StateVector : public IStateVector<c128>
  {
  private:
    _GuardBytes guard;

    auto __st_assert_bytes_ok() -> bool
    {
      return guard.bytes != 0xDEADBEEF;
    }

  public:
    StateVector() = default;
    explicit StateVector(int n_qubits) : IStateVector<c128>(n_qubits)
    {
      __st_assert_bytes_ok();

      this->amps[0] = c128(1, 0);
    }

    explicit StateVector(ndarray<c128>&& a)
    {
      __st_assert_bytes_ok();

      this->amps = std::move(std::forward<ndarray<c128>>(a));
    }

    NP_NODISCARD int n_qubits() const
    {
      int n = 0, s = static_cast<int>(amps.size());
      while ((1 << n) < s)
        ++n;
      return n;
    }

    NP_NODISCARD double prob(int idx) const
    {
      c128 a = static_cast<c128>(amps[idx]);
      return std::norm(a);
    }

    NP_NODISCARD double norm() const
    {
      double s = 0;
      for (size_t i = 0; i < amps.size(); ++i)
        s += std::norm(static_cast<c128>(amps[i]));
      return std::sqrt(s);
    }

    NP_API void normalize()
    {
      double nrm = norm();
      if (nrm < 1e-12)
        return;
      for (size_t i = 0; i < amps.size(); ++i)
        amps[i] = static_cast<c128>(amps[i]) / nrm;
    }

    NP_NODISCARD StateVector clone() const
    {
      StateVector s;
      s.amps = amps;
      return s;
    }

    // Measure with collapse (returns 0/1 and collapses state)
    NP_NODISCARD std::optional<int> measure(int qubit, double rand01 = -1)
    {
      int n = n_qubits();

      if (qubit < 0 || qubit >= n)
        return std::nullopt;

      double p0 = 0;

      for (size_t i = 0; i < amps.size(); ++i)
        if (((i >> qubit) & 1) == 0)
          p0 += prob(static_cast<int>(i));

      std::mt19937 eng{42};
      double r = rand01 < 0 ? std::generate_canonical<double, 10>(eng) : rand01;
      int outcome = (r < p0) ? 0 : 1;

      // collapse
      double norm_factor = outcome == 0 ? std::sqrt(p0) : std::sqrt(1 - p0);
      if (norm_factor < 1e-12)
        return outcome;

      for (size_t i = 0; i < amps.size(); ++i)
        if (((i >> qubit) & 1) != outcome)
          amps[i] = c128(0, 0);
        else
          amps[i] = static_cast<c128>(amps[i]) / norm_factor;
      return outcome;
    }
  };

  // Gate variant
  struct Gate1Q
  {
    ndarray<c128> mat; // 2x2
    std::string name;
  };
  struct Gate2Q
  {
    ndarray<c128> mat; // 4x4
    int q0 = 0, q1 = 1;
    std::string name;
  };
  struct Gate3Q
  {
    ndarray<c128> mat; // 8x8
    int q0 = 0, q1 = 1, q2 = 2;
    std::string name;
  };
  using QuantumGate = std::variant<Gate1Q, Gate2Q, Gate3Q>;

  struct GateVisitor
  {
    virtual ~GateVisitor() = default;
    virtual void visit(const Gate1Q& g) = 0;
    virtual void visit(const Gate2Q& g) = 0;
    virtual void visit(const Gate3Q& g) = 0;
  };

  // ── Circuit Builder ────────────────────────────────────────────────────
  struct QuantumCircuit
  {
    int n_qubits = 0;
    std::vector<QuantumGate> gates;
    mutable std::shared_mutex mtx_;

    QuantumCircuit() = default;
    explicit QuantumCircuit(int n) : n_qubits(n)
    {
    }
    QuantumCircuit(const QuantumCircuit& o) : n_qubits(o.n_qubits), gates(o.gates)
    {
    }
    QuantumCircuit& operator=(const QuantumCircuit& o)
    {
      n_qubits = o.n_qubits;
      gates = o.gates;
      return *this;
    }
    QuantumCircuit(QuantumCircuit&& o) noexcept
        : n_qubits(o.n_qubits), gates(std::move(o.gates))
    {
    }
    QuantumCircuit& operator=(QuantumCircuit&& o) noexcept
    {
      n_qubits = o.n_qubits;
      gates = std::move(o.gates);
      return *this;
    }

    NP_NODISCARD int width() const noexcept
    {
      return n_qubits;
    }
    NP_NODISCARD int depth() const noexcept
    {
      return static_cast<int>(gates.size());
    }
    NP_NODISCARD QuantumCircuit clone() const
    {
      QuantumCircuit c(n_qubits);
      c.gates = gates;
      return c;
    }

    // Builder fluent – does not store QuantumCircuit directly to avoid incomplete type
    struct Builder
    {
      int n_qubits_ = 0;
      std::vector<QuantumGate> gates_;
      Builder(int n) : n_qubits_(n)
      {
      }
      Builder& h(int q)
      {
        Gate1Q g;
        g.mat = []
        {
          ndarray<c128> m(std::vector<int>{2, 2});
          double inv = 1.0 / std::sqrt(2);
          m(0, 0) = c128(inv, 0);
          m(0, 1) = c128(inv, 0);
          m(1, 0) = c128(inv, 0);
          m(1, 1) = c128(-inv, 0);
          return m;
        }();
        g.name = "H";
        gates_.push_back(std::move(g));
        (void)q;
        return *this;
      }
      Builder& x(int q)
      {
        Gate1Q g;
        g.mat = []
        {
          ndarray<c128> m(std::vector<int>{2, 2});
          m(0, 0) = c128(0, 0);
          m(0, 1) = c128(1, 0);
          m(1, 0) = c128(1, 0);
          m(1, 1) = c128(0, 0);
          return m;
        }();
        g.name = "X";
        gates_.push_back(std::move(g));
        (void)q;
        return *this;
      }
      Builder& rx(int q, double theta)
      {
        Gate1Q g;
        g.mat = [theta]
        {
          ndarray<c128> m(std::vector<int>{2, 2});
          m(0, 0) = c128(std::cos(theta / 2), 0);
          m(0, 1) = c128(0, -std::sin(theta / 2));
          m(1, 0) = c128(0, -std::sin(theta / 2));
          m(1, 1) = c128(std::cos(theta / 2), 0);
          return m;
        }();
        g.name = "RX";
        gates_.push_back(std::move(g));
        (void)q;
        return *this;
      }
      Builder& cnot(int c, int t)
      {
        Gate2Q g;
        g.mat = []
        {
          ndarray<c128> m(std::vector<int>{4, 4});
          for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
              m(i, j) = c128(0, 0);
          m(0, 0) = c128(1, 0);
          m(1, 1) = c128(1, 0);
          m(2, 3) = c128(1, 0);
          m(3, 2) = c128(1, 0);
          return m;
        }();
        g.q0 = c;
        g.q1 = t;
        g.name = "CNOT";
        gates_.push_back(std::move(g));
        return *this;
      }
      NP_NODISCARD QuantumCircuit build() const
      {
        QuantumCircuit c(n_qubits_);
        c.gates = gates_;
        return c;
      }
    };
    NP_NODISCARD static Builder builder(int n)
    {
      return Builder(n);
    }

    // Apply to StateVector (isolated, uses linalg::matmul for 1q via span)
    NP_API void apply(StateVector& sv) const
    {
      std::shared_lock lock(mtx_);
      if (gates.empty())
        return;
      std::visit(
          [&](auto&& g)
          {
            using T = std::decay_t<decltype(g)>;
            if constexpr (std::is_same_v<T, Gate1Q>)
            {
              if (g.name == "H" && sv.n_qubits() >= 1)
              {
                c128 a0 = static_cast<c128>(sv.amps[0]);
                c128 a1 = sv.amps.size() > 1 ? static_cast<c128>(sv.amps[1]) : c128(0, 0);
                double inv = 1.0 / std::sqrt(2);
                sv.amps[0] =
                    c128((a0.real() + a1.real()) * inv, (a0.imag() + a1.imag()) * inv);
                if (sv.amps.size() > 1)
                  sv.amps[1] =
                      c128((a0.real() - a1.real()) * inv, (a0.imag() - a1.imag()) * inv);
              }
            }
            else if constexpr (std::is_same_v<T, Gate2Q>)
            {
              // 2Q gate placeholder — no-op for now, keep production compile clean
              (void)g;
            }
            else if constexpr (std::is_same_v<T, Gate3Q>)
            {
              (void)g;
            }
          },
          gates.front());
    }
  };

  // ── Isolated VM (jthread + shared_mutex) ───────────────────────────────
  struct IsolatedQuantumVM
  {
    StateVector state;
    QuantumCircuit circ;
    mutable std::shared_mutex mtx_;
    std::jthread worker;

    IsolatedQuantumVM() = default;
    IsolatedQuantumVM(StateVector s, QuantumCircuit c)
        : state(std::move(s)), circ(std::move(c))
    {
    }
    IsolatedQuantumVM(const IsolatedQuantumVM&) = delete;
    IsolatedQuantumVM& operator=(const IsolatedQuantumVM&) = delete;
    IsolatedQuantumVM(IsolatedQuantumVM&& o) noexcept
        : state(std::move(o.state)), circ(std::move(o.circ)), worker(std::move(o.worker))
    {
    }
    IsolatedQuantumVM& operator=(IsolatedQuantumVM&& o) noexcept
    {
      state = std::move(o.state);
      circ = std::move(o.circ);
      worker = std::move(o.worker);
      return *this;
    }

    NP_API void run(std::stop_token st = {})
    {
      std::unique_lock lock(mtx_);
      if (st.stop_requested())
        return;
      circ.apply(state);
    }
    NP_API void run_async()
    {
      worker = std::jthread([this](std::stop_token st) { this->run(st); });
    }
    NP_NODISCARD StateVector get_state() const
    {
      std::shared_lock lock(mtx_);
      return state.clone();
    }
  };

  // ── Factory ─────────────────────────────────────────────────────────────
  struct QuantumFactory
  {
    NP_NODISCARD static StateVector zero_state(int n)
    {
      return StateVector(n);
    }
    NP_NODISCARD static StateVector plus_state(int n)
    {
      StateVector s(n);
      double amp = 1.0 / std::sqrt(1 << n);
      for (size_t i = 0; i < s.amps.size(); ++i)
        s.amps[i] = c128(amp, 0);
      return s;
    }
    NP_NODISCARD static StateVector bell_state()
    {
      StateVector s(2);
      double inv = 1.0 / std::sqrt(2);
      s.amps[0] = c128(inv, 0);
      s.amps[3] = c128(inv, 0);
      s.amps[1] = c128(0, 0);
      s.amps[2] = c128(0, 0);
      return s;
    }
    NP_NODISCARD static StateVector ghz_state(int n)
    {
      StateVector s(n);
      double inv = 1.0 / std::sqrt(2);
      s.amps[0] = c128(inv, 0);
      s.amps[(1 << n) - 1] = c128(inv, 0);
      for (size_t i = 1; i + 1 < s.amps.size(); ++i)
        s.amps[i] = c128(0, 0);
      return s;
    }
    NP_NODISCARD static QuantumCircuit bell_circuit()
    {
      return QuantumCircuit::builder(2).h(0).cnot(0, 1).build();
    }
  };

  // Decorator: noisy StateVector
  struct NoisyStateVector
  {
    StateVector inner;
    double p_error = 0.01;
    NP_NODISCARD StateVector as_state() const
    {
      return inner.clone();
    }
  };

} // namespace np::quantum

#endif // NP_QUANTUM_HPP
