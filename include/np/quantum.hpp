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
 * Design: **Builder** (QuantumCircuit::Builder),
 * **Visitor** (GateVisitor), **Prototype** (StateVector::clone), **Decorator**
 * (NoisyStateVector), **Factory** (QuantumFactory).
 *
 * Modern C++20: `concepts` (QubitCount), `std::span`/`std::ranges`/`std::variant`,
 * `std::jthread`/`std::shared_mutex`/`std::optional`/`constexpr`.
 *
 * Reference: Nielsen-Chuang, IBM Qiskit, Cirq. State evolution is an
 * in-house stride-k state-vector update (no LAPACK/cuBLAS dependency).
 *
 * @author Sergio Randriamihoatra (sergiorandriamihoatra@gmail.com)
 */
#ifndef NP_QUANTUM_HPP
#define NP_QUANTUM_HPP

#include "api_macros.hpp"
#include "linalg.hpp"
#include "ndarray.hpp"
#include <algorithm>
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

#if __has_include(<boost/math/tools/complex.hpp>)
#include <boost/math/tools/complex.hpp>
#define NP_HAS_BOOST_COMPLEX 1
#endif

namespace np::quantum
{

#ifndef __NP_C64_DTYPE_STD
using c64 = std::complex<float>;
#endif // __NP_C64_DTYPE_STD
#ifndef __NP_C128_DTYPE_STD
using c128 = std::complex<double>;
#endif // __NP_C128_DTYPE_STD

#define __NP_QUBIT_COUNT_MAX 20
// NOTE: an earlier revision added `requires(T n) { n >= 1 && n <= ...; }`,
// but a requires-expression only checks syntactic validity, so ANY integral
// type satisfied it. The concept now constrains the type only; callers
// validate the value at runtime (see IStateVector ctor).
template <typename T>
concept QubitCount = std::is_integral_v<T>;

namespace detail
{
template <typename T> struct is_complex : std::false_type
{
};
template <typename T> struct is_complex<std::complex<T>> : std::true_type
{
};
template <typename T> inline constexpr bool is_complex_v = is_complex<std::remove_cv_t<T>>::value;
} // namespace detail

#if defined(NP_HAS_BOOST_COMPLEX)
template <typename T>
concept ComplexType = boost::math::tools::is_complex_type<T>::value || detail::is_complex_v<T>;
#else
template <typename T>
concept ComplexType = detail::is_complex_v<T>;
#endif

template <class T>
    requires ComplexType<T>
class IStateVector
{
  public:
    ndarray<T> amps;

    IStateVector() = default;

    explicit IStateVector(int n_qubits) : amps(validated_shape(n_qubits))
    {
        // NOTE (honesty audit): an earlier revision had _GuardBytes /
        // is_corrupted / __st_assert_bytes_ok "tamper detection" here whose
        // result was computed and discarded in every ctor (and whose
        // predicate was inverted: true meant corrupt). It never fired by
        // construction, so it is deleted rather than fixed.
    }

  private:
    // Validates BEFORE the 1<<n shift runs (negative/huge n would otherwise
    // be UB or a doomed allocation inside the mem-initializer).
    NP_NODISCARD static std::vector<int> validated_shape(int n_qubits)
    {
        if (n_qubits < 1 || n_qubits > __NP_QUBIT_COUNT_MAX)
        {
            throw std::invalid_argument("StateVector: n_qubits must be in [1, 20]");
        }
        return std::vector<int>{1 << n_qubits};
    }
};

// StateVector
class StateVector : public IStateVector<c128>
{
  public:
    StateVector() = default;
    explicit StateVector(int n_qubits) : IStateVector<c128>(n_qubits)
    {
        this->amps[0] = c128(1, 0);
    }

    explicit StateVector(ndarray<c128> &&a)
    {
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
    int q = 0;         // target qubit (bit index into the basis state)
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
    virtual void visit(const Gate1Q &g) = 0;
    virtual void visit(const Gate2Q &g) = 0;
    virtual void visit(const Gate3Q &g) = 0;
};

namespace detail
{
// Apply a dim x dim unitary (row-major, length dim*dim) to the listed
// qubits of a 2^n state vector, in place.
//
// Qubit convention (documented choice): qubit k <-> bit k of the basis
// index (little-endian), matching StateVector::measure()'s
// `((i >> qubit) & 1)`. Multi-qubit matrix rows/cols are indexed
// [b_{qk-1} ... b_{q1} b_{q0}], consistent with the Builder's CNOT/CZ/SWAP
// tables. Throws invalid_argument on out-of-range/duplicate qubits or a
// wrong-sized matrix. Cost O(2^n * dim^2); n is capped at 20 by
// __NP_QUBIT_COUNT_MAX.
inline void apply_kq(ndarray<c128> &amps, const std::vector<int> &qubits, const c128 *mat, std::size_t dim)
{
    const std::size_t n_amps = amps.size();
    for (int q : qubits)
    {
        if (q < 0 || static_cast<std::size_t>(q) >= 8 * sizeof(std::size_t))
        {
            throw std::invalid_argument("apply: qubit index out of range");
        }
    }
    for (std::size_t a = 0; a < qubits.size(); ++a)
    {
        for (std::size_t b = a + 1; b < qubits.size(); ++b)
        {
            if (qubits[a] == qubits[b])
            {
                throw std::invalid_argument("apply: duplicate qubit index in gate");
            }
        }
    }
    std::size_t qmask = 0;
    for (int q : qubits)
    {
        qmask |= std::size_t{1} << static_cast<std::size_t>(q);
    }
    std::vector<c128> vin(dim), vout(dim);
    std::vector<std::size_t> idx(dim);
    auto &buf = amps.data();
    for (std::size_t base = 0; base < n_amps; ++base)
    {
        if ((base & qmask) != 0)
        {
            continue;
        }
        for (std::size_t t = 0; t < dim; ++t)
        {
            std::size_t k = base;
            for (std::size_t j = 0; j < qubits.size(); ++j)
            {
                if ((t >> j) & std::size_t{1})
                {
                    k |= std::size_t{1} << static_cast<std::size_t>(qubits[j]);
                }
            }
            idx[t] = k;
            vin[t] = static_cast<c128>(buf[k]);
        }
        for (std::size_t r = 0; r < dim; ++r)
        {
            c128 acc(0, 0);
            for (std::size_t c = 0; c < dim; ++c)
            {
                acc += mat[r * dim + c] * vin[c];
            }
            vout[r] = acc;
        }
        for (std::size_t t = 0; t < dim; ++t)
        {
            buf[idx[t]] = vout[t];
        }
    }
}
} // namespace detail

// Circuit Builder
struct QuantumCircuit
{
    int n_qubits = 0;
    std::vector<QuantumGate> gates;
    mutable std::shared_mutex mtx_;

    QuantumCircuit() = default;
    explicit QuantumCircuit(int n) : n_qubits(n)
    {
    }
    QuantumCircuit(const QuantumCircuit &o) : n_qubits(o.n_qubits), gates(o.gates)
    {
    }
    QuantumCircuit &operator=(const QuantumCircuit &o)
    {
        n_qubits = o.n_qubits;
        gates = o.gates;
        return *this;
    }
    QuantumCircuit(QuantumCircuit &&o) noexcept : n_qubits(o.n_qubits), gates(std::move(o.gates))
    {
    }
    QuantumCircuit &operator=(QuantumCircuit &&o) noexcept
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
        Builder &h(int q)
        {
            Gate1Q g;
            g.mat = [] {
                ndarray<c128> m(std::vector<int>{2, 2});
                double inv = 1.0 / std::sqrt(2);
                m(0, 0) = c128(inv, 0);
                m(0, 1) = c128(inv, 0);
                m(1, 0) = c128(inv, 0);
                m(1, 1) = c128(-inv, 0);
                return m;
            }();
            g.q = q;
            g.name = "H";
            gates_.push_back(std::move(g));
            return *this;
        }
        Builder &y(int q)
        {
            Gate1Q g;
            g.mat = [] {
                ndarray<c128> m(std::vector<int>{2, 2});
                m(0, 0) = c128(0, 0);
                m(0, 1) = c128(0, -1);
                m(1, 0) = c128(0, 1);
                m(1, 1) = c128(0, 0);
                return m;
            }();
            g.q = q;
            g.name = "Y";
            gates_.push_back(std::move(g));
            return *this;
        }
        Builder &z(int q)
        {
            Gate1Q g;
            g.mat = [] {
                ndarray<c128> m(std::vector<int>{2, 2});
                m(0, 0) = c128(1, 0);
                m(0, 1) = c128(0, 0);
                m(1, 0) = c128(0, 0);
                m(1, 1) = c128(-1, 0);
                return m;
            }();
            g.q = q;
            g.name = "Z";
            gates_.push_back(std::move(g));
            return *this;
        }
        Builder &s(int q)
        {
            Gate1Q g;
            g.mat = [] {
                ndarray<c128> m(std::vector<int>{2, 2});
                m(0, 0) = c128(1, 0);
                m(0, 1) = c128(0, 0);
                m(1, 0) = c128(0, 0);
                m(1, 1) = c128(0, 1);
                return m;
            }();
            g.q = q;
            g.name = "S";
            gates_.push_back(std::move(g));
            return *this;
        }
        Builder &t(int q)
        {
            Gate1Q g;
            const double c = std::cos(3.141592653589793 / 4.0);
            const double s = std::sin(3.141592653589793 / 4.0);
            g.mat = [c, s] {
                ndarray<c128> m(std::vector<int>{2, 2});
                m(0, 0) = c128(1, 0);
                m(0, 1) = c128(0, 0);
                m(1, 0) = c128(0, 0);
                m(1, 1) = c128(c, s);
                return m;
            }();
            g.q = q;
            g.name = "T";
            gates_.push_back(std::move(g));
            return *this;
        }
        Builder &ry(int q, double theta)
        {
            Gate1Q g;
            g.mat = [theta] {
                ndarray<c128> m(std::vector<int>{2, 2});
                m(0, 0) = c128(std::cos(theta / 2), 0);
                m(0, 1) = c128(-std::sin(theta / 2), 0);
                m(1, 0) = c128(std::sin(theta / 2), 0);
                m(1, 1) = c128(std::cos(theta / 2), 0);
                return m;
            }();
            g.q = q;
            g.name = "RY";
            gates_.push_back(std::move(g));
            return *this;
        }
        Builder &rz(int q, double theta)
        {
            Gate1Q g;
            g.mat = [theta] {
                ndarray<c128> m(std::vector<int>{2, 2});
                m(0, 0) = c128(std::cos(theta / 2), -std::sin(theta / 2));
                m(0, 1) = c128(0, 0);
                m(1, 0) = c128(0, 0);
                m(1, 1) = c128(std::cos(theta / 2), std::sin(theta / 2));
                return m;
            }();
            g.q = q;
            g.name = "RZ";
            gates_.push_back(std::move(g));
            return *this;
        }
        Builder &x(int q)
        {
            Gate1Q g;
            g.mat = [] {
                ndarray<c128> m(std::vector<int>{2, 2});
                m(0, 0) = c128(0, 0);
                m(0, 1) = c128(1, 0);
                m(1, 0) = c128(1, 0);
                m(1, 1) = c128(0, 0);
                return m;
            }();
            g.name = "X";
            g.q = q;
            gates_.push_back(std::move(g));
            return *this;
        }
        Builder &rx(int q, double theta)
        {
            Gate1Q g;
            g.mat = [theta] {
                ndarray<c128> m(std::vector<int>{2, 2});
                m(0, 0) = c128(std::cos(theta / 2), 0);
                m(0, 1) = c128(0, -std::sin(theta / 2));
                m(1, 0) = c128(0, -std::sin(theta / 2));
                m(1, 1) = c128(std::cos(theta / 2), 0);
                return m;
            }();
            g.name = "RX";
            g.q = q;
            gates_.push_back(std::move(g));
            return *this;
        }
        Builder &cz(int c, int t)
        {
            Gate2Q g;
            g.mat = [] {
                ndarray<c128> m(std::vector<int>{4, 4});
                for (int i = 0; i < 4; ++i)
                    for (int j = 0; j < 4; ++j)
                        m(i, j) = c128(0, 0);
                m(0, 0) = c128(1, 0);
                m(1, 1) = c128(1, 0);
                m(2, 2) = c128(1, 0);
                m(3, 3) = c128(-1, 0);
                return m;
            }();
            g.q0 = c;
            g.q1 = t;
            g.name = "CZ";
            gates_.push_back(std::move(g));
            return *this;
        }
        Builder &swap(int a, int b)
        {
            Gate2Q g;
            g.mat = [] {
                ndarray<c128> m(std::vector<int>{4, 4});
                for (int i = 0; i < 4; ++i)
                    for (int j = 0; j < 4; ++j)
                        m(i, j) = c128(0, 0);
                m(0, 0) = c128(1, 0);
                m(1, 2) = c128(1, 0);
                m(2, 1) = c128(1, 0);
                m(3, 3) = c128(1, 0);
                return m;
            }();
            g.q0 = a;
            g.q1 = b;
            g.name = "SWAP";
            gates_.push_back(std::move(g));
            return *this;
        }
        Builder &toffoli(int c0, int c1, int t)
        {
            Gate3Q g;
            // LSB-first like CNOT above: controls are bits q0,q1, so the
            // flip pair is |011> (idx 3) <-> |111> (idx 7), not rows 6/7.
            g.mat = [] {
                ndarray<c128> m(std::vector<int>{8, 8});
                for (int i = 0; i < 8; ++i)
                    for (int j = 0; j < 8; ++j)
                        m(i, j) = c128(i == j ? 1 : 0, 0);
                m(3, 3) = c128(0, 0);
                m(7, 7) = c128(0, 0);
                m(3, 7) = c128(1, 0);
                m(7, 3) = c128(1, 0);
                return m;
            }();
            g.q0 = c0;
            g.q1 = c1;
            g.q2 = t;
            g.name = "Toffoli";
            gates_.push_back(std::move(g));
            return *this;
        }
        Builder &cnot(int c, int t)
        {
            Gate2Q g;
            // NOTE (honesty audit): this table previously swapped rows 2/3,
            // which is CNOT only under MSB-first ordering — but measure()
            // reads qubit k as bit k (LSB-first), so |01> never flipped.
            // Fixed: control = bit q0, target = bit q1 (|01> <-> |11>).
            g.mat = [] {
                ndarray<c128> m(std::vector<int>{4, 4});
                for (int i = 0; i < 4; ++i)
                    for (int j = 0; j < 4; ++j)
                        m(i, j) = c128(0, 0);
                m(0, 0) = c128(1, 0);
                m(1, 3) = c128(1, 0);
                m(2, 2) = c128(1, 0);
                m(3, 1) = c128(1, 0);
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

    // Apply every gate in order to StateVector via detail::apply_kq
    // (stride-k state-vector update; qubit k <-> bit k, see detail).
    // NOTE (honesty audit): an earlier revision visited only gates.front(),
    // applied only "H" to amps[0..1] regardless of qubit index, and no-op'd
    // 2Q/3Q gates — bell_circuit() could never entangle. Every gate now
    // applies with its stored matrix and qubit indices; malformed gates
    // (wrong matrix shape, bad/duplicate qubits) throw invalid_argument.
    NP_API void apply(StateVector &sv) const
    {
        std::shared_lock lock(mtx_);
        const std::size_t s = sv.amps.size();
        if (s == 0 || (s & (s - 1)) != 0 || s > (std::size_t{1} << __NP_QUBIT_COUNT_MAX))
        {
            throw std::invalid_argument("apply: state size must be a nonzero power of two within qubit cap");
        }
        int n = 0;
        while ((std::size_t{1} << n) < s)
        {
            ++n;
        }
        const auto check_qubits = [&](const std::vector<int> &qs) {
            for (int q : qs)
            {
                if (q < 0 || q >= n)
                {
                    throw std::invalid_argument("apply: qubit index out of range");
                }
            }
        };
        const auto mat_data = [&](const ndarray<c128> &m, int dim) -> const c128 * {
            if (m.shape != std::vector<int>{dim, dim})
            {
                throw std::invalid_argument("apply: gate matrix has wrong shape");
            }
            return m.data().data();
        };
        for (const QuantumGate &gate : gates)
        {
            std::visit(
                [&](auto &&g) {
                    using T = std::decay_t<decltype(g)>;
                    if constexpr (std::is_same_v<T, Gate1Q>)
                    {
                        check_qubits({g.q});
                        detail::apply_kq(sv.amps, {g.q}, mat_data(g.mat, 2), 2);
                    }
                    else if constexpr (std::is_same_v<T, Gate2Q>)
                    {
                        check_qubits({g.q0, g.q1});
                        detail::apply_kq(sv.amps, {g.q0, g.q1}, mat_data(g.mat, 4), 4);
                    }
                    else if constexpr (std::is_same_v<T, Gate3Q>)
                    {
                        check_qubits({g.q0, g.q1, g.q2});
                        detail::apply_kq(sv.amps, {g.q0, g.q1, g.q2}, mat_data(g.mat, 8), 8);
                    }
                },
                gate);
        }
    }
};

// Isolated VM (jthread + shared_mutex)
struct IsolatedQuantumVM
{
    StateVector state;
    QuantumCircuit circ;
    mutable std::shared_mutex mtx_;
    std::jthread worker;

    IsolatedQuantumVM() = default;
    IsolatedQuantumVM(StateVector s, QuantumCircuit c) : state(std::move(s)), circ(std::move(c))
    {
    }
    IsolatedQuantumVM(const IsolatedQuantumVM &) = delete;
    IsolatedQuantumVM &operator=(const IsolatedQuantumVM &) = delete;
    IsolatedQuantumVM(IsolatedQuantumVM &&o) noexcept
        : state(std::move(o.state)), circ(std::move(o.circ)), worker(std::move(o.worker))
    {
    }
    IsolatedQuantumVM &operator=(IsolatedQuantumVM &&o) noexcept
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

// Factory
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
