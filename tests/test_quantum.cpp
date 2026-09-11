/**
 * @file test_quantum.cpp
 */
#include "test_util.hpp"
#include <np/np.hpp>
int main()
{
    using namespace np::quantum;
    auto s = QuantumFactory::zero_state(2);
    test::check(s.n_qubits() == 2, "zero_state");
    test::check(std::abs(s.prob(0) - 1) < 1e-9, "prob 0");
    auto p = QuantumFactory::plus_state(1);
    test::check(std::abs(p.prob(0) - 0.5) < 1e-9, "plus_state");
    // Bell circuit H(0)+CNOT(0,1) must entangle: |00> -> (|00>+|11>)/sqrt2.
    // (The old apply() visited only gates.front() on amps[0..1], so this
    // two-gate circuit could never produce entanglement.)
    {
        auto bell = QuantumFactory::bell_circuit();
        test::check(bell.depth() == 2, "bell depth");
        StateVector s(2);
        bell.apply(s);
        auto ref = QuantumFactory::bell_state();
        bool ok = true;
        for (size_t i = 0; i < 4; ++i)
            ok = ok && std::abs(s.amps.at(i) - ref.amps.at(i)) < 1e-12;
        test::check(ok, "bell circuit fidelity");
    }
    // X on qubit 1 of a 2-qubit register (stride-k, not amps[0..1]).
    {
        StateVector s(2); // |00>
        QuantumCircuit::builder(2).x(1).build().apply(s);
        test::check(std::abs(s.prob(2) - 1.0) < 1e-12, "X on qubit 1");
    }
    // RX(pi) == X up to global phase: probabilities must match X.
    {
        StateVector a(1), b(1);
        QuantumCircuit::builder(1).x(0).build().apply(a);
        QuantumCircuit::builder(1).rx(0, 3.141592653589793).build().apply(b);
        test::check(std::abs(a.prob(0) - b.prob(0)) < 1e-9 && std::abs(a.prob(1) - b.prob(1)) < 1e-9,
                    "RX(pi) matches X probs");
    }
    // Toffoli |110> -> |111>.
    {
        StateVector s(3);
        QuantumCircuit::builder(3).x(0).build().apply(s);
        QuantumCircuit::builder(3).x(1).build().apply(s);
        QuantumCircuit::builder(3).toffoli(0, 1, 2).build().apply(s);
        test::check(std::abs(s.prob(7) - 1.0) < 1e-12, "toffoli flips target");
    }
    // Malformed circuits throw instead of silently miscomputing.
    {
        bool threw = false;
        try
        {
            StateVector s(1);
            QuantumCircuit::builder(1).x(5).build().apply(s);
        }
        catch (const std::invalid_argument &)
        {
            threw = true;
        }
        test::check(threw, "bad qubit index throws");
        threw = false;
        try
        {
            Gate1Q bad;
            bad.mat = np::ndarray<c128>(std::vector<int>{3, 3});
            bad.q = 0;
            QuantumCircuit c(1);
            c.gates.push_back(bad);
            StateVector t(1);
            c.apply(t);
        }
        catch (const std::invalid_argument &)
        {
            threw = true;
        }
        test::check(threw, "wrong-shape matrix throws");
        threw = false;
        try
        {
            StateVector s(0);
            (void)s;
        }
        catch (const std::invalid_argument &)
        {
            threw = true;
        }
        test::check(threw, "zero qubits throws");
    }
    return test::failures() ? 1 : 0;
}
