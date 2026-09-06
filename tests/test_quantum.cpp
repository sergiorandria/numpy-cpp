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
    return test::failures() ? 1 : 0;
}
