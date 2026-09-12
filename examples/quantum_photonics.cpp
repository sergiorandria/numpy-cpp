/**
 * @example quantum_photonics.cpp
 * Quantum StateVector + Mach-Zehnder photonics
 */
#include <iostream>
#include <np/np.hpp>

int main()
{
    auto s = np::quantum::QuantumFactory::plus_state(2);
    std::cout << "plus prob0 " << s.prob(0) << "\n";
    auto mesh = np::photonics::PhotonicsFactory::identity(2);
    auto x = np::ndarray<std::complex<double>>(std::vector<int>{2});
    x[0] = {1, 0};
    x[1] = {0, 0};
    auto y = mesh.apply(x);
    std::cout << "photonics " << y[0] << "\n";

    auto w = np::eye<float>(2);
    np::analog::Crossbar cb(w);
    auto xv = np::ndarray<float>(std::vector<int>{2});
    xv[0] = 1;
    xv[1] = 2;
    std::cout << "analog dot " << cb.dot(xv)[0] << "\n";
    return 0;
}
