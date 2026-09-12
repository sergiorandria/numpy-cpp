/**
 * @example spectral_hodge.cpp
 * Spectral sequence + Hodge star via lattice
 */
#include <iostream>
#include <np/np.hpp>
int main()
{
    auto lat = np::lattice::LatticeFactory::cubic<double>(2);
    auto ss = np::spectral::lattice_spectral(lat);
    std::cout << "spectral " << ss.bundle_name << " collapses " << ss.collapses << "\n";
    return 0;
}
