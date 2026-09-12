/**
 * @example padic_hensel.cpp
 * p-adic Hensel lift x^2=2 mod 7^n
 */
#include <iostream>
#include <np/np.hpp>

int main()
{
    using namespace np::padic;
    Padic<int64_t> x0(7, 3, 6); // 3^2=2 mod7
    auto f = [](const Padic<int64_t> &x) { return Padic<int64_t>(x.p, x.value * x.value - 2, x.prec); };
    auto df = [](const Padic<int64_t> &x) { return Padic<int64_t>(x.p, 2 * x.value, x.prec); };
    HenselStrategy<int64_t> hs(10);
    auto root = hs.lift(x0, f, df);
    std::cout << "root " << root.value << " check " << (root.value * root.value - 2) % 7 << "\n";

    // Padic lattice
    auto lat = np::lattice::LatticeFactory::cubic<int64_t>(2);
    PadicLattice<int64_t> pl(lat, 7, 10);
    std::cout << "padic lattice rank " << pl.rank() << " vol " << pl.p_adic_volume() << "\n";

    // Lattice p-adic
    auto pl2 = to_padic_lattice(lat, 7, 10);
    std::cout << "to_padic " << pl2.rank() << "\n";
    return 0;
}
