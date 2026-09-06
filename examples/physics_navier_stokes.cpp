/**
 * @example physics_navier_stokes.cpp
 * Navier-Stokes 2D lid-driven cavity via np::physics
 */
#include <iostream>
#include <np/np.hpp>

int main()
{
    auto ns = np::physics::NavierStokes2D(32, 32, 100);
    ns.state.u(16, 16) = 1.0;
    for (int i = 0; i < 5; ++i)
        ns.step();
    std::cout << "ke " << ns.kinetic_energy() << " div " << ns.max_divergence() << "\n";
    auto ns2 = np::physics::NavierStokes2D(16, 16, 200);
    std::cout << "builder " << ns2.state.nx << "x" << ns2.state.ny << "\n";
    return 0;
}
