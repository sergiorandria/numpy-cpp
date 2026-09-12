/**
 * @example physics_extended.cpp
 * Extended np::physics tour: heat decay, projectile range, Burgers
 * steepening, wave energy, oscillator, pendulum and a binary orbit.
 */
#include <iostream>
#include <np/np.hpp>

int main()
{
    using namespace np::physics;

    // Heat: Gaussian hot spot diffuses away through cold walls.
    Heat2D h(21, 21, 0.05, 0.01);
    h.set_gaussian(0.5, 0.5, 0.08);
    std::cout << "heat peak0 " << h.max_temp() << " total0 " << h.total_heat() << "\n";
    for (int i = 0; i < 60; ++i)
    {
        h.step();
    }
    std::cout << "heat peak60 " << h.max_temp() << " total60 " << h.total_heat() << "\n";

    // Projectile: vacuum range vs quadratic drag.
    Projectile free(9.81, 0.0), draggy(9.81, 0.05);
    const double v0 = 20.0, ang = M_PI / 4.0;
    std::cout << "range vacuum " << Projectile::range_vacuum(v0, ang) << " simulated "
              << Projectile::range(free.simulate(v0, ang, 0.001)) << " drag "
              << Projectile::range(draggy.simulate(v0, ang, 0.001)) << "\n";

    // Burgers: sine wave steepens, viscosity dissipates the peak.
    Burgers1D b(64, 0.05, 0.0005);
    b.set_sine(1.0);
    for (int i = 0; i < 40; ++i)
    {
        b.step();
    }
    std::cout << "burgers peak " << b.max_abs() << " mass " << b.mass() << "\n";

    // Potential flow: Laplace solve recovers the uniform freestream.
    PotentialFlow2D pf(24, 24, 1.0);
    pf.iters = 2000;
    pf.solve();
    auto vel = pf.velocity();
    std::cout << "potential centerline u " << vel.u(12, 12) << " v " << vel.v(12, 12) << "\n";

    // Waves: plucked string keeps its energy.
    Wave1D w(101, 1.0, 0.004);
    w.pluck_gaussian(0.5, 0.05);
    const double e0 = w.energy();
    for (int i = 0; i < 200; ++i)
    {
        w.step();
    }
    std::cout << "wave energy0 " << e0 << " energy200 " << w.energy() << "\n";

    // Oscillator + pendulum periods.
    HarmonicOscillator o(1.0, 1.0, 1.0, 0.0);
    std::cout << "oscillator period " << o.period() << " energy " << o.energy() << "\n";
    Pendulum pd(1.0, 9.81, 0.05, 0.0);
    std::cout << "pendulum small-angle period " << pd.period_small() << "\n";

    // Binary orbit: one revolution.
    NBody nb(1.0, 1e-6);
    const double om = std::sqrt(2.0), v = om * 0.5;
    nb.add_body(1.0, {-0.5, 0.0, 0.0}, {0.0, -v, 0.0});
    nb.add_body(1.0, {0.5, 0.0, 0.0}, {0.0, v, 0.0});
    const double e0b = nb.total_energy();
    const int n = static_cast<int>(2.0 * M_PI / om / 0.001);
    for (int i = 0; i < n; ++i)
    {
        nb.step_verlet(0.001);
    }
    std::cout << "nbody body0 (" << nb.pos[0][0] << ", " << nb.pos[0][1] << ") energy drift "
              << nb.total_energy() - e0b << "\n";

    // Boussinesq smoke: warm blob below drives flow.
    Boussinesq2D q(20, 20, 100.0);
    q.T(6, 10) += 0.4;
    for (int i = 0; i < 30; ++i)
    {
        q.step();
    }
    std::cout << "boussinesq speed " << q.max_speed() << " maxt " << q.max_temp() << "\n";
    return 0;
}
