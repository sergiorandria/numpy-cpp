/**
 * @file test_physics.cpp
 * @brief Tests for np::physics — fluids, heat, ballistics, waves, mechanics.
 */
#include "test_util.hpp"
#include <np/np.hpp>

namespace
{

bool is_finite_val(double x)
{
    return std::isfinite(x);
}

template <typename F> bool all_finite_field(const F &a)
{
    for (std::size_t n = 0; n < a.size(); ++n)
    {
        if (!is_finite_val(a.data()[n]))
        {
            return false;
        }
    }
    return true;
}

} // namespace

int main()
{
    using namespace np::physics;

    // ── Burgers1D: uniform flow is an exact invariant (periodic) ──
    {
        Burgers1D b(32, 0.01, 0.001);
        for (int i = 0; i < b.nx; ++i)
        {
            b.u(i) = 2.0;
        }
        b.step();
        test::check(test::approx(b.mass(), 2.0 * b.L, 1e-9), "burgers1d uniform mass");
        test::check(test::approx(b.max_abs(), 2.0, 1e-9), "burgers1d uniform invariant");
    }
    // ── Burgers1D: sine steepens but viscosity bounds it ──
    {
        Burgers1D b(64, 0.05, 0.0005);
        b.set_sine(1.0);
        const double m0 = b.max_abs();
        for (int s = 0; s < 40; ++s)
        {
            b.step();
        }
        test::check(is_finite_val(b.max_abs()) && b.max_abs() <= m0 + 1e-9, "burgers1d sine bounded");
        test::check(b.max_abs() < m0, "burgers1d sine decays");
    }
    // ── Burgers2D: quiescent stays quiescent ──
    {
        Burgers2D b(16, 16, 0.01, 0.001);
        b.step();
        test::check(b.max_speed() == 0.0, "burgers2d zero stays zero");
        b.u(8, 8) = 1.0;
        for (int s = 0; s < 5; ++s)
        {
            b.step();
        }
        test::check(is_finite_val(b.max_speed()), "burgers2d bump finite");
    }
    // ── Stokes2D: zero stays zero; bump keeps walls, kills divergence ──
    {
        Stokes2D s(16, 16, 1.0);
        s.step();
        test::check(s.kinetic_energy() == 0.0, "stokes2d zero energy");
        s.state.u(8, 8) = 1.0;
        s.step();
        bool walls = true;
        for (int i = 0; i < 16; ++i)
        {
            walls &= (s.state.u(0, i) == 0.0 && s.state.u(15, i) == 0.0);
        }
        test::check(walls, "stokes2d walls");
        test::check(is_finite_val(s.max_divergence()), "stokes2d div finite");
    }
    // ── PotentialFlow2D: converges to uniform freestream ──
    {
        PotentialFlow2D pf(24, 24, 1.0);
        pf.iters = 2000;
        pf.solve();
        auto vel = pf.velocity();
        double merr = 0.0, vmax = 0.0;
        for (int j = 4; j < 20; ++j)
        {
            for (int i = 4; i < 20; ++i)
            {
                merr = std::max(merr, std::abs(vel.u(j, i) - 1.0));
                vmax = std::max(vmax, std::abs(vel.v(j, i)));
            }
        }
        test::check(merr < 0.05, "potential u->freestream");
        test::check(vmax < 0.05, "potential v->0");
    }
    // ── AdvectionDiffusion2D: pulse translates with the flow ──
    {
        AdvectionDiffusion2D a(61, 21, 1.0, 0.0, 0.0, 0.0005);
        a.set_gaussian(0.3, 0.5, 0.03);
        const double m0 = a.total_mass();
        auto c0 = a.centroid();
        for (int s = 0; s < 200; ++s)
        {
            a.step(); // t = 0.1 -> dx_expected = 0.1
        }
        auto c1 = a.centroid();
        test::check(test::approx(c1.x, c0.x + 0.1, 1e-2), "advdiff centroid moves");
        test::check(test::approx(c1.y, c0.y, 1e-9), "advdiff no cross drift");
        test::check(test::approx(a.total_mass(), m0, 1e-2), "advdiff mass conserved");
    }
    // ── Heat2D: hot spot decays, stays symmetric and nonnegative ──
    {
        Heat2D h(21, 21, 0.05, 0.01);
        h.set_gaussian(0.5, 0.5, 0.08);
        const double t0 = h.total_heat(), peak0 = h.max_temp();
        for (int s = 0; s < 60; ++s)
        {
            h.step();
        }
        test::check(h.max_temp() < peak0, "heat2d peak decays");
        test::check(h.total_heat() < t0, "heat2d leaks through cold walls");
        test::check(h.max_temp() >= 0.0, "heat2d nonnegative");
        bool sym = true;
        for (int j = 0; j < 21 && sym; ++j)
        {
            for (int i = 0; i < 21; ++i)
            {
                if (!test::approx(h.T(j, i), h.T(j, 20 - i), 1e-9) || !test::approx(h.T(j, i), h.T(20 - j, i), 1e-9))
                {
                    sym = false;
                    break;
                }
            }
        }
        test::check(sym, "heat2d symmetry");
    }
    // ── Heat3D: smoke — center decays, field bounded ──
    {
        Heat3D h(11, 11, 11, 0.05, 0.005);
        h.T(5, 5, 5) = 1.0;
        for (int s = 0; s < 20; ++s)
        {
            h.step();
        }
        test::check(h.T(5, 5, 5) < 1.0, "heat3d center decays");
        test::check(all_finite_field(h.T), "heat3d finite");
        test::check(h.max_temp() <= 1.0 + 1e-12, "heat3d bounded");
    }
    // ── Boussinesq2D: warm blob drives flow, temperature bounded ──
    {
        Boussinesq2D q(20, 20, 100.0);
        q.T(6, 10) += 0.4; // warm perturbation in the lower half
        for (int s = 0; s < 30; ++s)
        {
            q.step();
        }
        test::check(q.max_speed() > 1e-9, "boussinesq convection starts");
        test::check(q.max_temp() <= q.T_hot + 1e-9, "boussinesq temp bounded");
    }
    // ── Projectile: vacuum range matches analytics ──
    {
        Projectile p(9.81, 0.0, 0.0);
        const double v0 = 10.0, ang = M_PI / 4.0;
        auto traj = p.simulate(v0, ang, 0.0005);
        const double r = Projectile::range(traj);
        test::check(test::approx(r, Projectile::range_vacuum(v0, ang), 1e-3), "projectile vacuum range");
        test::check(!traj.empty() && traj.back().y == 0.0, "projectile lands");
    }
    // ── Projectile: drag shortens the flight ──
    {
        Projectile free(9.81, 0.0), drag(9.81, 0.05);
        const double rf = Projectile::range(free.simulate(20.0, M_PI / 4.0, 0.001));
        const double rd = Projectile::range(drag.simulate(20.0, M_PI / 4.0, 0.001));
        test::check(rd < rf && rd > 0.0, "projectile drag shortens");
    }
    // ── Wave1D: symmetric pluck stays symmetric and bounded ──
    {
        Wave1D w(101, 1.0, 0.004); // CFL 0.4
        w.pluck_gaussian(0.5, 0.05);
        const double e0 = w.energy(), m0 = w.max_abs();
        for (int s = 0; s < 200; ++s)
        {
            w.step();
        }
        bool sym = true;
        for (int i = 0; i < 101; ++i)
        {
            if (!test::approx(w.u(i), w.u(100 - i), 1e-9))
            {
                sym = false;
                break;
            }
        }
        test::check(sym, "wave1d symmetry");
        test::check(w.max_abs() <= m0 + 1e-9, "wave1d bounded");
        test::check(test::approx(w.energy(), e0, 2e-2), "wave1d energy conserved");
    }
    // ── Wave2D: smoke — bounded, symmetric ──
    {
        Wave2D w(31, 31, 1.0, 0.004);
        w.pluck_gaussian(0.5, 0.5, 0.06);
        const double m0 = w.max_abs();
        for (int s = 0; s < 60; ++s)
        {
            w.step();
        }
        test::check(w.max_abs() <= m0 + 1e-9, "wave2d bounded");
        bool sym = true;
        for (int j = 0; j < 31 && sym; ++j)
        {
            for (int i = 0; i < 31; ++i)
            {
                if (!test::approx(w.u(j, i), w.u(j, 30 - i), 1e-9))
                {
                    sym = false;
                    break;
                }
            }
        }
        test::check(sym, "wave2d symmetry");
    }
    // ── HarmonicOscillator: returns after one period ──
    {
        HarmonicOscillator o(1.0, 1.0, 1.0, 0.0);
        const double e0 = o.energy(), T = o.period();
        const int n = static_cast<int>(T / 0.001);
        for (int s = 0; s < n; ++s)
        {
            o.step_verlet(0.001);
        }
        test::check(test::approx(o.x, 1.0, 1e-2), "oscillator period return");
        test::check(test::approx(o.energy(), e0, 1e-6), "oscillator energy");
    }
    // ── Pendulum: small-angle period matches 2π√(L/g) ──
    {
        Pendulum pd(1.0, 9.81, 0.05, 0.0);
        const double T = pd.period_small();
        const int n = static_cast<int>(T / 0.0005);
        for (int s = 0; s < n; ++s)
        {
            pd.step_rk4(0.0005);
        }
        test::check(test::approx(pd.theta, 0.05, 1e-2), "pendulum small-angle period");
    }
    // ── NBody: circular binary returns; momentum conserved ──
    {
        NBody nb(1.0, 1e-6);
        const double om = std::sqrt(2.0); // omega² = G*M/d³, M = 2, d = 1
        const double v = om * 0.5;
        nb.add_body(1.0, {-0.5, 0.0, 0.0}, {0.0, -v, 0.0});
        nb.add_body(1.0, {0.5, 0.0, 0.0}, {0.0, v, 0.0});
        const double e0 = nb.total_energy();
        const double T = 2.0 * M_PI / om;
        const int n = static_cast<int>(T / 0.001);
        for (int s = 0; s < n; ++s)
        {
            nb.step_verlet(0.001);
        }
        const double d0 = std::abs(nb.pos[0][0] + 0.5) + std::abs(nb.pos[0][1]) + std::abs(nb.pos[1][0] - 0.5);
        test::check(d0 < 0.05, "nbody binary period return");
        auto mom = nb.total_momentum();
        test::check(std::abs(mom[0]) + std::abs(mom[1]) + std::abs(mom[2]) < 1e-9, "nbody momentum");
        test::check(test::approx(nb.total_energy(), e0, 1e-3), "nbody energy");
    }

    // ── Physical constants & unit conversions ──
    {
        using namespace np::physics::constants;
        test::check(c == 299792458.0, "constants c exact");
        test::check(test::approx(ev_to_j(1.0), 1.602176634e-19, 1e-12), "ev_to_j");
        test::check(test::approx(j_to_ev(ev_to_j(13.6)), 13.6, 1e-9), "j_to_ev roundtrip");
        test::check(test::approx(amu_to_kg(1.0), 1.66053906660e-27, 1e-9), "amu_to_kg");
        test::check(test::approx(R_gas, kB * NA, 1e-12), "R_gas consistency");
    }
    // ── Relativity: gamma, addition, invariant ──
    {
        using namespace np::physics;
        const double c = constants::c;
        test::check(test::approx(lorentz_gamma(0.0), 1.0, 1e-12), "gamma(0)");
        test::check(test::approx(lorentz_gamma(0.6 * c), 1.25, 1e-9), "gamma(0.6c)");
        test::check(test::approx(velocity_add(c, 0.5 * c), c, 1e-9), "velocity_add caps at c");
        test::check(test::approx(relativistic_kinetic(1.0, 0.0), 0.0, 1e-12), "ke at rest");
        const double m = 1.0, v = 0.6 * c;
        const double E = relativistic_energy(m, v), p = relativistic_momentum(m, v);
        test::check(test::approx(invariant_mass(E, p), m, 1e-9), "E^2-(pc)^2 invariant");
        auto lt = lorentz_transform(0.0, 0.0, 0.5 * c);
        test::check(test::approx(lt[0], 0.0, 1e-12) && test::approx(lt[1], 0.0, 1e-12), "boost fixes origin");
        test::check(test::approx(doppler_longitudinal(500.0, 0.0), 500.0, 1e-12), "doppler static");
    }
    // ── EM: Lorentz force, cyclotron, Boris, loop field, optics ──
    {
        using namespace np::physics;
        std::array<double, 3> E{-1.0, 0.0, 0.0}, B{0.0, 0.0, 1.0}, v{0.0, 1.0, 0.0};
        auto F = lorentz_force(2.0, E, B, v);
        test::check(test::approx(F[0], 0.0, 1e-12) && test::approx(F[1], 0.0, 1e-12) && test::approx(F[2], 0.0, 1e-12),
                    "lorentz crossed-field cancel");
        test::check(test::approx(cyclotron_frequency(1.602176634e-19, 1.0, 9.1093837015e-31), 1.758820e11, 1e-4),
                    "cyclotron frequency");
        test::check(test::approx(biot_savart_loop_axis(1.0, 1.0, 0.0), constants::mu0 / 2.0, 1e-12),
                    "loop center mu0 I/2R");
        // Boris: uniform B, no E -> speed preserved over one gyroperiod.
        {
            const double q = 1.602176634e-19, m = 9.1093837015e-31, B0 = 1.0;
            ChargedParticle pt(q, m, {0.0, 0.0, 0.0}, {1.0e6, 0.0, 0.0});
            const double T = 2.0 * M_PI * m / (q * B0);
            const int n = 2000;
            const double dt = T / n;
            const double e0 = pt.kinetic_energy();
            std::array<double, 3> Ez{0.0, 0.0, 0.0}, Bz{0.0, 0.0, B0};
            for (int s = 0; s < n; ++s)
            {
                pt.boris_step(Ez, Bz, dt);
            }
            test::check(test::approx(pt.kinetic_energy(), e0, 1e-6), "boris energy conserved");
            const double r = std::hypot(pt.pos[0], pt.pos[1]);
            test::check(r < 0.05 * m * 1.0e6 / (q * B0) + 1e-9, "boris gyro-orbit closes");
        }
        // Optics: Snell, TIR, Fresnel normal incidence, thin lens.
        {
            auto t2 = snell(1.0, 1.5, 0.0);
            test::check(t2.has_value() && test::approx(*t2, 0.0, 1e-12), "snell normal");
            test::check(!snell(1.5, 1.0, M_PI / 3.0).has_value(), "snell TIR empty");
            test::check(test::approx(fresnel_reflectance(1.0, 1.5, 0.0), 0.04, 1e-9), "fresnel normal 4%");
            test::check(test::approx(fresnel_reflectance(1.5, 1.0, M_PI / 3.0), 1.0, 1e-12), "fresnel TIR total");
            auto di = thin_lens_image(0.1, 0.3);
            test::check(di.has_value() && test::approx(*di, 0.15, 1e-9), "thin lens");
            test::check(!thin_lens_image(0.1, 0.1).has_value(), "thin lens infinity");
        }
    }
    // ── Statmech: MB speeds, ideal gas, blackbody, heat capacities ──
    {
        using namespace np::physics;
        const double m = constants::m_e, T = 300.0;
        test::check(test::approx(mb_rms_speed(m, T) * mb_rms_speed(m, T), 3.0 * constants::kB * T / m, 1e-9),
                    "mb rms^2");
        test::check(test::approx(mb_most_probable(m, T), std::sqrt(2.0 * constants::kB * T / m), 1e-12),
                    "mb most probable");
        test::check(maxwell_boltzmann_pdf(-1.0, m, T) == 0.0, "mb pdf negative v");
        test::check(test::approx(ideal_gas_pressure(1.0, 1.0, 300.0), 4.141947e-21, 1e-9), "ideal gas");
        test::check(test::approx(stefan_boltzmann_exitance(5778.0), 6.33e7, 1e-2), "stefan-boltzmann sun");
        test::check(test::approx(wien_peak_wavelength(5778.0), 5.01e-7, 1e-2), "wien sun peak");
        test::check(planck_radiance(1e30, 300.0) == 0.0, "planck overflow guard");
        test::check(test::approx(einstein_heat_capacity(1e6, 300.0), 3.0 * constants::R_gas, 1e-6),
                    "einstein high-T 3R");
        test::check(einstein_heat_capacity(1.0, 1e6) < 1e-6, "einstein low-T frozen");
        test::check(schottky_heat_capacity(1e9, 1.0) < 1e-9, "schottky low-T empty");
        test::check(test::approx(partition_2level(0.0, 300.0), 2.0, 1e-12), "partition degenerate");
        test::check(test::approx(entropy_of_mixing(0.5), constants::R_gas * std::log(2.0), 1e-9), "mixing entropy max");
    }
    // ── Quantum: box, HO, hydrogen, Rabi, tunneling, spin, TISE ──
    {
        using namespace np::physics::qm;
        test::check(test::approx(particle_in_box_energy(1, 1.0, 1.0),
                                 np::physics::constants::h * np::physics::constants::h / 8.0, 1e-9),
                    "box E1");
        test::check(test::approx(particle_in_box_energy(2, 1.0, 1.0) / particle_in_box_energy(1, 1.0, 1.0), 4.0, 1e-12),
                    "box n^2 ladder");
        test::check(test::approx(ho_energy(0, 1.0), 0.5 * np::physics::constants::hbar, 1e-12), "ho zero point");
        test::check(test::approx(hydrogen_energy(1) / np::physics::constants::eV, -13.605693, 1e-5), "hydrogen E1");
        test::check(test::approx(hydrogen_energy(2) / hydrogen_energy(1), 0.25, 1e-12), "hydrogen 1/n^2");
        test::check(test::approx(rabi_probability(1.0, M_PI, 0.0), 1.0, 1e-9), "rabi full flip");
        test::check(test::approx(rabi_probability(1.0, 0.0, 0.0), 0.0, 1e-12), "rabi t=0");
        const double Tr = tunnel_rectangular(5.0 * np::physics::constants::eV, 10.0 * np::physics::constants::eV, 1e-10,
                                             np::physics::constants::m_e);
        test::check(Tr > 0.0 && Tr < 1.0, "tunneling partial");
        auto sx = pauli_x();
        test::check(sx(0, 1).real() == 1.0 && sx(1, 0).real() == 1.0 && sx(0, 0).real() == 0.0, "pauli_x");
        auto b = bloch_vector(0.0, 0.0);
        test::check(test::approx(b[2], 1.0, 1e-12), "bloch north pole");
        // TISE: harmonic well ground state ~= hbar*omega/2.
        {
            const int N = 200;
            const double xmin = -2e-9, xmax = 2e-9, m = np::physics::constants::m_e, om = 2e15;
            np::ndarray<double> V(std::vector<int>{N});
            for (int i = 0; i < N; ++i)
            {
                const double x = xmin + (xmax - xmin) * i / (N - 1);
                V.at(static_cast<std::size_t>(i)) = 0.5 * m * om * om * x * x;
            }
            auto sol = tise_1d(V, xmin, xmax, 2);
            const double e0 = np::physics::constants::hbar * om / 2.0;
            test::check(test::approx(sol.energies.at(0), e0, 2e-2), "tise ho ground state");
            test::check(sol.energies.at(1) > sol.energies.at(0), "tise ascending");
        }
    }
    // ── Nuclear & astro ──
    {
        using namespace np::physics;
        test::check(test::approx(decay_remaining(1.0, 1.0, 1.0), 0.5, 1e-12), "decay half-life");
        test::check(test::approx(decay_activity(1.0, 1.0), std::log(2.0), 1e-12), "activity");
        test::check(test::approx(semf_binding_mev(26, 56) / 56.0, 8.79, 2e-2), "semf Fe-56");
        test::check(test::approx(escape_velocity(5.972e24, 6.371e6), 11186.0, 2e-2), "earth escape");
        test::check(test::approx(schwarzschild_radius(1.98847e30) / 1000.0, 2.95, 1e-2), "sun schwarzschild km");
        test::check(test::approx(hubble_velocity(2.27e-18, 3.085677581e22), 70000.0, 1e-2), "hubble 1Mpc");
    }
    // ── Dimensionless numbers ──
    {
        using namespace np::physics;
        test::check(test::approx(reynolds(1000.0, 1.0, 0.1, 1e-3), 1e5, 1e-12), "reynolds water pipe");
        test::check(test::approx(mach_number(340.0, 340.0), 1.0, 1e-12), "mach 1");
        test::check(test::approx(stokes_drag(1e-3, 1e-6, 1e-3), 6.0 * M_PI * 1e-12, 1e-9), "stokes drag");
        test::check(nusselt_laminar_flat(1e4, 0.7) > 0.0, "nusselt positive");
        test::check(test::approx(prandtl(1.81e-5, 1006.0, 0.026), 0.70, 1e-2), "prandtl air");
        test::check(test::approx(froude(2.0, 10.0), 2.0 / std::sqrt(98.1), 1e-12), "froude");
    }
    // ── Leftover spot checks: plasma, speeds, spin, orbitals ──
    {
        using namespace np::physics;
        test::check(test::approx(plasma_frequency(1e18), 5.64e10, 1e-2), "plasma frequency");
        test::check(test::approx(debye_length(1e18, 1e4), 6.9e-6, 1e-2), "debye length");
        test::check(test::approx(gyroradius(9.1093837015e-31, 1e6, 1.602176634e-19, 1.0), 5.69e-6, 1e-2), "gyroradius");
        test::check(larmor_power(1.602176634e-19, 1.0) > 0.0, "larmor positive");
        test::check(test::approx(mb_mean_speed(constants::m_e, 300.0),
                                 std::sqrt(8.0 * constants::kB * 300.0 / (constants::pi * constants::m_e)), 1e-12),
                    "mb mean speed");
        test::check(test::approx(thermal_wavelength(constants::m_e, 300.0), 4.3e-9, 1e-2), "thermal wavelength");
        test::check(test::approx(time_dilate(1.0, 0.6 * constants::c), 1.25, 1e-9), "time dilation");
        test::check(test::approx(length_contract(1.0, 0.6 * constants::c), 0.8, 1e-9), "length contraction");
        test::check(test::approx(circular_velocity(5.972e24, 6.771e6), 7672.0, 1e-2), "leo velocity");
        test::check(test::approx(orbital_period(5.972e24, 6.771e6), 5546.0, 1e-2), "leo period");
        const double m = constants::m_e, om = 2e15;
        const double x0 = qm::ho_length(m, om);
        test::check(
            test::approx(qm::ho_psi(0, m, om, 0.0), 1.0 / (std::pow(constants::pi, 0.25) * std::sqrt(x0)), 1e-9),
            "ho ground psi(0)");
        test::check(test::approx(qm::hydrogen_1s_psi(0.0),
                                 1.0 / std::sqrt(constants::pi * std::pow(constants::a0_bohr, 3)), 1e-9),
                    "hydrogen 1s psi(0)");
        test::check(qm::pauli_y()(0, 1) == std::complex<double>(0.0, -1.0), "pauli_y");
        test::check(qm::pauli_z()(0, 0).real() == 1.0 && qm::pauli_z()(1, 1).real() == -1.0, "pauli_z");
    }

    // ── FFT Poisson is a real spectral solve (was a no-op side-effect) ──
    {
        using namespace np::physics;
        // Periodic sine mode: discrete residual of the FFT solution must be
        // ~machine precision in the interior.
        const int n = 17;
        NavierStokes2D ns(n, n);
        np::ndarray<double> rhs(std::vector<int>{n, n});
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
            {
                const double x = static_cast<double>(i) / (n - 1);
                const double y = static_cast<double>(j) / (n - 1);
                rhs(j, i) = std::sin(2 * 3.141592653589793 * x) * std::sin(2 * 3.141592653589793 * y);
            }
        ns.pressure_poisson_fft(rhs);
        const double dx = 1.0 / (n - 1);
        double maxres = 0.0, mean = 0.0;
        for (int j = 1; j < n - 1; ++j)
            for (int i = 1; i < n - 1; ++i)
            {
                const double lap = (ns.state.p(j, i + 1) + ns.state.p(j, i - 1) + ns.state.p(j + 1, i) +
                                    ns.state.p(j - 1, i) - 4 * ns.state.p(j, i)) /
                                   (dx * dx);
                maxres = std::max(maxres, std::abs(lap - rhs(j, i)));
                mean += ns.state.p(j, i);
            }
        test::check(maxres < 1e-8, "fft poisson discrete residual");
        test::check(std::abs(mean) / ((n - 2) * (n - 2)) < 1e-8, "fft poisson zero mean");
        // FFTPoisson solver agrees with the member path.
        FFTPoisson f;
        np::ndarray<double> p2(std::vector<int>{n, n});
        f.solve(p2, rhs, dx, dx, 10);
        double maxdiff = 0.0;
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
                maxdiff = std::max(maxdiff, std::abs(p2(j, i) - ns.state.p(j, i)));
        test::check(maxdiff < 1e-12, "FFTPoisson matches member path");
    }
    // ── lattice_refine gates on vorticity and interpolates ──
    {
        // Quiescent field: below threshold, returned unchanged.
        FluidState calm(8, 8);
        auto same = lattice_refine(calm, 1.0);
        test::check(same.nx == 8 && same.ny == 8, "refine calm unchanged");
        // Linear shear u=y has |vorticity|=1 > thresh: refines 2x, and a
        // linear field interpolates exactly.
        FluidState shear(5, 5);
        for (int j = 0; j < 5; ++j)
            for (int i = 0; i < 5; ++i)
                shear.u(j, i) = static_cast<double>(j) / 4.0;
        auto fine = lattice_refine(shear, 0.5);
        test::check(fine.nx == 9 && fine.ny == 9, "refine doubles grid");
        test::check(std::abs(fine.u(4, 4) - 0.5) < 1e-12, "refine interpolates linear");
        test::check(std::abs(fine.u(8, 8) - 1.0) < 1e-12, "refine preserves endpoint");
        // p-adic unit check is no longer constant-true.
        test::check(!is_padic_unit_Re(0.0), "zero not a unit");
        test::check(!is_padic_unit_Re(10.0), "10 not a 5-adic unit");
        test::check(is_padic_unit_Re(7.0), "7 is a 5-adic unit");
    }
    // ── kinetic_energy_simd matches scalar path ──
    {
        NavierStokes2D ns(9, 9);
        ns.state.u(4, 4) = 2.0;
        ns.state.v(3, 3) = -1.0;
        test::check(std::abs(ns.kinetic_energy_simd() - ns.kinetic_energy()) < 1e-12, "ke simd agrees");
    }

    return test::failures() ? 1 : 0;
}
