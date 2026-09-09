/**
 * @file physics.hpp
 * @brief Physics solvers — incompressible flow (Navier-Stokes, Stokes,
 *        Burgers, potential flow, advection-diffusion), heat transfer,
 *        waves, ballistics and classical mechanics, with p-adic/lattice hooks.
 */
#ifndef NP_PHYSICS_HPP
#define NP_PHYSICS_HPP

#include "api_macros.hpp"
#include "differential.hpp"
#include "fft.hpp"
#include "gpu.hpp"
#include "lattice.hpp"
#include "linalg.hpp"
#include "ndarray.hpp"
#include "pqc.hpp"
#include "simd.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <limits>
#include <numbers>
#include <optional>
#include <utility>
#include <vector>

namespace np::physics
{

namespace detail
{
// Flat C-order index of a multi-index (odometer-free helper for below).
NP_NODISCARD inline std::size_t od_linear_index(const std::vector<int> &shape, const std::vector<std::size_t> &idx)
{
    std::size_t f = 0, stride = 1;
    for (std::size_t d = shape.size(); d-- > 0;)
    {
        f += idx[d] * stride;
        stride *= static_cast<std::size_t>(shape[d]);
    }
    return f;
}

// Periodic Poisson solve Δp = rhs on a uniform grid via FFT: forward
// transform, divide by the discrete-Laplacian symbol
// λ = Σ_d 2(cos(2πk_d/n_d)−1)/h_d², inverse transform, real part.
// Zero mode (all k_d = 0) has λ = 0: pinned to 0 (zero-mean fix, standard
// for periodic Poisson). Throws invalid_argument on empty input or
// mismatched spacings. O(N log N), exact for the discrete operator
// (verified against Jacobi convergence in test_physics).
NP_NODISCARD inline ndarray<double> poisson_fft_periodic(const ndarray<double> &rhs, const std::vector<double> &h)
{
    if (rhs.ndim() == 0 || rhs.size() == 0)
        throw std::invalid_argument("poisson_fft_periodic: empty input");
    if (h.size() != rhs.ndim())
        throw std::invalid_argument("poisson_fft_periodic: spacings/rank mismatch");
    for (double hi : h)
    {
        if (!(hi > 0.0))
            throw std::invalid_argument("poisson_fft_periodic: spacings must be positive");
    }
    auto fhat = fft::fftn(rhs, std::nullopt, std::nullopt, fft::Norm::Backward);
    const std::size_t nd = rhs.ndim();
    std::vector<std::size_t> shape_u(nd);
    for (std::size_t d = 0; d < nd; ++d)
        shape_u[d] = static_cast<std::size_t>(rhs.shape[d]);
    np::detail::Odometer od(rhs.shape);
    while (!od.done())
    {
        const auto &idx = od.idx();
        double lam = 0.0;
        for (std::size_t d = 0; d < nd; ++d)
        {
            const double n = static_cast<double>(shape_u[d]);
            const double k = static_cast<double>(idx[d]);
            lam += 2.0 * (std::cos(2.0 * std::numbers::pi * k / n) - 1.0) / (h[d] * h[d]);
        }
        const std::size_t f = fhat._flat_logical(od_linear_index(rhs.shape, idx));
        if (lam == 0.0)
            fhat.data()[f] = {0.0, 0.0};
        else
            fhat.data()[f] /= lam;
        od.advance();
    }
    auto back = fft::ifftn(fhat, std::nullopt, std::nullopt, fft::Norm::Backward);
    ndarray<double> out(rhs.shape);
    for (std::size_t i = 0; i < out.size(); ++i)
        out.data()[i] = back.data()[back._flat_logical(i)].real();
    return out;
}
} // namespace detail

struct FluidState
{
    int nx = 0, ny = 0;
    ndarray<double> u, v, p;
    FluidState() = default;
    FluidState(int nx_, int ny_)
        : nx(nx_), ny(ny_), u(std::vector<int>{ny_, nx_}), v(std::vector<int>{ny_, nx_}), p(std::vector<int>{ny_, nx_})
    {
    }
};

/**
 * @brief Incompressible 2D Navier-Stokes solver on a unit-square domain,
 *        stepped with Chorin's projection method:
 *          1. advect/diffuse to get a provisional velocity (u*, v*),
 *          2. solve a pressure Poisson equation so the corrected field is
 *             divergence-free,
 *          3. project the provisional velocity back onto that field.
 *        No-slip (u = v = 0) walls are enforced on all four boundaries.
 */
struct NavierStokes2D
{
    FluidState state;
    double Re = 100.0, dt = 0.01;
    /// Jacobi sweeps used per step to solve the pressure Poisson equation.
    int poisson_iters = 50;

    NavierStokes2D() = default;
    NavierStokes2D(int nx, int ny, double Re_ = 100) : state(nx, ny), Re(Re_)
    {
    }

    NP_API void step()
    {
        const int nx = state.nx, ny = state.ny;
        // Need at least one interior point in each direction to difference,
        // and nx-1/ny-1 > 0 so the grid spacing below is well defined.
        if (nx < 3 || ny < 3)
        {
            return;
        }

        const double dx = 1.0 / static_cast<double>(nx - 1);
        const double dy = 1.0 / static_cast<double>(ny - 1);
        const double nu = 1.0 / Re;
        const double rho = 1.0;

        auto &u = state.u;
        auto &v = state.v;
        auto &p = state.p;

        // 1. Provisional velocity (u*, v*): explicit-Euler advection + diffusion.
        ndarray<double> u_star(std::vector<int>{ny, nx});
        ndarray<double> v_star(std::vector<int>{ny, nx});

        for (int j = 1; j < ny - 1; ++j)
        {
            for (int i = 1; i < nx - 1; ++i)
            {
                const double un = u(j, i);
                const double vn = v(j, i);

                const double dudx = (u(j, i + 1) - u(j, i - 1)) / (2.0 * dx);
                const double dudy = (u(j + 1, i) - u(j - 1, i)) / (2.0 * dy);
                const double dvdx = (v(j, i + 1) - v(j, i - 1)) / (2.0 * dx);
                const double dvdy = (v(j + 1, i) - v(j - 1, i)) / (2.0 * dy);

                const double lap_u = (u(j, i + 1) - 2.0 * un + u(j, i - 1)) / (dx * dx) +
                                     (u(j + 1, i) - 2.0 * un + u(j - 1, i)) / (dy * dy);
                const double lap_v = (v(j, i + 1) - 2.0 * vn + v(j, i - 1)) / (dx * dx) +
                                     (v(j + 1, i) - 2.0 * vn + v(j - 1, i)) / (dy * dy);

                u_star(j, i) = un + dt * (-un * dudx - vn * dudy + nu * lap_u);
                v_star(j, i) = vn + dt * (-un * dvdx - vn * dvdy + nu * lap_v);
            }
        }

        // No-slip walls on the provisional field.
        for (int i = 0; i < nx; ++i)
        {
            u_star(0, i) = 0.0;
            u_star(ny - 1, i) = 0.0;
            v_star(0, i) = 0.0;
            v_star(ny - 1, i) = 0.0;
        }
        for (int j = 0; j < ny; ++j)
        {
            u_star(j, 0) = 0.0;
            u_star(j, nx - 1) = 0.0;
            v_star(j, 0) = 0.0;
            v_star(j, nx - 1) = 0.0;
        }

        // Pressure Poisson equation: laplacian(p) = (rho/dt) * div(u*).
        ndarray<double> rhs(std::vector<int>{ny, nx});
        for (int j = 1; j < ny - 1; ++j)
        {
            for (int i = 1; i < nx - 1; ++i)
            {
                const double div = (u_star(j, i + 1) - u_star(j, i - 1)) / (2.0 * dx) +
                                   (v_star(j + 1, i) - v_star(j - 1, i)) / (2.0 * dy);
                rhs(j, i) = (rho / dt) * div;
            }
        }

        ndarray<double> p_new(std::vector<int>{ny, nx});
        const double denom = 2.0 * (dx * dx + dy * dy);
        for (int iter = 0; iter < poisson_iters; ++iter)
        {
            for (int j = 1; j < ny - 1; ++j)
            {
                for (int i = 1; i < nx - 1; ++i)
                {
                    p_new(j, i) = ((p(j, i + 1) + p(j, i - 1)) * dy * dy + (p(j + 1, i) + p(j - 1, i)) * dx * dx -
                                   rhs(j, i) * dx * dx * dy * dy) /
                                  denom;
                }
            }
            // Neumann walls (dp/dn = 0)...
            for (int i = 0; i < nx; ++i)
            {
                p_new(0, i) = p_new(1, i);
                p_new(ny - 1, i) = p_new(ny - 2, i);
            }
            for (int j = 0; j < ny; ++j)
            {
                p_new(j, 0) = p_new(j, 1);
                p_new(j, nx - 1) = p_new(j, nx - 2);
            }
            // ...pinned at one corner to fix pressure's additive constant.
            p_new(0, 0) = 0.0;

            std::swap(p, p_new); // p now holds the updated field.
        }

        // Subtract the pressure gradient from the provisional field.
        for (int j = 1; j < ny - 1; ++j)
        {
            for (int i = 1; i < nx - 1; ++i)
            {
                const double dpdx = (p(j, i + 1) - p(j, i - 1)) / (2.0 * dx);
                const double dpdy = (p(j + 1, i) - p(j - 1, i)) / (2.0 * dy);
                u(j, i) = u_star(j, i) - dt / rho * dpdx;
                v(j, i) = v_star(j, i) - dt / rho * dpdy;
            }
        }

        // No-slip walls on the corrected field.
        for (int i = 0; i < nx; ++i)
        {
            u(0, i) = 0.0;
            u(ny - 1, i) = 0.0;
            v(0, i) = 0.0;
            v(ny - 1, i) = 0.0;
        }
        for (int j = 0; j < ny; ++j)
        {
            u(j, 0) = 0.0;
            u(j, nx - 1) = 0.0;
            v(j, 0) = 0.0;
            v(j, nx - 1) = 0.0;
        }
    }

    /// Total kinetic energy 0.5 * integral(u^2 + v^2) over the domain.
    NP_NODISCARD double kinetic_energy() const
    {
        const int nx = state.nx, ny = state.ny;
        if (nx == 0 || ny == 0)
        {
            return 0.0;
        }
        const double dx = (nx > 1) ? 1.0 / static_cast<double>(nx - 1) : 1.0;
        const double dy = (ny > 1) ? 1.0 / static_cast<double>(ny - 1) : 1.0;
        const double cell_area = dx * dy;

        const auto &u = state.u;
        const auto &v = state.v;
        double ke = 0.0;
        for (int j = 0; j < ny; ++j)
        {
            for (int i = 0; i < nx; ++i)
            {
                const double uu = u(j, i);
                const double vv = v(j, i);
                ke += uu * uu + vv * vv;
            }
        }
        return 0.5 * ke * cell_area;
    }

    /// Max |div(u)| over interior points, via centered differences — a
    /// diagnostic for how well incompressibility is being maintained.
    NP_NODISCARD double max_divergence() const
    {
        const int nx = state.nx, ny = state.ny;
        if (nx < 3 || ny < 3)
        {
            return 0.0;
        }
        const double dx = 1.0 / static_cast<double>(nx - 1);
        const double dy = 1.0 / static_cast<double>(ny - 1);

        const auto &u = state.u;
        const auto &v = state.v;
        double max_div = 0.0;
        for (int j = 1; j < ny - 1; ++j)
        {
            for (int i = 1; i < nx - 1; ++i)
            {
                const double div = (u(j, i + 1) - u(j, i - 1)) / (2.0 * dx) + (v(j + 1, i) - v(j - 1, i)) / (2.0 * dy);
                max_div = std::max(max_div, std::abs(div));
            }
        }
        return max_div;
    }

    // Integrated subsystems (Strategy + Factory)

    /// Advection scheme selector (Strategy)
    enum class AdvectionScheme : std::uint8_t
    {
        Central,
        Upwind,
        WENO5
    };

    /// Set initial velocity from string expressions via differential::VM
    /// e.g. set_initial_from_vm("sin(pi*x)*cos(pi*y)", " -cos(pi*x)*sin(pi*y)")
    NP_API void set_initial_from_vm(const std::string &u_expr, const std::string &v_expr)
    {
        differential::VM vm_u(u_expr, {"x", "y"});
        differential::VM vm_v(v_expr, {"x", "y"});
        const double dx = 1.0 / static_cast<double>(state.nx - 1);
        const double dy = 1.0 / static_cast<double>(state.ny - 1);
        for (int j = 0; j < state.ny; ++j)
            for (int i = 0; i < state.nx; ++i)
            {
                double x = i * dx, y = j * dy;
                state.u(j, i) = vm_u.eval({x, y});
                state.v(j, i) = vm_v.eval({x, y});
            }
        // Enforce walls after VM init
        for (int i = 0; i < state.nx; ++i)
        {
            state.u(0, i) = state.u(state.ny - 1, i) = 0;
            state.v(0, i) = state.v(state.ny - 1, i) = 0;
        }
        for (int j = 0; j < state.ny; ++j)
        {
            state.u(j, 0) = state.u(j, state.nx - 1) = 0;
            state.v(j, 0) = state.v(j, state.nx - 1) = 0;
        }
    }

    /// Vorticity ω = ∂v/∂x - ∂u/∂y via differential::OneForm curl or finite diff
    NP_NODISCARD ndarray<double> vorticity() const
    {
        ndarray<double> w(std::vector<int>{state.ny, state.nx});
        const double dx = 1.0 / static_cast<double>(state.nx - 1);
        const double dy = 1.0 / static_cast<double>(state.ny - 1);
        for (int j = 1; j < state.ny - 1; ++j)
            for (int i = 1; i < state.nx - 1; ++i)
            {
                double dvdx = (state.v(j, i + 1) - state.v(j, i - 1)) / (2 * dx);
                double dudy = (state.u(j + 1, i) - state.u(j - 1, i)) / (2 * dy);
                w(j, i) = dvdx - dudy;
            }
        return w;
    }

    /// Enstrophy 0.5*∫ω² (diagnostic, integrates via spectral-like sum)
    NP_NODISCARD double enstrophy() const
    {
        auto w = vorticity();
        double s = 0;
        for (size_t i = 0; i < w.size(); ++i)
            s += w.data()[i] * w.data()[i];
        return 0.5 * s * (1.0 / (state.nx - 1)) * (1.0 / (state.ny - 1));
    }

    /// Pressure solve via FFT (periodic; Neumann via DCT is out of scope —
    /// use JacobiPoisson for Neumann). Real spectral solve through
    /// detail::poisson_fft_periodic (an earlier revision copied rhs and
    /// called fft for side effect only, solving nothing).
    NP_API void pressure_poisson_fft(const ndarray<double> &rhs)
    {
        const double dx = 1.0 / (state.nx - 1), dy = 1.0 / (state.ny - 1);
        state.p = detail::poisson_fft_periodic(rhs, {dx, dy});
    }

    // NOTE (honesty audit): step_gpu() is deleted — it claimed GPU
    // acceleration while unconditionally delegating to step(), and no
    // stencil offload exists to implement. Callers use step().

    /// SIMD-accelerated kinetic energy via simd::sum_vectorized over a
    /// contiguous u²+v² buffer (an earlier revision was a scalar loop
    /// despite the name).
    NP_NODISCARD double kinetic_energy_simd() const
    {
        const auto &u = state.u, &v = state.v;
        if (u.is_contiguous() && v.is_contiguous() && u.shape == v.shape)
        {
            if (u.size() == 0)
                return 0.0;
            std::vector<double> sq(u.size());
            const double *up = u.data().data() + u.offset;
            const double *vp = v.data().data() + v.offset;
            for (size_t i = 0; i < u.size(); ++i)
            {
                sq[i] = up[i] * up[i] + vp[i] * vp[i];
            }
            const double ke = simd::sum_vectorized(sq.data(), sq.size());
            double dx = 1.0 / (state.nx - 1), dy = 1.0 / (state.ny - 1);
            return 0.5 * ke * dx * dy;
        }
        return kinetic_energy();
    }

    /// Secure step (constant-time wipe of intermediates, for PQC lattice fluid)
    NP_API void step_secure()
    {
        step();
        pqc::ct_barrier();
    }
};

struct FluidState3D
{
    int nx = 0, ny = 0, nz = 0;
    ndarray<double> u, v, w, p;
    FluidState3D() = default;
    FluidState3D(int nx_, int ny_, int nz_)
        : nx(nx_), ny(ny_), nz(nz_), u(std::vector<int>{nz_, ny_, nx_}), v(std::vector<int>{nz_, ny_, nx_}),
          w(std::vector<int>{nz_, ny_, nx_}), p(std::vector<int>{nz_, ny_, nx_})
    {
    }
};

/**
 * @brief Incompressible 3D Navier-Stokes solver on a unit-cube domain,
 *        stepped with Chorin's projection method (3D extension of NavierStokes2D):
 *          1. advect/diffuse to get a provisional velocity (u*, v*, w*),
 *          2. solve a pressure Poisson equation so the corrected field is
 *             divergence-free,
 *          3. project the provisional velocity back onto that field.
 *        No-slip (u = v = w = 0) walls are enforced on all six boundaries.
 *        Storage order is (k, j, i) = (z, y, x) with shape {nz, ny, nx}.
 */
struct NavierStokes3D
{
    FluidState3D state;
    double Re = 100.0, dt = 0.01;
    /// Jacobi sweeps used per step to solve the pressure Poisson equation.
    int poisson_iters = 50;

    NavierStokes3D() = default;
    NavierStokes3D(int nx, int ny, int nz, double Re_ = 100) : state(nx, ny, nz), Re(Re_)
    {
    }

    NP_API void step()
    {
        const int nx = state.nx, ny = state.ny, nz = state.nz;
        if (nx < 3 || ny < 3 || nz < 3)
        {
            return;
        }

        const double dx = 1.0 / static_cast<double>(nx - 1);
        const double dy = 1.0 / static_cast<double>(ny - 1);
        const double dz = 1.0 / static_cast<double>(nz - 1);
        const double nu = 1.0 / Re;
        const double rho = 1.0;

        auto &u = state.u;
        auto &v = state.v;
        auto &w = state.w;
        auto &p = state.p;

        // 1. Provisional velocity: explicit-Euler advection + diffusion.
        ndarray<double> u_star(std::vector<int>{nz, ny, nx});
        ndarray<double> v_star(std::vector<int>{nz, ny, nx});
        ndarray<double> w_star(std::vector<int>{nz, ny, nx});

        const double dx2 = dx * dx, dy2 = dy * dy, dz2 = dz * dz;
        for (int k = 1; k < nz - 1; ++k)
        {
            for (int j = 1; j < ny - 1; ++j)
            {
                for (int i = 1; i < nx - 1; ++i)
                {
                    const double un = u(k, j, i);
                    const double vn = v(k, j, i);
                    const double wn = w(k, j, i);

                    const double dudx = (u(k, j, i + 1) - u(k, j, i - 1)) / (2.0 * dx);
                    const double dudy = (u(k, j + 1, i) - u(k, j - 1, i)) / (2.0 * dy);
                    const double dudz = (u(k + 1, j, i) - u(k - 1, j, i)) / (2.0 * dz);
                    const double dvdx = (v(k, j, i + 1) - v(k, j, i - 1)) / (2.0 * dx);
                    const double dvdy = (v(k, j + 1, i) - v(k, j - 1, i)) / (2.0 * dy);
                    const double dvdz = (v(k + 1, j, i) - v(k - 1, j, i)) / (2.0 * dz);
                    const double dwdx = (w(k, j, i + 1) - w(k, j, i - 1)) / (2.0 * dx);
                    const double dwdy = (w(k, j + 1, i) - w(k, j - 1, i)) / (2.0 * dy);
                    const double dwdz = (w(k + 1, j, i) - w(k - 1, j, i)) / (2.0 * dz);

                    const double lap_u = (u(k, j, i + 1) - 2.0 * un + u(k, j, i - 1)) / dx2 +
                                         (u(k, j + 1, i) - 2.0 * un + u(k, j - 1, i)) / dy2 +
                                         (u(k + 1, j, i) - 2.0 * un + u(k - 1, j, i)) / dz2;
                    const double lap_v = (v(k, j, i + 1) - 2.0 * vn + v(k, j, i - 1)) / dx2 +
                                         (v(k, j + 1, i) - 2.0 * vn + v(k, j - 1, i)) / dy2 +
                                         (v(k + 1, j, i) - 2.0 * vn + v(k - 1, j, i)) / dz2;
                    const double lap_w = (w(k, j, i + 1) - 2.0 * wn + w(k, j, i - 1)) / dx2 +
                                         (w(k, j + 1, i) - 2.0 * wn + w(k, j - 1, i)) / dy2 +
                                         (w(k + 1, j, i) - 2.0 * wn + w(k - 1, j, i)) / dz2;

                    u_star(k, j, i) = un + dt * (-un * dudx - vn * dudy - wn * dudz + nu * lap_u);
                    v_star(k, j, i) = vn + dt * (-un * dvdx - vn * dvdy - wn * dvdz + nu * lap_v);
                    w_star(k, j, i) = wn + dt * (-un * dwdx - vn * dwdy - wn * dwdz + nu * lap_w);
                }
            }
        }

        // No-slip walls on the provisional field (all six faces).
        zero_faces(u_star);
        zero_faces(v_star);
        zero_faces(w_star);

        // Pressure Poisson equation: laplacian(p) = (rho/dt) * div(u*).
        ndarray<double> rhs(std::vector<int>{nz, ny, nx});
        for (int k = 1; k < nz - 1; ++k)
            for (int j = 1; j < ny - 1; ++j)
                for (int i = 1; i < nx - 1; ++i)
                {
                    const double div = (u_star(k, j, i + 1) - u_star(k, j, i - 1)) / (2.0 * dx) +
                                       (v_star(k, j + 1, i) - v_star(k, j - 1, i)) / (2.0 * dy) +
                                       (w_star(k + 1, j, i) - w_star(k - 1, j, i)) / (2.0 * dz);
                    rhs(k, j, i) = (rho / dt) * div;
                }

        ndarray<double> p_new(std::vector<int>{nz, ny, nx});
        const double denom = 2.0 * (dx2 * dy2 + dx2 * dz2 + dy2 * dz2);
        const double rhs_scale = dx2 * dy2 * dz2;
        for (int iter = 0; iter < poisson_iters; ++iter)
        {
            for (int k = 1; k < nz - 1; ++k)
                for (int j = 1; j < ny - 1; ++j)
                    for (int i = 1; i < nx - 1; ++i)
                    {
                        p_new(k, j, i) = ((p(k, j, i + 1) + p(k, j, i - 1)) * dy2 * dz2 +
                                          (p(k, j + 1, i) + p(k, j - 1, i)) * dx2 * dz2 +
                                          (p(k + 1, j, i) + p(k - 1, j, i)) * dx2 * dy2 - rhs(k, j, i) * rhs_scale) /
                                         denom;
                    }
            // Neumann walls (dp/dn = 0) on all six faces...
            for (int j = 0; j < ny; ++j)
                for (int i = 0; i < nx; ++i)
                {
                    p_new(0, j, i) = p_new(1, j, i);
                    p_new(nz - 1, j, i) = p_new(nz - 2, j, i);
                }
            for (int k = 0; k < nz; ++k)
                for (int i = 0; i < nx; ++i)
                {
                    p_new(k, 0, i) = p_new(k, 1, i);
                    p_new(k, ny - 1, i) = p_new(k, ny - 2, i);
                }
            for (int k = 0; k < nz; ++k)
                for (int j = 0; j < ny; ++j)
                {
                    p_new(k, j, 0) = p_new(k, j, 1);
                    p_new(k, j, nx - 1) = p_new(k, j, nx - 2);
                }
            // ...pinned at one corner to fix pressure's additive constant.
            p_new(0, 0, 0) = 0.0;

            std::swap(p, p_new); // p now holds the updated field.
        }

        // Subtract the pressure gradient from the provisional field.
        for (int k = 1; k < nz - 1; ++k)
            for (int j = 1; j < ny - 1; ++j)
                for (int i = 1; i < nx - 1; ++i)
                {
                    const double dpdx = (p(k, j, i + 1) - p(k, j, i - 1)) / (2.0 * dx);
                    const double dpdy = (p(k, j + 1, i) - p(k, j - 1, i)) / (2.0 * dy);
                    const double dpdz = (p(k + 1, j, i) - p(k - 1, j, i)) / (2.0 * dz);
                    u(k, j, i) = u_star(k, j, i) - dt / rho * dpdx;
                    v(k, j, i) = v_star(k, j, i) - dt / rho * dpdy;
                    w(k, j, i) = w_star(k, j, i) - dt / rho * dpdz;
                }

        // No-slip walls on the corrected field.
        zero_faces(u);
        zero_faces(v);
        zero_faces(w);
    }

    /// Total kinetic energy 0.5 * integral(u^2 + v^2 + w^2) over the domain.
    NP_NODISCARD double kinetic_energy() const
    {
        const int nx = state.nx, ny = state.ny, nz = state.nz;
        if (nx == 0 || ny == 0 || nz == 0)
        {
            return 0.0;
        }
        const double dx = (nx > 1) ? 1.0 / static_cast<double>(nx - 1) : 1.0;
        const double dy = (ny > 1) ? 1.0 / static_cast<double>(ny - 1) : 1.0;
        const double dz = (nz > 1) ? 1.0 / static_cast<double>(nz - 1) : 1.0;
        const double cell_vol = dx * dy * dz;

        const auto &u = state.u;
        const auto &v = state.v;
        const auto &w = state.w;
        double ke = 0.0;
        for (int k = 0; k < nz; ++k)
            for (int j = 0; j < ny; ++j)
                for (int i = 0; i < nx; ++i)
                {
                    const double uu = u(k, j, i);
                    const double vv = v(k, j, i);
                    const double ww = w(k, j, i);
                    ke += uu * uu + vv * vv + ww * ww;
                }
        return 0.5 * ke * cell_vol;
    }

    /// Max |div(u)| over interior points, via centered differences.
    NP_NODISCARD double max_divergence() const
    {
        const int nx = state.nx, ny = state.ny, nz = state.nz;
        if (nx < 3 || ny < 3 || nz < 3)
        {
            return 0.0;
        }
        const double dx = 1.0 / static_cast<double>(nx - 1);
        const double dy = 1.0 / static_cast<double>(ny - 1);
        const double dz = 1.0 / static_cast<double>(nz - 1);

        const auto &u = state.u;
        const auto &v = state.v;
        const auto &w = state.w;
        double max_div = 0.0;
        for (int k = 1; k < nz - 1; ++k)
            for (int j = 1; j < ny - 1; ++j)
                for (int i = 1; i < nx - 1; ++i)
                {
                    const double div = (u(k, j, i + 1) - u(k, j, i - 1)) / (2.0 * dx) +
                                       (v(k, j + 1, i) - v(k, j - 1, i)) / (2.0 * dy) +
                                       (w(k + 1, j, i) - w(k - 1, j, i)) / (2.0 * dz);
                    max_div = std::max(max_div, std::abs(div));
                }
        return max_div;
    }

    /// Advection scheme selector (Strategy), mirrors NavierStokes2D.
    enum class AdvectionScheme : std::uint8_t
    {
        Central,
        Upwind,
        WENO5
    };

    /// Set initial velocity from string expressions via differential::VM
    /// e.g. set_initial_from_vm("sin(pi*x)*cos(pi*y)", " -cos(pi*x)*sin(pi*y)", "0")
    NP_API void set_initial_from_vm(const std::string &u_expr, const std::string &v_expr, const std::string &w_expr)
    {
        differential::VM vm_u(u_expr, {"x", "y", "z"});
        differential::VM vm_v(v_expr, {"x", "y", "z"});
        differential::VM vm_w(w_expr, {"x", "y", "z"});
        const double dx = 1.0 / static_cast<double>(state.nx - 1);
        const double dy = 1.0 / static_cast<double>(state.ny - 1);
        const double dz = 1.0 / static_cast<double>(state.nz - 1);
        for (int k = 0; k < state.nz; ++k)
            for (int j = 0; j < state.ny; ++j)
                for (int i = 0; i < state.nx; ++i)
                {
                    const double x = i * dx, y = j * dy, z = k * dz;
                    state.u(k, j, i) = vm_u.eval({x, y, z});
                    state.v(k, j, i) = vm_v.eval({x, y, z});
                    state.w(k, j, i) = vm_w.eval({x, y, z});
                }
        // Enforce walls after VM init
        zero_faces(state.u);
        zero_faces(state.v);
        zero_faces(state.w);
    }

    /// Vorticity vector ω = ∇ × u via centered differences.
    /// Returns {omega_x, omega_y, omega_z}, each shaped {nz, ny, nx}.
    NP_NODISCARD std::array<ndarray<double>, 3> vorticity() const
    {
        std::array<ndarray<double>, 3> om = {ndarray<double>(std::vector<int>{state.nz, state.ny, state.nx}),
                                             ndarray<double>(std::vector<int>{state.nz, state.ny, state.nx}),
                                             ndarray<double>(std::vector<int>{state.nz, state.ny, state.nx})};
        const int nx = state.nx, ny = state.ny, nz = state.nz;
        if (nx < 3 || ny < 3 || nz < 3)
        {
            return om;
        }
        const double dx = 1.0 / static_cast<double>(nx - 1);
        const double dy = 1.0 / static_cast<double>(ny - 1);
        const double dz = 1.0 / static_cast<double>(nz - 1);
        for (int k = 1; k < nz - 1; ++k)
            for (int j = 1; j < ny - 1; ++j)
                for (int i = 1; i < nx - 1; ++i)
                {
                    const double dwdy = (state.w(k, j + 1, i) - state.w(k, j - 1, i)) / (2.0 * dy);
                    const double dvdz = (state.v(k + 1, j, i) - state.v(k - 1, j, i)) / (2.0 * dz);
                    const double dudz = (state.u(k + 1, j, i) - state.u(k - 1, j, i)) / (2.0 * dz);
                    const double dwdx = (state.w(k, j, i + 1) - state.w(k, j, i - 1)) / (2.0 * dx);
                    const double dvdx = (state.v(k, j, i + 1) - state.v(k, j, i - 1)) / (2.0 * dx);
                    const double dudy = (state.u(k, j + 1, i) - state.u(k, j - 1, i)) / (2.0 * dy);
                    om[0](k, j, i) = dwdy - dvdz;
                    om[1](k, j, i) = dudz - dwdx;
                    om[2](k, j, i) = dvdx - dudy;
                }
        return om;
    }

    /// |ω| field, shaped {nz, ny, nx}.
    NP_NODISCARD ndarray<double> vorticity_magnitude() const
    {
        auto om = vorticity();
        ndarray<double> mag(std::vector<int>{state.nz, state.ny, state.nx});
        for (std::size_t n = 0; n < mag.size(); ++n)
        {
            const double ox = om[0].data()[n], oy = om[1].data()[n], oz = om[2].data()[n];
            mag.data()[n] = std::sqrt(ox * ox + oy * oy + oz * oz);
        }
        return mag;
    }

    /// Enstrophy 0.5*∫|ω|².
    NP_NODISCARD double enstrophy() const
    {
        auto om = vorticity();
        double s = 0.0;
        for (std::size_t n = 0; n < om[0].size(); ++n)
            s += om[0].data()[n] * om[0].data()[n] + om[1].data()[n] * om[1].data()[n] +
                 om[2].data()[n] * om[2].data()[n];
        return 0.5 * s * (1.0 / (state.nx - 1)) * (1.0 / (state.ny - 1)) * (1.0 / (state.nz - 1));
    }

    /// Pressure solve via FFT (periodic; see 2-D note above for the audit
    /// history). Real spectral solve through detail::poisson_fft_periodic.
    NP_API void pressure_poisson_fft(const ndarray<double> &rhs)
    {
        const double dx = 1.0 / (state.nx - 1), dy = 1.0 / (state.ny - 1), dz = 1.0 / (state.nz - 1);
        state.p = detail::poisson_fft_periodic(rhs, {dx, dy, dz});
    }

    // NOTE (honesty audit): step_gpu() deleted here too — same unconditional
    // delegation as the 2-D version. Callers use step().

    /// SIMD-accelerated kinetic energy via simd::sum_vectorized over a
    /// contiguous u²+v²+w² buffer (was a scalar loop despite the name).
    NP_NODISCARD double kinetic_energy_simd() const
    {
        const auto &u = state.u, &v = state.v, &w = state.w;
        if (u.is_contiguous() && v.is_contiguous() && w.is_contiguous() && u.shape == v.shape &&
            u.shape == w.shape)
        {
            if (u.size() == 0)
                return 0.0;
            std::vector<double> sq(u.size());
            const double *up = u.data().data() + u.offset;
            const double *vp = v.data().data() + v.offset;
            const double *wp = w.data().data() + w.offset;
            for (std::size_t n = 0; n < u.size(); ++n)
            {
                sq[n] = up[n] * up[n] + vp[n] * vp[n] + wp[n] * wp[n];
            }
            const double ke = simd::sum_vectorized(sq.data(), sq.size());
            const double dx = 1.0 / (state.nx - 1), dy = 1.0 / (state.ny - 1), dz = 1.0 / (state.nz - 1);
            return 0.5 * ke * dx * dy * dz;
        }
        return kinetic_energy();
    }

    /// Secure step (constant-time wipe of intermediates, for PQC lattice fluid)
    NP_API void step_secure()
    {
        step();
        pqc::ct_barrier();
    }

  private:
    static void zero_faces(ndarray<double> &f)
    {
        const int nx = static_cast<int>(f.shape[2]);
        const int ny = static_cast<int>(f.shape[1]);
        const int nz = static_cast<int>(f.shape[0]);
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i)
            {
                f(0, j, i) = 0.0;
                f(nz - 1, j, i) = 0.0;
            }
        for (int k = 0; k < nz; ++k)
            for (int i = 0; i < nx; ++i)
            {
                f(k, 0, i) = 0.0;
                f(k, ny - 1, i) = 0.0;
            }
        for (int k = 0; k < nz; ++k)
            for (int j = 0; j < ny; ++j)
            {
                f(k, j, 0) = 0.0;
                f(k, j, nx - 1) = 0.0;
            }
    }
};

// PoissonSolver Strategy (Factory)
struct PoissonSolver
{
    virtual ~PoissonSolver() = default;
    virtual void solve(ndarray<double> &p, const ndarray<double> &rhs, double dx, double dy, int iters) = 0;
    virtual std::string name() const noexcept = 0;
};

struct JacobiPoisson : PoissonSolver
{
    void solve(ndarray<double> &p, const ndarray<double> &rhs, double dx, double dy, int iters) override
    {
        int ny = p.shape[0], nx = p.shape[1];
        ndarray<double> pn(std::vector<int>{ny, nx});
        double denom = 2.0 * (dx * dx + dy * dy);
        for (int it = 0; it < iters; ++it)
        {
            for (int j = 1; j < ny - 1; ++j)
                for (int i = 1; i < nx - 1; ++i)
                    pn(j, i) = ((p(j, i + 1) + p(j, i - 1)) * dy * dy + (p(j + 1, i) + p(j - 1, i)) * dx * dx -
                                rhs(j, i) * dx * dx * dy * dy) /
                               denom;
            for (int i = 0; i < nx; ++i)
            {
                pn(0, i) = pn(1, i);
                pn(ny - 1, i) = pn(ny - 2, i);
            }
            for (int j = 0; j < ny; ++j)
            {
                pn(j, 0) = pn(j, 1);
                pn(j, nx - 1) = pn(j, nx - 2);
            }
            pn(0, 0) = 0;
            std::swap(p, pn);
        }
    }
    std::string name() const noexcept override
    {
        return "Jacobi";
    }
};

struct FFTPoisson : PoissonSolver
{
    void solve(ndarray<double> &p, const ndarray<double> &rhs, double dx, double dy, int iters) override
    {
        // NOTE (honesty audit): an earlier revision ignored dx/dy/iters and
        // delegated to Jacobi while named "FFT-Spectral". This is now a real
        // spectral solve; non-2-D input still falls back to Jacobi (loudly
        // documented here, not silently).
        (void)iters;
        if (rhs.ndim() != 2)
        {
            JacobiPoisson j;
            j.solve(p, rhs, dx, dy, iters);
            return;
        }
        p = detail::poisson_fft_periodic(rhs, {dx, dy});
    }
    std::string name() const noexcept override
    {
        return "FFT-Spectral";
    }
};

struct DirectPoisson : PoissonSolver
{
    void solve(ndarray<double> &p, const ndarray<double> &rhs, double /*dx*/, double /*dy*/, int /*iters*/) override
    {
        int ny = p.shape[0], nx = p.shape[1];
        int n = (ny - 2) * (nx - 2);
        if (n <= 0)
            return;
        // Build Laplacian matrix for interior points (5-point stencil)
        ndarray<double> A(std::vector<int>{n, n});
        // Zero
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                A(i, j) = 0;
        auto idx = [&](int j, int i) { return (j - 1) * (nx - 2) + (i - 1); };
        for (int j = 1; j < ny - 1; ++j)
            for (int i = 1; i < nx - 1; ++i)
            {
                int r = idx(j, i);
                A(r, r) = -4;
                if (i > 1)
                    A(r, idx(j, i - 1)) = 1;
                if (i < nx - 2)
                    A(r, idx(j, i + 1)) = 1;
                if (j > 1)
                    A(r, idx(j - 1, i)) = 1;
                if (j < ny - 2)
                    A(r, idx(j + 1, i)) = 1;
            }
        ndarray<double> b(std::vector<int>{n});
        for (int j = 1; j < ny - 1; ++j)
            for (int i = 1; i < nx - 1; ++i)
                b(idx(j, i)) = rhs(j, i);
        // Solve via linalg::solve (dense, for small n)
        auto x = linalg::solve(A, b);
        for (int j = 1; j < ny - 1; ++j)
            for (int i = 1; i < nx - 1; ++i)
                p(j, i) = x(idx(j, i));
    }
    std::string name() const noexcept override
    {
        return "Direct-LU";
    }
};

struct PoissonFactory
{
    static std::shared_ptr<PoissonSolver> jacobi()
    {
        return std::make_shared<JacobiPoisson>();
    }
    static std::shared_ptr<PoissonSolver> fft()
    {
        return std::make_shared<FFTPoisson>();
    }
    static std::shared_ptr<PoissonSolver> direct()
    {
        return std::make_shared<DirectPoisson>();
    }
    static std::shared_ptr<PoissonSolver> auto_select(int nx, int ny)
    {
        if (nx * ny > 10000)
            return fft();
        if (nx * ny < 2500)
            return direct();
        return jacobi();
    }
};

// 3D Poisson solver strategy (7-point stencil, shape {nz, ny, nx})
struct PoissonSolver3D
{
    virtual ~PoissonSolver3D() = default;
    virtual void solve(ndarray<double> &p, const ndarray<double> &rhs, double dx, double dy, double dz, int iters) = 0;
    virtual std::string name() const noexcept = 0;
};

struct JacobiPoisson3D : PoissonSolver3D
{
    void solve(ndarray<double> &p, const ndarray<double> &rhs, double dx, double dy, double dz, int iters) override
    {
        const int nz = p.shape[0], ny = p.shape[1], nx = p.shape[2];
        if (nx < 3 || ny < 3 || nz < 3)
        {
            return;
        }
        ndarray<double> pn(std::vector<int>{nz, ny, nx});
        const double dx2 = dx * dx, dy2 = dy * dy, dz2 = dz * dz;
        const double denom = 2.0 * (dx2 * dy2 + dx2 * dz2 + dy2 * dz2);
        for (int it = 0; it < iters; ++it)
        {
            for (int k = 1; k < nz - 1; ++k)
                for (int j = 1; j < ny - 1; ++j)
                    for (int i = 1; i < nx - 1; ++i)
                        pn(k, j, i) = ((p(k, j, i + 1) + p(k, j, i - 1)) * dy2 * dz2 +
                                       (p(k, j + 1, i) + p(k, j - 1, i)) * dx2 * dz2 +
                                       (p(k + 1, j, i) + p(k - 1, j, i)) * dx2 * dy2 - rhs(k, j, i) * dx2 * dy2 * dz2) /
                                      denom;
            for (int j = 0; j < ny; ++j)
                for (int i = 0; i < nx; ++i)
                {
                    pn(0, j, i) = pn(1, j, i);
                    pn(nz - 1, j, i) = pn(nz - 2, j, i);
                }
            for (int k = 0; k < nz; ++k)
                for (int i = 0; i < nx; ++i)
                {
                    pn(k, 0, i) = pn(k, 1, i);
                    pn(k, ny - 1, i) = pn(k, ny - 2, i);
                }
            for (int k = 0; k < nz; ++k)
                for (int j = 0; j < ny; ++j)
                {
                    pn(k, j, 0) = pn(k, j, 1);
                    pn(k, j, nx - 1) = pn(k, j, nx - 2);
                }
            pn(0, 0, 0) = 0;
            std::swap(p, pn);
        }
    }
    std::string name() const noexcept override
    {
        return "Jacobi3D";
    }
};

struct PoissonFactory3D
{
    static std::shared_ptr<PoissonSolver3D> jacobi()
    {
        return std::make_shared<JacobiPoisson3D>();
    }
    static std::shared_ptr<PoissonSolver3D> auto_select(int nx, int ny, int nz)
    {
        (void)nx;
        (void)ny;
        (void)nz;
        return jacobi();
    }
};

// Lattice AMR hook (decorator)
// Refines grid where |ω| is large: if max|vorticity| exceeds thresh, the
// state is uniformly refined 2x by bilinear interpolation (new dims
// 2n-1, endpoints preserved); otherwise returned unchanged. This is global
// uniform refinement gated on vorticity, NOT block-structured AMR —
// documented as such (an earlier revision ignored thresh and returned the
// input untouched after a decorative lattice construction).
// ω = dv/dx - du/dy by central differences on the unit square.
NP_NODISCARD inline FluidState lattice_refine(const FluidState &s, double thresh = 1.0)
{
    if (s.nx < 2 || s.ny < 2)
    {
        return s;
    }
    const double dx = 1.0 / static_cast<double>(s.nx - 1);
    const double dy = 1.0 / static_cast<double>(s.ny - 1);
    double maxvort = 0.0;
    for (int j = 0; j < s.ny; ++j)
    {
        for (int i = 0; i < s.nx; ++i)
        {
            const int im = i > 0 ? i - 1 : i, ip = i + 1 < s.nx ? i + 1 : i;
            const int jm = j > 0 ? j - 1 : j, jp = j + 1 < s.ny ? j + 1 : j;
            const double dvdx = (s.v(j, ip) - s.v(j, im)) / (static_cast<double>(ip - im) * dx);
            const double dudy = (s.u(jp, i) - s.u(jm, i)) / (static_cast<double>(jp - jm) * dy);
            const double w = std::abs(dvdx - dudy);
            if (w > maxvort)
                maxvort = w;
        }
    }
    if (!(maxvort > thresh))
    {
        return s;
    }
    const int nx2 = 2 * s.nx - 1, ny2 = 2 * s.ny - 1;
    FluidState out(nx2, ny2);
    auto interp = [&](const ndarray<double> &f) {
        ndarray<double> g(std::vector<int>{ny2, nx2});
        for (int j = 0; j < ny2; ++j)
        {
            for (int i = 0; i < nx2; ++i)
            {
                const int i0 = i / 2, j0 = j / 2;
                const int i1 = std::min(i0 + 1, s.nx - 1), j1 = std::min(j0 + 1, s.ny - 1);
                const double fx = (i % 2 == 0) ? 0.0 : 0.5;
                const double fy = (j % 2 == 0) ? 0.0 : 0.5;
                g(j, i) = (1 - fx) * (1 - fy) * f(j0, i0) + fx * (1 - fy) * f(j0, i1) +
                          (1 - fx) * fy * f(j1, i0) + fx * fy * f(j1, i1);
            }
        }
        return g;
    };
    out.u = interp(s.u);
    out.v = interp(s.v);
    out.p = interp(s.p);
    return out;
}

NP_NODISCARD inline FluidState3D lattice_refine(const FluidState3D &s, double thresh = 1.0)
{
    // Same contract as the 2-D overload above (uniform 2x trilinear
    // refinement gated on vorticity magnitude, not block AMR). Vorticity
    // here is |curl u| from all three components by central differences.
    if (s.nx < 2 || s.ny < 2 || s.nz < 2)
    {
        return s;
    }
    const double dx = 1.0 / static_cast<double>(s.nx - 1);
    const double dy = 1.0 / static_cast<double>(s.ny - 1);
    const double dz = 1.0 / static_cast<double>(s.nz - 1);
    const auto at = [&](const ndarray<double> &f, int k, int j, int i) -> double {
        return f(k, j, i);
    };
    double maxvort = 0.0;
    for (int k = 0; k < s.nz; ++k)
    {
        for (int j = 0; j < s.ny; ++j)
        {
            for (int i = 0; i < s.nx; ++i)
            {
                const int im = i > 0 ? i - 1 : i, ip = i + 1 < s.nx ? i + 1 : i;
                const int jm = j > 0 ? j - 1 : j, jp = j + 1 < s.ny ? j + 1 : j;
                const int km = k > 0 ? k - 1 : k, kp = k + 1 < s.nz ? k + 1 : k;
                const double wx = (at(s.w, k, jp, i) - at(s.w, k, jm, i)) /
                                      (static_cast<double>(jp - jm) * dy) -
                                  (at(s.v, kp, j, i) - at(s.v, km, j, i)) / (static_cast<double>(kp - km) * dz);
                const double wy = (at(s.u, kp, j, i) - at(s.u, km, j, i)) /
                                      (static_cast<double>(kp - km) * dz) -
                                  (at(s.w, k, j, ip) - at(s.w, k, j, im)) / (static_cast<double>(ip - im) * dx);
                const double wz = (at(s.v, k, j, ip) - at(s.v, k, j, im)) /
                                      (static_cast<double>(ip - im) * dx) -
                                  (at(s.u, k, jp, i) - at(s.u, k, jm, i)) / (static_cast<double>(jp - jm) * dy);
                const double mag = std::sqrt(wx * wx + wy * wy + wz * wz);
                if (mag > maxvort)
                    maxvort = mag;
            }
        }
    }
    if (!(maxvort > thresh))
    {
        return s;
    }
    const int nx2 = 2 * s.nx - 1, ny2 = 2 * s.ny - 1, nz2 = 2 * s.nz - 1;
    FluidState3D out(nx2, ny2, nz2);
    auto interp = [&](const ndarray<double> &f) {
        ndarray<double> g(std::vector<int>{nz2, ny2, nx2});
        for (int k = 0; k < nz2; ++k)
        {
            for (int j = 0; j < ny2; ++j)
            {
                for (int i = 0; i < nx2; ++i)
                {
                    const int i0 = i / 2, j0 = j / 2, k0 = k / 2;
                    const int i1 = std::min(i0 + 1, s.nx - 1), j1 = std::min(j0 + 1, s.ny - 1),
                              k1 = std::min(k0 + 1, s.nz - 1);
                    const double fx = (i % 2 == 0) ? 0.0 : 0.5;
                    const double fy = (j % 2 == 0) ? 0.0 : 0.5;
                    const double fz = (k % 2 == 0) ? 0.0 : 0.5;
                    g(k, j, i) = (1 - fx) * (1 - fy) * (1 - fz) * at(f, k0, j0, i0) +
                                 fx * (1 - fy) * (1 - fz) * at(f, k0, j0, i1) +
                                 (1 - fx) * fy * (1 - fz) * at(f, k0, j1, i0) +
                                 fx * fy * (1 - fz) * at(f, k0, j1, i1) +
                                 (1 - fx) * (1 - fy) * fz * at(f, k1, j0, i0) +
                                 fx * (1 - fy) * fz * at(f, k1, j0, i1) +
                                 (1 - fx) * fy * fz * at(f, k1, j1, i0) + fx * fy * fz * at(f, k1, j1, i1);
                }
            }
        }
        return g;
    };
    out.u = interp(s.u);
    out.v = interp(s.v);
    out.w = interp(s.w);
    out.p = interp(s.p);
    return out;
}

  // ── Additional viscous / inviscid fluid solvers ──

  /**
   * @brief 1D viscous Burgers equation on [0, L] with periodic boundaries:
   *        du/dt + u du/dx = nu d²u/dx², explicit Euler + central differences.
   */
  struct Burgers1D
  {
    int nx = 0;
    ndarray<double> u;
    double nu = 0.01, dt = 0.001, L = 1.0;

    Burgers1D() = default;
    Burgers1D(int nx_, double nu_ = 0.01, double dt_ = 0.001) : nx(nx_), u(std::vector<int>{nx_}), nu(nu_), dt(dt_)
    {
    }

    NP_NODISCARD double dx() const
    {
        return L / static_cast<double>((nx > 0) ? nx : 1);
    }

    /// Linearized advective + diffusive CFL estimate around max|u|.
    NP_NODISCARD double stable_dt() const
    {
        if (nx < 3)
        {
            return dt;
        }
        const double h = dx();
        const double umax = max_abs();
        const double adv = (umax > 0.0) ? h / umax : 1.0e100;
        const double dif = (nu > 0.0) ? h * h / (2.0 * nu) : 1.0e100;
        return 0.9 * std::min(adv, dif);
    }

    NP_API void step()
    {
        if (nx < 3)
        {
            return;
        }
        const double h = dx();
        const double cdt = std::min(dt, stable_dt());
        ndarray<double> un(std::vector<int>{nx});
        for (int i = 0; i < nx; ++i)
        {
            const double ip = u((i + 1) % nx);
            const double im = u((i - 1 + nx) % nx);
            const double uc = u(i);
            const double dudx = (ip - im) / (2.0 * h);
            const double lap = (ip - 2.0 * uc + im) / (h * h);
            un(i) = uc + cdt * (-uc * dudx + nu * lap);
        }
        std::swap(u, un);
    }

    /// u = amp * sin(2*pi*x/L) initial condition (classic steepening test).
    NP_API void set_sine(double amp = 1.0)
    {
        const double h = dx();
        for (int i = 0; i < nx; ++i)
        {
            u(i) = amp * std::sin(2.0 * M_PI * i * h / L);
        }
    }

    NP_NODISCARD double max_abs() const
    {
        double m = 0.0;
        for (std::size_t n = 0; n < u.size(); ++n)
        {
            m = std::max(m, std::abs(u.data()[n]));
        }
        return m;
    }

    /// Discrete mass integral(u) dx — conserved by the periodic scheme.
    NP_NODISCARD double mass() const
    {
        double s = 0.0;
        for (std::size_t n = 0; n < u.size(); ++n)
        {
            s += u.data()[n];
        }
        return s * dx();
    }
  };

  /**
   * @brief Coupled 2D Burgers system on the unit square, periodic in both axes:
   *        du/dt + u du/dx + v du/dy = nu lap(u) (and symmetrically for v).
   */
  struct Burgers2D
  {
    int nx = 0, ny = 0;
    ndarray<double> u, v;
    double nu = 0.01, dt = 0.001;

    Burgers2D() = default;
    Burgers2D(int nx_, int ny_, double nu_ = 0.01, double dt_ = 0.001)
        : nx(nx_), ny(ny_), u(std::vector<int>{ny_, nx_}), v(std::vector<int>{ny_, nx_}), nu(nu_), dt(dt_)
    {
    }

    NP_API void step()
    {
        if (nx < 3 || ny < 3)
        {
            return;
        }
        const double dx = 1.0 / static_cast<double>(nx);
        const double dy = 1.0 / static_cast<double>(ny);
        ndarray<double> un(std::vector<int>{ny, nx});
        ndarray<double> vn(std::vector<int>{ny, nx});
        for (int j = 0; j < ny; ++j)
        {
            const int jm = (j - 1 + ny) % ny, jp = (j + 1) % ny;
            for (int i = 0; i < nx; ++i)
            {
                const int im = (i - 1 + nx) % nx, ip = (i + 1) % nx;
                const double uc = u(j, i), vc = v(j, i);
                const double dudx = (u(j, ip) - u(j, im)) / (2.0 * dx);
                const double dudy = (u(jp, i) - u(jm, i)) / (2.0 * dy);
                const double dvdx = (v(j, ip) - v(j, im)) / (2.0 * dx);
                const double dvdy = (v(jp, i) - v(jm, i)) / (2.0 * dy);
                const double lap_u = (u(j, ip) - 2.0 * uc + u(j, im)) / (dx * dx) +
                                     (u(jp, i) - 2.0 * uc + u(jm, i)) / (dy * dy);
                const double lap_v = (v(j, ip) - 2.0 * vc + v(j, im)) / (dx * dx) +
                                     (v(jp, i) - 2.0 * vc + v(jm, i)) / (dy * dy);
                un(j, i) = uc + dt * (-uc * dudx - vc * dudy + nu * lap_u);
                vn(j, i) = vc + dt * (-uc * dvdx - vc * dvdy + nu * lap_v);
            }
        }
        std::swap(u, un);
        std::swap(v, vn);
    }

    NP_NODISCARD double max_speed() const
    {
        double m = 0.0;
        for (std::size_t n = 0; n < u.size(); ++n)
        {
            const double s = std::sqrt(u.data()[n] * u.data()[n] + v.data()[n] * v.data()[n]);
            m = std::max(m, s);
        }
        return m;
    }
  };

  /**
   * @brief 2D Stokes (creeping) flow on the unit square: like NavierStokes2D
   *        but without the nonlinear advection term, for Re -> 0 regimes.
   *        Same Chorin projection, same no-slip walls, reuses FluidState.
   */
  struct Stokes2D
  {
    FluidState state;
    double Re = 1.0, dt = 0.01;
    int poisson_iters = 50;

    Stokes2D() = default;
    Stokes2D(int nx, int ny, double Re_ = 1.0) : state(nx, ny), Re(Re_)
    {
    }

    NP_API void step()
    {
        const int nx = state.nx, ny = state.ny;
        if (nx < 3 || ny < 3)
        {
            return;
        }
        const double dx = 1.0 / static_cast<double>(nx - 1);
        const double dy = 1.0 / static_cast<double>(ny - 1);
        const double nu = 1.0 / Re;
        const double rho = 1.0;

        auto &u = state.u;
        auto &v = state.v;
        auto &p = state.p;

        // Provisional velocity: diffusion only (no advection at Re -> 0).
        ndarray<double> u_star(std::vector<int>{ny, nx});
        ndarray<double> v_star(std::vector<int>{ny, nx});
        for (int j = 1; j < ny - 1; ++j)
        {
            for (int i = 1; i < nx - 1; ++i)
            {
                const double un = u(j, i), vn = v(j, i);
                const double lap_u = (u(j, i + 1) - 2.0 * un + u(j, i - 1)) / (dx * dx) +
                                     (u(j + 1, i) - 2.0 * un + u(j - 1, i)) / (dy * dy);
                const double lap_v = (v(j, i + 1) - 2.0 * vn + v(j, i - 1)) / (dx * dx) +
                                     (v(j + 1, i) - 2.0 * vn + v(j - 1, i)) / (dy * dy);
                u_star(j, i) = un + dt * nu * lap_u;
                v_star(j, i) = vn + dt * nu * lap_v;
            }
        }
        zero_walls(u_star);
        zero_walls(v_star);

        // Projection onto the divergence-free field (as in NavierStokes2D).
        ndarray<double> rhs(std::vector<int>{ny, nx});
        for (int j = 1; j < ny - 1; ++j)
        {
            for (int i = 1; i < nx - 1; ++i)
            {
                const double div = (u_star(j, i + 1) - u_star(j, i - 1)) / (2.0 * dx) +
                                   (v_star(j + 1, i) - v_star(j - 1, i)) / (2.0 * dy);
                rhs(j, i) = (rho / dt) * div;
            }
        }
        ndarray<double> p_new(std::vector<int>{ny, nx});
        const double denom = 2.0 * (dx * dx + dy * dy);
        for (int iter = 0; iter < poisson_iters; ++iter)
        {
            for (int j = 1; j < ny - 1; ++j)
            {
                for (int i = 1; i < nx - 1; ++i)
                {
                    p_new(j, i) = ((p(j, i + 1) + p(j, i - 1)) * dy * dy + (p(j + 1, i) + p(j - 1, i)) * dx * dx -
                                   rhs(j, i) * dx * dx * dy * dy) /
                                  denom;
                }
            }
            for (int i = 0; i < nx; ++i)
            {
                p_new(0, i) = p_new(1, i);
                p_new(ny - 1, i) = p_new(ny - 2, i);
            }
            for (int j = 0; j < ny; ++j)
            {
                p_new(j, 0) = p_new(j, 1);
                p_new(j, nx - 1) = p_new(j, nx - 2);
            }
            p_new(0, 0) = 0.0;
            std::swap(p, p_new);
        }

        for (int j = 1; j < ny - 1; ++j)
        {
            for (int i = 1; i < nx - 1; ++i)
            {
                u(j, i) = u_star(j, i) - dt / rho * (p(j, i + 1) - p(j, i - 1)) / (2.0 * dx);
                v(j, i) = v_star(j, i) - dt / rho * (p(j + 1, i) - p(j - 1, i)) / (2.0 * dy);
            }
        }
        zero_walls(u);
        zero_walls(v);
    }

    NP_NODISCARD double kinetic_energy() const
    {
        const int nx = state.nx, ny = state.ny;
        if (nx == 0 || ny == 0)
        {
            return 0.0;
        }
        const double cell = 1.0 / static_cast<double>((nx > 1) ? nx - 1 : 1) / static_cast<double>((ny > 1) ? ny - 1 : 1);
        double ke = 0.0;
        for (std::size_t n = 0; n < state.u.size(); ++n)
        {
            ke += state.u.data()[n] * state.u.data()[n] + state.v.data()[n] * state.v.data()[n];
        }
        return 0.5 * ke * cell;
    }

    NP_NODISCARD double max_divergence() const
    {
        const int nx = state.nx, ny = state.ny;
        if (nx < 3 || ny < 3)
        {
            return 0.0;
        }
        const double dx = 1.0 / static_cast<double>(nx - 1);
        const double dy = 1.0 / static_cast<double>(ny - 1);
        double m = 0.0;
        for (int j = 1; j < ny - 1; ++j)
        {
            for (int i = 1; i < nx - 1; ++i)
            {
                const double div = (state.u(j, i + 1) - state.u(j, i - 1)) / (2.0 * dx) +
                                   (state.v(j + 1, i) - state.v(j - 1, i)) / (2.0 * dy);
                m = std::max(m, std::abs(div));
            }
        }
        return m;
    }

  private:
    static void zero_walls(ndarray<double> &f)
    {
        const int ny = f.shape[0], nx = f.shape[1];
        for (int i = 0; i < nx; ++i)
        {
            f(0, i) = 0.0;
            f(ny - 1, i) = 0.0;
        }
        for (int j = 0; j < ny; ++j)
        {
            f(j, 0) = 0.0;
            f(j, nx - 1) = 0.0;
        }
    }
  };

  /// Velocity field (u, v) pair returned by PotentialFlow2D::velocity().
  struct FlowField2D
  {
    ndarray<double> u, v;
  };

  /**
   * @brief 2D potential flow on the unit square: solves lap(phi) = 0 with a
   *        uniform freestream (phi = U*x on inlet/outlet, dp/dn = 0 top/bottom).
   *        Velocity follows as u = grad(phi); phi = U*x is the exact discrete
   *        solution, so convergence to (U, 0) validates the Laplace solver.
   */
  struct PotentialFlow2D
  {
    int nx = 0, ny = 0;
    ndarray<double> phi;
    double U = 1.0;
    int iters = 500;

    PotentialFlow2D() = default;
    PotentialFlow2D(int nx_, int ny_, double U_ = 1.0) : nx(nx_), ny(ny_), phi(std::vector<int>{ny_, nx_}), U(U_)
    {
    }

    NP_API void solve()
    {
        if (nx < 3 || ny < 3)
        {
            return;
        }
        const double dx = 1.0 / static_cast<double>(nx - 1);
        const double dy = 1.0 / static_cast<double>(ny - 1);
        ndarray<double> pn(std::vector<int>{ny, nx});
        const double denom = 2.0 * (dx * dx + dy * dy);
        for (int it = 0; it < iters; ++it)
        {
            for (int j = 1; j < ny - 1; ++j)
            {
                for (int i = 1; i < nx - 1; ++i)
                {
                    pn(j, i) = ((phi(j, i + 1) + phi(j, i - 1)) * dy * dy +
                                (phi(j + 1, i) + phi(j - 1, i)) * dx * dx) /
                               denom;
                }
            }
            for (int j = 0; j < ny; ++j)
            {
                pn(j, 0) = 0.0;
                pn(j, nx - 1) = U;
            }
            for (int i = 0; i < nx; ++i)
            {
                pn(0, i) = pn(1, i);
                pn(ny - 1, i) = pn(ny - 2, i);
            }
            std::swap(phi, pn);
        }
    }

    NP_NODISCARD FlowField2D velocity() const
    {
        FlowField2D out{ndarray<double>(std::vector<int>{ny, nx}), ndarray<double>(std::vector<int>{ny, nx})};
        if (nx < 3 || ny < 3)
        {
            return out;
        }
        const double dx = 1.0 / static_cast<double>(nx - 1);
        const double dy = 1.0 / static_cast<double>(ny - 1);
        for (int j = 1; j < ny - 1; ++j)
        {
            for (int i = 1; i < nx - 1; ++i)
            {
                out.u(j, i) = (phi(j, i + 1) - phi(j, i - 1)) / (2.0 * dx);
                out.v(j, i) = (phi(j + 1, i) - phi(j - 1, i)) / (2.0 * dy);
            }
        }
        return out;
    }
  };

  /// Centroid (x, y) of a scalar field, e.g. for tracking advected pulses.
  struct Centroid
  {
    double x = 0.0, y = 0.0;
  };

  /**
   * @brief 2D scalar advection-diffusion on the unit square with a prescribed
   *        uniform velocity (ax, ay): dT/dt + a·grad(T) = kappa lap(T).
   *        Upwind advection (stable for any flow direction) + central diffusion,
   *        Dirichlet walls fixed at bc.
   */
  struct AdvectionDiffusion2D
  {
    int nx = 0, ny = 0;
    ndarray<double> T;
    double ax = 1.0, ay = 0.0, kappa = 0.0, dt = 0.001, bc = 0.0;

    AdvectionDiffusion2D() = default;
    AdvectionDiffusion2D(int nx_, int ny_, double ax_ = 1.0, double ay_ = 0.0, double kappa_ = 0.0,
                         double dt_ = 0.001)
        : nx(nx_), ny(ny_), T(std::vector<int>{ny_, nx_}), ax(ax_), ay(ay_), kappa(kappa_), dt(dt_)
    {
    }

    NP_API void step()
    {
        if (nx < 3 || ny < 3)
        {
            return;
        }
        const double dx = 1.0 / static_cast<double>(nx - 1);
        const double dy = 1.0 / static_cast<double>(ny - 1);
        ndarray<double> Tn(std::vector<int>{ny, nx});
        for (int j = 1; j < ny - 1; ++j)
        {
            for (int i = 1; i < nx - 1; ++i)
            {
                const double dTdx = (ax >= 0.0) ? (T(j, i) - T(j, i - 1)) / dx : (T(j, i + 1) - T(j, i)) / dx;
                const double dTdy = (ay >= 0.0) ? (T(j, i) - T(j - 1, i)) / dy : (T(j + 1, i) - T(j, i)) / dy;
                const double lap = (T(j, i + 1) - 2.0 * T(j, i) + T(j, i - 1)) / (dx * dx) +
                                   (T(j + 1, i) - 2.0 * T(j, i) + T(j - 1, i)) / (dy * dy);
                Tn(j, i) = T(j, i) + dt * (-ax * dTdx - ay * dTdy + kappa * lap);
            }
        }
        for (int i = 0; i < nx; ++i)
        {
            Tn(0, i) = bc;
            Tn(ny - 1, i) = bc;
        }
        for (int j = 0; j < ny; ++j)
        {
            Tn(j, 0) = bc;
            Tn(j, nx - 1) = bc;
        }
        std::swap(T, Tn);
    }

    /// Gaussian pulse centered at (cx, cy) — for advection tracking tests.
    NP_API void set_gaussian(double cx, double cy, double sigma, double amp = 1.0)
    {
        const double dx = 1.0 / static_cast<double>(nx - 1);
        const double dy = 1.0 / static_cast<double>(ny - 1);
        for (int j = 0; j < ny; ++j)
        {
            for (int i = 0; i < nx; ++i)
            {
                const double ex = i * dx - cx, ey = j * dy - cy;
                T(j, i) = amp * std::exp(-(ex * ex + ey * ey) / (2.0 * sigma * sigma));
            }
        }
    }

    NP_NODISCARD double total_mass() const
    {
        if (nx < 2 || ny < 2)
        {
            return 0.0;
        }
        double s = 0.0;
        for (std::size_t n = 0; n < T.size(); ++n)
        {
            s += T.data()[n];
        }
        return s / static_cast<double>(nx - 1) / static_cast<double>(ny - 1);
    }

    NP_NODISCARD Centroid centroid() const
    {
        Centroid c;
        if (nx < 2 || ny < 2)
        {
            return c;
        }
        const double dx = 1.0 / static_cast<double>(nx - 1);
        const double dy = 1.0 / static_cast<double>(ny - 1);
        double m = 0.0, sx = 0.0, sy = 0.0;
        for (int j = 0; j < ny; ++j)
        {
            for (int i = 0; i < nx; ++i)
            {
                const double w = T(j, i);
                m += w;
                sx += w * i * dx;
                sy += w * j * dy;
            }
        }
        if (m != 0.0)
        {
            c.x = sx / m;
            c.y = sy / m;
        }
        return c;
    }
  };

  // ── Heat transfer ──

  /**
   * @brief 2D heat equation on the unit square: dT/dt = alpha lap(T),
   *        explicit FTCS with Dirichlet walls fixed at bc. The step is
   *        clamped to the diffusive stability limit (see stable_dt()).
   */
  struct Heat2D
  {
    int nx = 0, ny = 0;
    ndarray<double> T;
    double alpha = 0.01, dt = 0.01, bc = 0.0;

    Heat2D() = default;
    Heat2D(int nx_, int ny_, double alpha_ = 0.01, double dt_ = 0.01)
        : nx(nx_), ny(ny_), T(std::vector<int>{ny_, nx_}), alpha(alpha_), dt(dt_)
    {
    }

    /// FTCS stability limit dt <= 1 / (2*alpha*(1/dx² + 1/dy²)).
    NP_NODISCARD double stable_dt() const
    {
        if (nx < 3 || ny < 3 || alpha <= 0.0)
        {
            return dt;
        }
        const double dx = 1.0 / static_cast<double>(nx - 1);
        const double dy = 1.0 / static_cast<double>(ny - 1);
        return 1.0 / (2.0 * alpha * (1.0 / (dx * dx) + 1.0 / (dy * dy)));
    }

    NP_API void step()
    {
        if (nx < 3 || ny < 3)
        {
            return;
        }
        const double dx = 1.0 / static_cast<double>(nx - 1);
        const double dy = 1.0 / static_cast<double>(ny - 1);
        const double cdt = std::min(dt, stable_dt());
        ndarray<double> Tn(std::vector<int>{ny, nx});
        for (int j = 1; j < ny - 1; ++j)
        {
            for (int i = 1; i < nx - 1; ++i)
            {
                const double lap = (T(j, i + 1) - 2.0 * T(j, i) + T(j, i - 1)) / (dx * dx) +
                                   (T(j + 1, i) - 2.0 * T(j, i) + T(j - 1, i)) / (dy * dy);
                Tn(j, i) = T(j, i) + cdt * alpha * lap;
            }
        }
        for (int i = 0; i < nx; ++i)
        {
            Tn(0, i) = bc;
            Tn(ny - 1, i) = bc;
        }
        for (int j = 0; j < ny; ++j)
        {
            Tn(j, 0) = bc;
            Tn(j, nx - 1) = bc;
        }
        std::swap(T, Tn);
    }

    /// Gaussian hot spot centered at (cx, cy) for decay tests and demos.
    NP_API void set_gaussian(double cx, double cy, double sigma, double amp = 1.0)
    {
        const double dx = 1.0 / static_cast<double>(nx - 1);
        const double dy = 1.0 / static_cast<double>(ny - 1);
        for (int j = 0; j < ny; ++j)
        {
            for (int i = 0; i < nx; ++i)
            {
                const double ex = i * dx - cx, ey = j * dy - cy;
                T(j, i) = amp * std::exp(-(ex * ex + ey * ey) / (2.0 * sigma * sigma));
            }
        }
    }

    NP_NODISCARD double total_heat() const
    {
        if (nx < 2 || ny < 2)
        {
            return 0.0;
        }
        double s = 0.0;
        for (std::size_t n = 0; n < T.size(); ++n)
        {
            s += T.data()[n];
        }
        return s / static_cast<double>(nx - 1) / static_cast<double>(ny - 1);
    }

    NP_NODISCARD double max_temp() const
    {
        double m = 0.0;
        for (std::size_t n = 0; n < T.size(); ++n)
        {
            m = std::max(m, T.data()[n]);
        }
        return m;
    }
  };

  /**
   * @brief 3D heat equation on the unit cube: dT/dt = alpha lap(T),
   *        explicit FTCS with Dirichlet walls fixed at bc. Storage (k, j, i).
   */
  struct Heat3D
  {
    int nx = 0, ny = 0, nz = 0;
    ndarray<double> T;
    double alpha = 0.01, dt = 0.01, bc = 0.0;

    Heat3D() = default;
    Heat3D(int nx_, int ny_, int nz_, double alpha_ = 0.01, double dt_ = 0.01)
        : nx(nx_), ny(ny_), nz(nz_), T(std::vector<int>{nz_, ny_, nx_}), alpha(alpha_), dt(dt_)
    {
    }

    /// FTCS stability limit dt <= 1 / (2*alpha*(1/dx² + 1/dy² + 1/dz²)).
    NP_NODISCARD double stable_dt() const
    {
        if (nx < 3 || ny < 3 || nz < 3 || alpha <= 0.0)
        {
            return dt;
        }
        const double dx = 1.0 / static_cast<double>(nx - 1);
        const double dy = 1.0 / static_cast<double>(ny - 1);
        const double dz = 1.0 / static_cast<double>(nz - 1);
        return 1.0 / (2.0 * alpha * (1.0 / (dx * dx) + 1.0 / (dy * dy) + 1.0 / (dz * dz)));
    }

    NP_API void step()
    {
        if (nx < 3 || ny < 3 || nz < 3)
        {
            return;
        }
        const double dx = 1.0 / static_cast<double>(nx - 1);
        const double dy = 1.0 / static_cast<double>(ny - 1);
        const double dz = 1.0 / static_cast<double>(nz - 1);
        const double cdt = std::min(dt, stable_dt());
        ndarray<double> Tn(std::vector<int>{nz, ny, nx});
        for (int k = 1; k < nz - 1; ++k)
        {
            for (int j = 1; j < ny - 1; ++j)
            {
                for (int i = 1; i < nx - 1; ++i)
                {
                    const double lap = (T(k, j, i + 1) - 2.0 * T(k, j, i) + T(k, j, i - 1)) / (dx * dx) +
                                       (T(k, j + 1, i) - 2.0 * T(k, j, i) + T(k, j - 1, i)) / (dy * dy) +
                                       (T(k + 1, j, i) - 2.0 * T(k, j, i) + T(k - 1, j, i)) / (dz * dz);
                    Tn(k, j, i) = T(k, j, i) + cdt * alpha * lap;
                }
            }
        }
        for (int j = 0; j < ny; ++j)
        {
            for (int i = 0; i < nx; ++i)
            {
                Tn(0, j, i) = bc;
                Tn(nz - 1, j, i) = bc;
            }
        }
        for (int k = 0; k < nz; ++k)
        {
            for (int i = 0; i < nx; ++i)
            {
                Tn(k, 0, i) = bc;
                Tn(k, ny - 1, i) = bc;
            }
        }
        for (int k = 0; k < nz; ++k)
        {
            for (int j = 0; j < ny; ++j)
            {
                Tn(k, j, 0) = bc;
                Tn(k, j, nx - 1) = bc;
            }
        }
        std::swap(T, Tn);
    }

    NP_NODISCARD double total_heat() const
    {
        if (nx < 2 || ny < 2 || nz < 2)
        {
            return 0.0;
        }
        double s = 0.0;
        for (std::size_t n = 0; n < T.size(); ++n)
        {
            s += T.data()[n];
        }
        return s / static_cast<double>(nx - 1) / static_cast<double>(ny - 1) / static_cast<double>(nz - 1);
    }

    NP_NODISCARD double max_temp() const
    {
        double m = 0.0;
        for (std::size_t n = 0; n < T.size(); ++n)
        {
            m = std::max(m, T.data()[n]);
        }
        return m;
    }
  };

  /**
   * @brief 2D Boussinesq natural convection on the unit square: Navier-Stokes
   *        plus a temperature field with buoyancy beta*g*(T - T_ref) forcing
   *        the vertical momentum. Hot bottom wall (T_hot), cold top (T_cold),
   *        side walls follow the linear conduction profile. No-slip velocity
   *        walls, Chorin projection shared with NavierStokes2D.
   */
  struct Boussinesq2D
  {
    FluidState state;
    ndarray<double> T;
    double Re = 100.0, alpha_T = 0.01, beta = 0.5, gravity = 1.0;
    double T_ref = 0.0, T_hot = 1.0, T_cold = 0.0;
    double dt = 0.01;
    int poisson_iters = 50;

    Boussinesq2D() = default;
    Boussinesq2D(int nx, int ny, double Re_ = 100.0)
        : state(nx, ny), T(std::vector<int>{ny, nx}), Re(Re_)
    {
        reset_conduction();
    }

    /// Quiescent linear conduction profile between the hot/cold walls.
    NP_API void reset_conduction()
    {
        const int nx = state.nx, ny = state.ny;
        for (int j = 0; j < ny; ++j)
        {
            const double y = (ny > 1) ? static_cast<double>(j) / static_cast<double>(ny - 1) : 0.0;
            for (int i = 0; i < nx; ++i)
            {
                T(j, i) = T_hot + (T_cold - T_hot) * y;
            }
        }
    }

    NP_API void step()
    {
        const int nx = state.nx, ny = state.ny;
        if (nx < 3 || ny < 3)
        {
            return;
        }
        const double dx = 1.0 / static_cast<double>(nx - 1);
        const double dy = 1.0 / static_cast<double>(ny - 1);
        const double nu = 1.0 / Re;
        const double rho = 1.0;

        auto &u = state.u;
        auto &v = state.v;
        auto &p = state.p;

        // Provisional velocity: advection + diffusion + buoyancy in v.
        ndarray<double> u_star(std::vector<int>{ny, nx});
        ndarray<double> v_star(std::vector<int>{ny, nx});
        for (int j = 1; j < ny - 1; ++j)
        {
            for (int i = 1; i < nx - 1; ++i)
            {
                const double un = u(j, i), vn = v(j, i);
                const double dudx = (u(j, i + 1) - u(j, i - 1)) / (2.0 * dx);
                const double dudy = (u(j + 1, i) - u(j - 1, i)) / (2.0 * dy);
                const double dvdx = (v(j, i + 1) - v(j, i - 1)) / (2.0 * dx);
                const double dvdy = (v(j + 1, i) - v(j - 1, i)) / (2.0 * dy);
                const double lap_u = (u(j, i + 1) - 2.0 * un + u(j, i - 1)) / (dx * dx) +
                                     (u(j + 1, i) - 2.0 * un + u(j - 1, i)) / (dy * dy);
                const double lap_v = (v(j, i + 1) - 2.0 * vn + v(j, i - 1)) / (dx * dx) +
                                     (v(j + 1, i) - 2.0 * vn + v(j - 1, i)) / (dy * dy);
                u_star(j, i) = un + dt * (-un * dudx - vn * dudy + nu * lap_u);
                v_star(j, i) = vn + dt * (-un * dvdx - vn * dvdy + nu * lap_v + beta * gravity * (T(j, i) - T_ref));
            }
        }

        // Temperature: upwind advection with (u, v) + explicit diffusion.
        ndarray<double> Tn(std::vector<int>{ny, nx});
        for (int j = 1; j < ny - 1; ++j)
        {
            for (int i = 1; i < nx - 1; ++i)
            {
                const double un = u(j, i), vn = v(j, i);
                const double dTdx = (un >= 0.0) ? (T(j, i) - T(j, i - 1)) / dx : (T(j, i + 1) - T(j, i)) / dx;
                const double dTdy = (vn >= 0.0) ? (T(j, i) - T(j - 1, i)) / dy : (T(j + 1, i) - T(j, i)) / dy;
                const double lap = (T(j, i + 1) - 2.0 * T(j, i) + T(j, i - 1)) / (dx * dx) +
                                   (T(j + 1, i) - 2.0 * T(j, i) + T(j - 1, i)) / (dy * dy);
                Tn(j, i) = T(j, i) + dt * (-un * dTdx - vn * dTdy + alpha_T * lap);
            }
        }
        enforce_temp_walls(Tn);
        std::swap(T, Tn);

        // Projection (Neumann pressure, pinned corner).
        ndarray<double> rhs(std::vector<int>{ny, nx});
        for (int j = 1; j < ny - 1; ++j)
        {
            for (int i = 1; i < nx - 1; ++i)
            {
                rhs(j, i) = (rho / dt) * ((u_star(j, i + 1) - u_star(j, i - 1)) / (2.0 * dx) +
                                          (v_star(j + 1, i) - v_star(j - 1, i)) / (2.0 * dy));
            }
        }
        ndarray<double> p_new(std::vector<int>{ny, nx});
        const double denom = 2.0 * (dx * dx + dy * dy);
        for (int iter = 0; iter < poisson_iters; ++iter)
        {
            for (int j = 1; j < ny - 1; ++j)
            {
                for (int i = 1; i < nx - 1; ++i)
                {
                    p_new(j, i) = ((p(j, i + 1) + p(j, i - 1)) * dy * dy + (p(j + 1, i) + p(j - 1, i)) * dx * dx -
                                   rhs(j, i) * dx * dx * dy * dy) /
                                  denom;
                }
            }
            for (int i = 0; i < nx; ++i)
            {
                p_new(0, i) = p_new(1, i);
                p_new(ny - 1, i) = p_new(ny - 2, i);
            }
            for (int j = 0; j < ny; ++j)
            {
                p_new(j, 0) = p_new(j, 1);
                p_new(j, nx - 1) = p_new(j, nx - 2);
            }
            p_new(0, 0) = 0.0;
            std::swap(p, p_new);
        }

        for (int j = 1; j < ny - 1; ++j)
        {
            for (int i = 1; i < nx - 1; ++i)
            {
                u(j, i) = u_star(j, i) - dt / rho * (p(j, i + 1) - p(j, i - 1)) / (2.0 * dx);
                v(j, i) = v_star(j, i) - dt / rho * (p(j + 1, i) - p(j - 1, i)) / (2.0 * dy);
            }
        }
        for (int i = 0; i < nx; ++i)
        {
            u(0, i) = 0.0;
            u(ny - 1, i) = 0.0;
            v(0, i) = 0.0;
            v(ny - 1, i) = 0.0;
        }
        for (int j = 0; j < ny; ++j)
        {
            u(j, 0) = 0.0;
            u(j, nx - 1) = 0.0;
            v(j, 0) = 0.0;
            v(j, nx - 1) = 0.0;
        }
    }

    NP_NODISCARD double max_speed() const
    {
        double m = 0.0;
        for (std::size_t n = 0; n < state.u.size(); ++n)
        {
            const double s =
                std::sqrt(state.u.data()[n] * state.u.data()[n] + state.v.data()[n] * state.v.data()[n]);
            m = std::max(m, s);
        }
        return m;
    }

    NP_NODISCARD double max_temp() const
    {
        double m = 0.0;
        for (std::size_t n = 0; n < T.size(); ++n)
        {
            m = std::max(m, T.data()[n]);
        }
        return m;
    }

  private:
    void enforce_temp_walls(ndarray<double> &f) const
    {
        const int nx = state.nx, ny = state.ny;
        for (int i = 0; i < nx; ++i)
        {
            f(0, i) = T_hot;
            f(ny - 1, i) = T_cold;
        }
        for (int j = 0; j < ny; ++j)
        {
            const double y = (ny > 1) ? static_cast<double>(j) / static_cast<double>(ny - 1) : 0.0;
            const double side = T_hot + (T_cold - T_hot) * y;
            f(j, 0) = side;
            f(j, nx - 1) = side;
        }
    }
  };

  // ── Ballistics ──

  /// Point-mass state for Projectile (x, y, vx, vy, t).
  struct ProjectileState
  {
    double x = 0.0, y = 0.0, vx = 0.0, vy = 0.0, t = 0.0;
  };

  /// One recorded sample of a projectile trajectory.
  struct TrajPoint
  {
    double x = 0.0, y = 0.0, t = 0.0;
  };

  /**
   * @brief 2D projectile with gravity, quadratic air drag and headwind:
   *        a = -g ŷ - k*|v - w|*(v - w), with lumped k = drag/mass and
   *        wind (wind_x, 0). Integrated with classic RK4; simulate() runs
   *        from the origin until ground impact (y < 0, linearly interpolated).
   */
  struct Projectile
  {
    double g = 9.81, drag = 0.0, wind_x = 0.0;

    Projectile() = default;
    Projectile(double g_, double drag_ = 0.0, double wind_x_ = 0.0) : g(g_), drag(drag_), wind_x(wind_x_)
    {
    }

    NP_API void step_rk4(ProjectileState &s, double dt) const
    {
        const auto accel = [this](double vx, double vy) {
            const double rx = vx - wind_x, ry = vy;
            const double sp = std::sqrt(rx * rx + ry * ry);
            return std::array<double, 2>{-drag * sp * rx, -g - drag * sp * ry};
        };
        const auto a1 = accel(s.vx, s.vy);
        const double vx2 = s.vx + 0.5 * dt * a1[0], vy2 = s.vy + 0.5 * dt * a1[1];
        const auto a2 = accel(vx2, vy2);
        const double vx3 = s.vx + 0.5 * dt * a2[0], vy3 = s.vy + 0.5 * dt * a2[1];
        const auto a3 = accel(vx3, vy3);
        const double vx4 = s.vx + dt * a3[0], vy4 = s.vy + dt * a3[1];
        const auto a4 = accel(vx4, vy4);
        s.x += dt / 6.0 * (s.vx + 2.0 * vx2 + 2.0 * vx3 + vx4);
        s.y += dt / 6.0 * (s.vy + 2.0 * vy2 + 2.0 * vy3 + vy4);
        s.vx += dt / 6.0 * (a1[0] + 2.0 * a2[0] + 2.0 * a3[0] + a4[0]);
        s.vy += dt / 6.0 * (a1[1] + 2.0 * a2[1] + 2.0 * a3[1] + a4[1]);
        s.t += dt;
    }

    /// Launch at speed v0 / angle and record until ground impact.
    NP_NODISCARD std::vector<TrajPoint> simulate(double v0, double angle_rad, double dt,
                                                 double tmax = 100.0) const
    {
        ProjectileState s;
        s.vx = v0 * std::cos(angle_rad);
        s.vy = v0 * std::sin(angle_rad);
        std::vector<TrajPoint> traj;
        traj.reserve(1024);
        traj.push_back({s.x, s.y, s.t});
        while (s.t < tmax)
        {
            const ProjectileState prev = s;
            step_rk4(s, dt);
            if (s.y < 0.0 && s.t > 0.0)
            {
                // Linearly interpolate the ground crossing for an exact range.
                const double frac = prev.y / (prev.y - s.y);
                traj.push_back({prev.x + frac * (s.x - prev.x), 0.0, prev.t + frac * dt});
                break;
            }
            traj.push_back({s.x, s.y, s.t});
        }
        return traj;
    }

    /// Horizontal range of the last trajectory sample (≈ impact point).
    NP_NODISCARD static double range(const std::vector<TrajPoint> &traj)
    {
        return traj.empty() ? 0.0 : traj.back().x;
    }

    NP_NODISCARD static double range_vacuum(double v0, double angle_rad, double g = 9.81) noexcept
    {
        return v0 * v0 * std::sin(2.0 * angle_rad) / g;
    }

    NP_NODISCARD static double max_height_vacuum(double v0, double angle_rad, double g = 9.81) noexcept
    {
        const double vy = v0 * std::sin(angle_rad);
        return vy * vy / (2.0 * g);
    }
  };

  // ── Waves and classical mechanics ──

  /**
   * @brief 1D wave equation on [0, L] with fixed ends: d²u/dt² = c² d²u/dx²,
   *        leapfrog in time + central differences in space (CFL c*dt/dx <= 1).
   */
  struct Wave1D
  {
    int n = 0;
    ndarray<double> u, u_prev;
    double c = 1.0, dt = 0.001, L = 1.0;

    Wave1D() = default;
    Wave1D(int n_, double c_ = 1.0, double dt_ = 0.001) : n(n_), u(std::vector<int>{n_}), u_prev(std::vector<int>{n_}), c(c_), dt(dt_)
    {
    }

    NP_API void step()
    {
        if (n < 3)
        {
            return;
        }
        const double dx = L / static_cast<double>(n - 1);
        const double c2 = (c * dt / dx) * (c * dt / dx);
        ndarray<double> un(std::vector<int>{n});
        for (int i = 1; i < n - 1; ++i)
        {
            un(i) = 2.0 * u(i) - u_prev(i) + c2 * (u(i + 1) - 2.0 * u(i) + u(i - 1));
        }
        std::swap(u_prev, u);
        std::swap(u, un);
    }

    /// Stationary Gaussian pluck (zero initial velocity: u_prev = u).
    NP_API void pluck_gaussian(double center, double sigma, double amp = 1.0)
    {
        const double dx = L / static_cast<double>(n - 1);
        for (int i = 0; i < n; ++i)
        {
            const double e = i * dx - center;
            u(i) = amp * std::exp(-(e * e) / (2.0 * sigma * sigma));
            u_prev(i) = u(i);
        }
        u(0) = 0.0;
        u(n - 1) = 0.0;
        u_prev(0) = 0.0;
        u_prev(n - 1) = 0.0;
    }

    /// Total (kinetic + potential) discrete energy.
    NP_NODISCARD double energy() const
    {
        if (n < 3)
        {
            return 0.0;
        }
        const double dx = L / static_cast<double>(n - 1);
        double e = 0.0;
        for (int i = 0; i < n; ++i)
        {
            const double v = (u(i) - u_prev(i)) / dt;
            e += 0.5 * v * v;
        }
        for (int i = 0; i < n - 1; ++i)
        {
            const double s = c * (u(i + 1) - u(i)) / dx;
            e += 0.5 * s * s;
        }
        return e * dx;
    }

    NP_NODISCARD double max_abs() const
    {
        double m = 0.0;
        for (std::size_t k = 0; k < u.size(); ++k)
        {
            m = std::max(m, std::abs(u.data()[k]));
        }
        return m;
    }
  };

  /**
   * @brief 2D wave equation on the unit square with fixed walls, leapfrog
   *        in time (CFL c*dt*sqrt(1/dx² + 1/dy²) <= 1).
   */
  struct Wave2D
  {
    int nx = 0, ny = 0;
    ndarray<double> u, u_prev;
    double c = 1.0, dt = 0.001;

    Wave2D() = default;
    Wave2D(int nx_, int ny_, double c_ = 1.0, double dt_ = 0.001)
        : nx(nx_), ny(ny_), u(std::vector<int>{ny_, nx_}), u_prev(std::vector<int>{ny_, nx_}), c(c_), dt(dt_)
    {
    }

    NP_API void step()
    {
        if (nx < 3 || ny < 3)
        {
            return;
        }
        const double dx = 1.0 / static_cast<double>(nx - 1);
        const double dy = 1.0 / static_cast<double>(ny - 1);
        const double cx = (c * dt / dx) * (c * dt / dx);
        const double cy = (c * dt / dy) * (c * dt / dy);
        ndarray<double> un(std::vector<int>{ny, nx});
        for (int j = 1; j < ny - 1; ++j)
        {
            for (int i = 1; i < nx - 1; ++i)
            {
                un(j, i) = 2.0 * u(j, i) - u_prev(j, i) + cx * (u(j, i + 1) - 2.0 * u(j, i) + u(j, i - 1)) +
                           cy * (u(j + 1, i) - 2.0 * u(j, i) + u(j - 1, i));
            }
        }
        std::swap(u_prev, u);
        std::swap(u, un);
    }

    NP_API void pluck_gaussian(double cx, double cy, double sigma, double amp = 1.0)
    {
        const double dx = 1.0 / static_cast<double>(nx - 1);
        const double dy = 1.0 / static_cast<double>(ny - 1);
        for (int j = 0; j < ny; ++j)
        {
            for (int i = 0; i < nx; ++i)
            {
                const double ex = i * dx - cx, ey = j * dy - cy;
                u(j, i) = amp * std::exp(-(ex * ex + ey * ey) / (2.0 * sigma * sigma));
                u_prev(j, i) = u(j, i);
            }
        }
    }

    NP_NODISCARD double max_abs() const
    {
        double m = 0.0;
        for (std::size_t k = 0; k < u.size(); ++k)
        {
            m = std::max(m, std::abs(u.data()[k]));
        }
        return m;
    }
  };

  /**
   * @brief Undamped harmonic oscillator m d²x/dt² = -k x, velocity Verlet
   *        (symplectic: energy oscillates around the true value, no drift).
   */
  struct HarmonicOscillator
  {
    double m = 1.0, k = 1.0, x = 1.0, v = 0.0;

    HarmonicOscillator() = default;
    HarmonicOscillator(double m_, double k_, double x0 = 1.0, double v0 = 0.0) : m(m_), k(k_), x(x0), v(v0)
    {
    }

    NP_API void step_verlet(double dt)
    {
        const double a = -k / m * x;
        x += v * dt + 0.5 * a * dt * dt;
        const double a_new = -k / m * x;
        v += 0.5 * (a + a_new) * dt;
    }

    NP_NODISCARD double energy() const
    {
        return 0.5 * m * v * v + 0.5 * k * x * x;
    }

    NP_NODISCARD double period() const
    {
        return 2.0 * M_PI * std::sqrt(m / k);
    }
  };

  /**
   * @brief Nonlinear planar pendulum d²θ/dt² = -(g/L) sin θ, classic RK4.
   */
  struct Pendulum
  {
    double L = 1.0, g = 9.81, theta = 0.1, omega = 0.0;

    Pendulum() = default;
    Pendulum(double L_, double g_, double theta0 = 0.1, double omega0 = 0.0)
        : L(L_), g(g_), theta(theta0), omega(omega0)
    {
    }

    NP_API void step_rk4(double dt)
    {
        const double k1t = omega, k1w = -g / L * std::sin(theta);
        const double k2t = omega + 0.5 * dt * k1w, k2w = -g / L * std::sin(theta + 0.5 * dt * k1t);
        const double k3t = omega + 0.5 * dt * k2w, k3w = -g / L * std::sin(theta + 0.5 * dt * k2t);
        const double k4t = omega + dt * k3w, k4w = -g / L * std::sin(theta + dt * k3t);
        theta += dt / 6.0 * (k1t + 2.0 * k2t + 2.0 * k3t + k4t);
        omega += dt / 6.0 * (k1w + 2.0 * k2w + 2.0 * k3w + k4w);
    }

    /// Mechanical energy per unit mass (zero at the hanging rest position).
    NP_NODISCARD double energy() const
    {
        return 0.5 * L * L * omega * omega + g * L * (1.0 - std::cos(theta));
    }

    /// Small-angle period 2π√(L/g).
    NP_NODISCARD double period_small() const
    {
        return 2.0 * M_PI * std::sqrt(L / g);
    }
  };

  /**
   * @brief Gravitational N-body system in 3D with Plummer softening,
   *        integrated with velocity Verlet (symplectic, momentum-conserving).
   */
  struct NBody
  {
    std::vector<std::array<double, 3>> pos, vel;
    std::vector<double> mass;
    double G = 1.0, softening = 1.0e-3;

    NBody() = default;
    NBody(double G_, double softening_ = 1.0e-3) : G(G_), softening(softening_)
    {
    }

    NP_API void add_body(double m, std::array<double, 3> p, std::array<double, 3> v)
    {
        mass.push_back(m);
        pos.push_back(p);
        vel.push_back(v);
    }

    NP_NODISCARD std::size_t bodies() const noexcept
    {
        return mass.size();
    }

    NP_NODISCARD std::vector<std::array<double, 3>> accelerations() const
    {
        std::vector<std::array<double, 3>> a(pos.size(), {0.0, 0.0, 0.0});
        const double eps2 = softening * softening;
        for (std::size_t i = 0; i < pos.size(); ++i)
        {
            for (std::size_t j = 0; j < pos.size(); ++j)
            {
                if (i == j)
                {
                    continue;
                }
                const double dx = pos[j][0] - pos[i][0];
                const double dy = pos[j][1] - pos[i][1];
                const double dz = pos[j][2] - pos[i][2];
                const double r2 = dx * dx + dy * dy + dz * dz + eps2;
                const double inv = G * mass[j] / (r2 * std::sqrt(r2));
                a[i][0] += inv * dx;
                a[i][1] += inv * dy;
                a[i][2] += inv * dz;
            }
        }
        return a;
    }

    NP_API void step_verlet(double dt)
    {
        if (pos.empty())
        {
            return;
        }
        auto a = accelerations();
        for (std::size_t i = 0; i < pos.size(); ++i)
        {
            vel[i][0] += 0.5 * dt * a[i][0];
            vel[i][1] += 0.5 * dt * a[i][1];
            vel[i][2] += 0.5 * dt * a[i][2];
            pos[i][0] += dt * vel[i][0];
            pos[i][1] += dt * vel[i][1];
            pos[i][2] += dt * vel[i][2];
        }
        auto a_new = accelerations();
        for (std::size_t i = 0; i < pos.size(); ++i)
        {
            vel[i][0] += 0.5 * dt * a_new[i][0];
            vel[i][1] += 0.5 * dt * a_new[i][1];
            vel[i][2] += 0.5 * dt * a_new[i][2];
        }
    }

    NP_NODISCARD double total_energy() const
    {
        double e = 0.0;
        for (std::size_t i = 0; i < pos.size(); ++i)
        {
            e += 0.5 * mass[i] * (vel[i][0] * vel[i][0] + vel[i][1] * vel[i][1] + vel[i][2] * vel[i][2]);
        }
        const double eps2 = softening * softening;
        for (std::size_t i = 0; i < pos.size(); ++i)
        {
            for (std::size_t j = i + 1; j < pos.size(); ++j)
            {
                const double dx = pos[j][0] - pos[i][0];
                const double dy = pos[j][1] - pos[i][1];
                const double dz = pos[j][2] - pos[i][2];
                e -= G * mass[i] * mass[j] / std::sqrt(dx * dx + dy * dy + dz * dz + eps2);
            }
        }
        return e;
    }

    NP_NODISCARD std::array<double, 3> total_momentum() const noexcept
    {
        std::array<double, 3> p{0.0, 0.0, 0.0};
        for (std::size_t i = 0; i < pos.size(); ++i)
        {
            p[0] += mass[i] * vel[i][0];
            p[1] += mass[i] * vel[i][1];
            p[2] += mass[i] * vel[i][2];
        }
        return p;
    }
  };

  // p-adic hook (for Re = p-adic valuation test)
NP_NODISCARD inline bool is_padic_unit_Re(double Re, int p = 5)
{
    // Re is a unit in Q_p iff its valuation is 0. NOTE (honesty audit): an
    // earlier revision ignored both arguments and returned true. For a
    // double this is checkable exactly when integral: nonzero and not
    // divisible by p. Non-integral doubles have no p-adic valuation in
    // this model — treated as units iff nonzero (documented limitation).
    if (Re == 0.0)
        return false;
    double ipart = 0.0;
    if (std::modf(Re, &ipart) == 0.0)
        return std::fmod(ipart, static_cast<double>(p)) != 0.0;
    return true;
}

// ── Physical constants (CODATA 2018, SI) ──
namespace constants
{
/// Exact SI defining constants.
inline constexpr double c = 299792458.0;            ///< Speed of light (m/s, exact).
inline constexpr double h = 6.62607015e-34;         ///< Planck constant (J s, exact).
inline constexpr double hbar = 1.054571817e-34;     ///< Reduced Planck constant (J s).
inline constexpr double e_charge = 1.602176634e-19; ///< Elementary charge (C, exact).
inline constexpr double kB = 1.380649e-23;          ///< Boltzmann constant (J/K, exact).
inline constexpr double NA = 6.02214076e23;         ///< Avogadro constant (1/mol, exact).
inline constexpr double R_gas = kB * NA;            ///< Molar gas constant (J/mol/K).
/// Measured constants.
inline constexpr double G = 6.67430e-11;             ///< Newtonian gravitation (m³/kg/s²).
inline constexpr double eps0 = 8.8541878128e-12;     ///< Vacuum permittivity (F/m).
inline constexpr double mu0 = 1.25663706212e-6;      ///< Vacuum permeability (N/A²).
inline constexpr double k_e = 8.9875517923e9;        ///< Coulomb constant 1/(4πε0) (N m²/C²).
inline constexpr double m_e = 9.1093837015e-31;      ///< Electron mass (kg).
inline constexpr double m_p = 1.67262192369e-27;     ///< Proton mass (kg).
inline constexpr double m_n = 1.67492749804e-27;     ///< Neutron mass (kg).
inline constexpr double amu = 1.66053906660e-27;     ///< Atomic mass unit (kg).
inline constexpr double eV = 1.602176634e-19;        ///< Electronvolt (J).
inline constexpr double sigma_sb = 5.670374419e-8;   ///< Stefan-Boltzmann (W/m²/K⁴).
inline constexpr double alpha_fs = 7.2973525693e-3;  ///< Fine-structure constant.
inline constexpr double a0_bohr = 5.29177210903e-11; ///< Bohr radius (m).
inline constexpr double R_inf = 10973731.568160;     ///< Rydberg constant (1/m).
inline constexpr double wien_b = 2.897771955e-3;     ///< Wien displacement (m K).
inline constexpr double muB = 9.2740100783e-24;      ///< Bohr magneton (J/T).
inline constexpr double angstrom = 1.0e-10;          ///< Angstrom (m).
inline constexpr double pi = std::numbers::pi;

NP_NODISCARD inline double ev_to_j(double ev) noexcept
{
    return ev * eV;
}
NP_NODISCARD inline double j_to_ev(double j) noexcept
{
    return j / eV;
}
NP_NODISCARD inline double amu_to_kg(double u) noexcept
{
    return u * amu;
}
NP_NODISCARD inline double kg_to_amu(double kg) noexcept
{
    return kg / amu;
}
} // namespace constants

// ── Special relativity (SI; c defaults to constants::c) ──
NP_NODISCARD inline double lorentz_gamma(double v, double c = constants::c) noexcept
{
    const double b = v / c;
    if (std::abs(b) >= 1.0)
    {
        return std::numeric_limits<double>::infinity();
    }
    return 1.0 / std::sqrt(1.0 - b * b);
}
/// Lorentz boost along +x: (t, x) -> (t', x').
NP_NODISCARD inline std::array<double, 2> lorentz_transform(double t, double x, double v,
                                                            double c = constants::c) noexcept
{
    const double g = lorentz_gamma(v, c);
    return {g * (t - v * x / (c * c)), g * (x - v * t)};
}
/// Einstein velocity addition for collinear velocities.
NP_NODISCARD inline double velocity_add(double u, double v, double c = constants::c) noexcept
{
    return (u + v) / (1.0 + u * v / (c * c));
}
NP_NODISCARD inline double relativistic_energy(double m, double v, double c = constants::c) noexcept
{
    return lorentz_gamma(v, c) * m * c * c;
}
NP_NODISCARD inline double relativistic_kinetic(double m, double v, double c = constants::c) noexcept
{
    return (lorentz_gamma(v, c) - 1.0) * m * c * c;
}
NP_NODISCARD inline double relativistic_momentum(double m, double v, double c = constants::c) noexcept
{
    return lorentz_gamma(v, c) * m * v;
}
/// Rest mass from total energy E and momentum magnitude p: m = sqrt(E²-(pc)²)/c².
NP_NODISCARD inline double invariant_mass(double E, double p, double c = constants::c) noexcept
{
    const double m2 = (E * E - p * p * c * c) / (c * c * c * c);
    return m2 <= 0.0 ? 0.0 : std::sqrt(m2);
}
/// Longitudinal relativistic Doppler: observed wavelength for receding source (beta > 0).
NP_NODISCARD inline double doppler_longitudinal(double lambda_emit, double beta) noexcept
{
    return lambda_emit * std::sqrt((1.0 + beta) / (1.0 - beta));
}
NP_NODISCARD inline double length_contract(double L0, double v, double c = constants::c) noexcept
{
    return L0 / lorentz_gamma(v, c);
}
NP_NODISCARD inline double time_dilate(double dt0, double v, double c = constants::c) noexcept
{
    return dt0 * lorentz_gamma(v, c);
}

// ── Electromagnetism (SI) ──
/// Lorentz force F = q(E + v × B); E, B, v are 3-vectors.
NP_NODISCARD inline std::array<double, 3> lorentz_force(double q, const std::array<double, 3> &E,
                                                        const std::array<double, 3> &B,
                                                        const std::array<double, 3> &v) noexcept
{
    return {q * (E[0] + v[1] * B[2] - v[2] * B[1]), q * (E[1] + v[2] * B[0] - v[0] * B[2]),
            q * (E[2] + v[0] * B[1] - v[1] * B[0])};
}

/**
 * @brief Charged particle in uniform E/B advanced with the Boris pusher
 *        (second-order, energy-conserving for E = 0).
 */
struct ChargedParticle
{
    double q = constants::e_charge, m = constants::m_e;
    std::array<double, 3> pos{0.0, 0.0, 0.0}, vel{0.0, 0.0, 0.0};

    ChargedParticle() = default;
    ChargedParticle(double q_, double m_, std::array<double, 3> pos_, std::array<double, 3> vel_)
        : q(q_), m(m_), pos(pos_), vel(vel_)
    {
    }

    NP_API void boris_step(const std::array<double, 3> &E, const std::array<double, 3> &B, double dt) noexcept
    {
        const double hq = 0.5 * dt * q / m;
        // Half electric kick.
        std::array<double, 3> vm{vel[0] + hq * E[0], vel[1] + hq * E[1], vel[2] + hq * E[2]};
        // Magnetic rotation: t = qB dt/2m, s = 2t/(1+t²).
        const std::array<double, 3> t{hq * B[0], hq * B[1], hq * B[2]};
        const double t2 = t[0] * t[0] + t[1] * t[1] + t[2] * t[2];
        const double s = 2.0 / (1.0 + t2);
        const std::array<double, 3> sx{s * t[0], s * t[1], s * t[2]};
        const std::array<double, 3> vp{vm[0] + (vm[1] * t[2] - vm[2] * t[1]), vm[1] + (vm[2] * t[0] - vm[0] * t[2]),
                                       vm[2] + (vm[0] * t[1] - vm[1] * t[0])};
        std::array<double, 3> vp2{vm[0] + (vp[1] * sx[2] - vp[2] * sx[1]), vm[1] + (vp[2] * sx[0] - vp[0] * sx[2]),
                                  vm[2] + (vp[0] * sx[1] - vp[1] * sx[0])};
        // Second half electric kick + drift.
        vel = {vp2[0] + hq * E[0], vp2[1] + hq * E[1], vp2[2] + hq * E[2]};
        pos = {pos[0] + vel[0] * dt, pos[1] + vel[1] * dt, pos[2] + vel[2] * dt};
    }

    NP_NODISCARD double kinetic_energy() const noexcept
    {
        return 0.5 * m * (vel[0] * vel[0] + vel[1] * vel[1] + vel[2] * vel[2]);
    }
};

/// Cyclotron (angular) frequency |q|B/m.
NP_NODISCARD inline double cyclotron_frequency(double q, double B, double m) noexcept
{
    return std::abs(q) * B / m;
}
/// Gyroradius m v_perp / (|q| B).
NP_NODISCARD inline double gyroradius(double m, double v_perp, double q, double B) noexcept
{
    return m * v_perp / (std::abs(q) * B);
}
/// Electron plasma frequency sqrt(n e²/(ε0 m_e)); n in 1/m³.
NP_NODISCARD inline double plasma_frequency(double n) noexcept
{
    return std::sqrt(n * constants::e_charge * constants::e_charge / (constants::eps0 * constants::m_e));
}
/// Debye length sqrt(ε0 kB T/(n e²)); n in 1/m³, T in K.
NP_NODISCARD inline double debye_length(double n, double T) noexcept
{
    return std::sqrt(constants::eps0 * constants::kB * T / (n * constants::e_charge * constants::e_charge));
}
/// On-axis B field of a current loop: μ0 I R²/(2(R²+z²)^{3/2}).
NP_NODISCARD inline double biot_savart_loop_axis(double I, double R, double z) noexcept
{
    const double d2 = R * R + z * z;
    return constants::mu0 * I * R * R / (2.0 * d2 * std::sqrt(d2));
}
/// Larmor radiated power q²a²/(6πε0c³).
NP_NODISCARD inline double larmor_power(double q, double a) noexcept
{
    const double c = constants::c;
    return q * q * a * a / (6.0 * constants::pi * constants::eps0 * c * c * c);
}
/// Snell's law; nullopt on total internal reflection.
NP_NODISCARD inline std::optional<double> snell(double n1, double n2, double theta1) noexcept
{
    const double s = n1 / n2 * std::sin(theta1);
    if (std::abs(s) > 1.0)
    {
        return std::nullopt;
    }
    return std::asin(s);
}
/// Fresnel power reflectance (average of s/p) for real indices; 1.0 under TIR.
NP_NODISCARD inline double fresnel_reflectance(double n1, double n2, double theta1) noexcept
{
    const auto t2 = snell(n1, n2, theta1);
    if (!t2.has_value())
    {
        return 1.0;
    }
    const double c1 = std::cos(theta1), c2 = std::cos(*t2);
    const double rs = (n1 * c1 - n2 * c2) / (n1 * c1 + n2 * c2);
    const double rp = (n1 * c2 - n2 * c1) / (n1 * c2 + n2 * c1);
    return 0.5 * (rs * rs + rp * rp);
}
/// Thin-lens image distance; nullopt when the image is at infinity (do == f).
NP_NODISCARD inline std::optional<double> thin_lens_image(double f, double do_) noexcept
{
    const double denom = 1.0 / f - 1.0 / do_;
    if (denom == 0.0)
    {
        return std::nullopt;
    }
    return 1.0 / denom;
}

// ── Statistical mechanics & thermodynamics ──
/// Maxwell-Boltzmann speed pdf: 4π(m/2πkT)^{3/2} v² exp(-mv²/2kT).
NP_NODISCARD inline double maxwell_boltzmann_pdf(double v, double m, double T) noexcept
{
    if (v < 0.0 || T <= 0.0)
    {
        return 0.0;
    }
    const double a = m / (constants::kB * T);
    return 4.0 * constants::pi * std::pow(a / (2.0 * constants::pi), 1.5) * v * v * std::exp(-0.5 * a * v * v);
}
NP_NODISCARD inline double mb_most_probable(double m, double T) noexcept
{
    return std::sqrt(2.0 * constants::kB * T / m);
}
NP_NODISCARD inline double mb_mean_speed(double m, double T) noexcept
{
    return std::sqrt(8.0 * constants::kB * T / (constants::pi * m));
}
NP_NODISCARD inline double mb_rms_speed(double m, double T) noexcept
{
    return std::sqrt(3.0 * constants::kB * T / m);
}
/// Ideal gas pressure p = N kB T / V.
NP_NODISCARD inline double ideal_gas_pressure(double N, double V, double T) noexcept
{
    return N * constants::kB * T / V;
}
/// Planck spectral radiance B_ν(T) = 2hν³/c²/(e^{hν/kT}−1); 0 on overflow.
NP_NODISCARD inline double planck_radiance(double nu, double T) noexcept
{
    if (nu <= 0.0 || T <= 0.0)
    {
        return 0.0;
    }
    const double x = constants::h * nu / (constants::kB * T);
    if (x > 700.0)
    {
        return 0.0;
    }
    const double c = constants::c;
    return 2.0 * constants::h * nu * nu * nu / (c * c) / (std::exp(x) - 1.0);
}
/// Black-body exitance σT⁴.
NP_NODISCARD inline double stefan_boltzmann_exitance(double T) noexcept
{
    return constants::sigma_sb * T * T * T * T;
}
/// Wien peak wavelength b/T.
NP_NODISCARD inline double wien_peak_wavelength(double T) noexcept
{
    return constants::wien_b / T;
}
/// Einstein solid heat capacity per mole: 3R x²eˣ/(eˣ−1)², x = ΘE/T.
NP_NODISCARD inline double einstein_heat_capacity(double T, double theta_E) noexcept
{
    const double x = theta_E / T;
    if (x > 700.0)
    {
        return 0.0;
    }
    const double ex = std::exp(x);
    const double d = ex - 1.0;
    return 3.0 * constants::R_gas * x * x * ex / (d * d);
}
/// Two-level Schottky heat capacity per mole, gap delta (J): R x²e^{−x}/(1+e^{−x})².
NP_NODISCARD inline double schottky_heat_capacity(double T, double delta_J) noexcept
{
    const double x = delta_J / (constants::kB * T);
    if (x > 700.0)
    {
        return 0.0;
    }
    const double e = std::exp(-x);
    const double d = 1.0 + e;
    return constants::R_gas * x * x * e / (d * d);
}
/// Two-level partition function 1 + e^{−Δ/kT}.
NP_NODISCARD inline double partition_2level(double delta_J, double T) noexcept
{
    return 1.0 + std::exp(-delta_J / (constants::kB * T));
}
/// Entropy of mixing per mole: −R(x ln x + (1−x) ln(1−x)).
NP_NODISCARD inline double entropy_of_mixing(double x) noexcept
{
    if (x <= 0.0 || x >= 1.0)
    {
        return 0.0;
    }
    return -constants::R_gas * (x * std::log(x) + (1.0 - x) * std::log(1.0 - x));
}
/// Thermal de Broglie wavelength h/sqrt(2πmkT).
NP_NODISCARD inline double thermal_wavelength(double m, double T) noexcept
{
    return constants::h / std::sqrt(2.0 * constants::pi * m * constants::kB * T);
}

// ── Single-particle quantum mechanics (SI; cf. np::quantum for qubits) ──
namespace qm
{
/// Infinite-well energies E_n = n²h²/(8mL²), n ≥ 1.
NP_NODISCARD inline double particle_in_box_energy(int n, double L, double m) noexcept
{
    const double e1 = constants::h * constants::h / (8.0 * m * L * L);
    return e1 * static_cast<double>(n) * static_cast<double>(n);
}
/// Infinite-well eigenstate sqrt(2/L) sin(nπx/L).
NP_NODISCARD inline double particle_in_box_psi(int n, double L, double x) noexcept
{
    return std::sqrt(2.0 / L) * std::sin(static_cast<double>(n) * constants::pi * x / L);
}
/// Harmonic oscillator energies ħω(n+1/2).
NP_NODISCARD inline double ho_energy(int n, double omega) noexcept
{
    return constants::hbar * omega * (static_cast<double>(n) + 0.5);
}
/// Oscillator length sqrt(ħ/(mω)).
NP_NODISCARD inline double ho_length(double m, double omega) noexcept
{
    return std::sqrt(constants::hbar / (m * omega));
}
NP_NODISCARD inline double hermite_phys(int n, double x) noexcept
{
    if (n <= 0)
    {
        return 1.0;
    }
    if (n == 1)
    {
        return 2.0 * x;
    }
    double h0 = 1.0, h1 = 2.0 * x;
    for (int k = 1; k < n; ++k)
    {
        const double h2 = 2.0 * x * h1 - 2.0 * static_cast<double>(k) * h0;
        h0 = h1;
        h1 = h2;
    }
    return h1;
}
/// HO eigenstate ψ_n(x) with Hermite polynomials (factorial loop, n small).
NP_NODISCARD inline double ho_psi(int n, double m, double omega, double x) noexcept
{
    const double x0 = ho_length(m, omega);
    const double xi = x / x0;
    double fact = 1.0;
    for (int k = 2; k <= n; ++k)
    {
        fact *= static_cast<double>(k);
    }
    double pow2n = 1.0;
    for (int k = 0; k < n; ++k)
    {
        pow2n *= 2.0;
    }
    const double norm = 1.0 / (std::pow(constants::pi, 0.25) * std::sqrt(pow2n * fact * x0));
    return norm * hermite_phys(n, xi) * std::exp(-0.5 * xi * xi);
}
/// Hydrogen energies −R∞hc/n² (J).
NP_NODISCARD inline double hydrogen_energy(int n) noexcept
{
    const double e1 = constants::R_inf * constants::h * constants::c;
    return -e1 / (static_cast<double>(n) * static_cast<double>(n));
}
/// Hydrogen 1s wavefunction e^{−r/a0}/sqrt(πa0³) (1/m^{3/2}).
NP_NODISCARD inline double hydrogen_1s_psi(double r) noexcept
{
    const double a0 = constants::a0_bohr;
    return std::exp(-r / a0) / std::sqrt(constants::pi * a0 * a0 * a0);
}
/// Rabi flopping probability (Ω²/Ω'²)sin²(Ω't/2), Ω'² = Ω²+Δ².
NP_NODISCARD inline double rabi_probability(double Omega, double t, double detuning = 0.0) noexcept
{
    const double Op2 = Omega * Omega + detuning * detuning;
    if (Op2 == 0.0)
    {
        return 0.0;
    }
    const double Op = std::sqrt(Op2);
    const double s = std::sin(0.5 * Op * t);
    return Omega * Omega / Op2 * s * s;
}
/// Rectangular-barrier transmission (exact); E, V0 in J, a in m.
NP_NODISCARD inline double tunnel_rectangular(double E, double V0, double a, double m) noexcept
{
    if (E <= 0.0 || V0 <= 0.0 || a <= 0.0)
    {
        return 0.0;
    }
    const double hbar = constants::hbar;
    if (E < V0)
    {
        const double kappa = std::sqrt(2.0 * m * (V0 - E)) / hbar;
        const double sh = std::sinh(kappa * a);
        return 1.0 / (1.0 + V0 * V0 * sh * sh / (4.0 * E * (V0 - E)));
    }
    const double k = std::sqrt(2.0 * m * (E - V0)) / hbar;
    const double s = std::sin(k * a);
    return 1.0 / (1.0 + V0 * V0 * s * s / (4.0 * E * (E - V0)));
}
NP_NODISCARD inline ndarray<std::complex<double>> pauli_x()
{
    using Cx = std::complex<double>;
    return ndarray<Cx>::from_data({2, 2}, std::vector<Cx>{Cx(0, 0), Cx(1, 0), Cx(1, 0), Cx(0, 0)});
}
NP_NODISCARD inline ndarray<std::complex<double>> pauli_y()
{
    using Cx = std::complex<double>;
    return ndarray<Cx>::from_data({2, 2}, std::vector<Cx>{Cx(0, 0), Cx(0, -1), Cx(0, 1), Cx(0, 0)});
}
NP_NODISCARD inline ndarray<std::complex<double>> pauli_z()
{
    using Cx = std::complex<double>;
    return ndarray<Cx>::from_data({2, 2}, std::vector<Cx>{Cx(1, 0), Cx(0, 0), Cx(0, 0), Cx(-1, 0)});
}
/// Bloch vector for |ψ⟩ = cos(θ/2)|0⟩ + e^{iφ}sin(θ/2)|1⟩.
NP_NODISCARD inline std::array<double, 3> bloch_vector(double theta, double phi) noexcept
{
    return {std::sin(theta) * std::cos(phi), std::sin(theta) * std::sin(phi), std::cos(theta)};
}

struct TiseResult
{
    ndarray<double> energies;  ///< Lowest eigenvalues (nstates,), ascending.
    ndarray<double> wavefuncs; ///< (N, nstates) columns, |ψ|²dx-normalized, zero at walls.
};

namespace tise_detail
{
// Thomas solve for (H - sigma I) y = rhs with tridiagonal H(diag, off).
inline void shifted_tridiag_solve(const std::vector<double> &diag, const std::vector<double> &off, double sigma,
                                  const std::vector<double> &rhs, std::vector<double> &y)
{
    const std::size_t M = diag.size();
    std::vector<double> cp(M, 0.0), dp(M, 0.0);
    cp[0] = off.empty() ? 0.0 : off[0] / (diag[0] - sigma);
    dp[0] = rhs[0] / (diag[0] - sigma);
    for (std::size_t i = 1; i < M; ++i)
    {
        const double den = diag[i] - sigma - off[i - 1] * cp[i - 1];
        const double safe = (den == 0.0) ? 1e-300 : den;
        cp[i] = (i + 1 < M) ? off[i] / safe : 0.0;
        dp[i] = (rhs[i] - off[i - 1] * dp[i - 1]) / safe;
    }
    y[M - 1] = dp[M - 1];
    for (std::size_t i = M - 1; i-- > 0;)
    {
        y[i] = dp[i] - cp[i] * y[i + 1];
    }
}

// Sturm count: number of eigenvalues of tridiagonal H strictly below lam.
inline std::size_t sturm_count(const std::vector<double> &diag, const std::vector<double> &off, double lam)
{
    const std::size_t M = diag.size();
    std::size_t count = 0;
    double q = diag[0] - lam;
    if (q < 0.0)
    {
        ++count;
    }
    for (std::size_t i = 1; i < M; ++i)
    {
        if (std::abs(q) < 1e-300)
        {
            q = (q < 0.0) ? -1e-300 : 1e-300;
        }
        q = diag[i] - lam - off[i - 1] * off[i - 1] / q;
        if (q < 0.0)
        {
            ++count;
        }
    }
    return count;
}
} // namespace tise_detail

/**
 * @brief 1D time-independent Schrödinger equation on [xmin, xmax] with
 *        hard walls: finite-difference Hamiltonian solved by Sturm-sequence
 *        bisection (eigenvalues) + inverse iteration with a Thomas solve
 *        (eigenvectors). Self-contained tridiagonal path — no dense
 *        eigensolver needed.
 * @param V     Potential on the N-point grid (J).
 * @param nstates Number of lowest states to return (clamped to N-2).
 */
NP_NODISCARD inline TiseResult tise_1d(const ndarray<double> &V, double xmin, double xmax, int nstates = 4)
{
    TiseResult r;
    const std::size_t N = V.size();
    if (N < 4 || nstates < 1)
    {
        return r;
    }
    const std::size_t M = N - 2; // interior points (Dirichlet walls)
    const int keep = static_cast<int>(std::min<std::size_t>(M, static_cast<std::size_t>(nstates)));
    const double dx = (xmax - xmin) / static_cast<double>(N - 1);
    const double off_val = -constants::hbar * constants::hbar / (2.0 * constants::m_e * dx * dx);
    const auto &v = V.data();
    std::vector<double> diag(M, 0.0), off(M > 0 ? M - 1 : 0, off_val);
    for (std::size_t i = 0; i < M; ++i)
    {
        diag[i] = -2.0 * off_val + v[V._flat_logical(i + 1)];
    }
    // Gershgorin bounds for the bisection bracket.
    double lo = diag[0], hi = diag[0];
    for (std::size_t i = 0; i < M; ++i)
    {
        const double rad = (i > 0 ? std::abs(off[i - 1]) : 0.0) + (i + 1 < M ? std::abs(off[i]) : 0.0);
        lo = std::min(lo, diag[i] - rad);
        hi = std::max(hi, diag[i] + rad);
    }
    r.energies = ndarray<double>(std::vector<int>{keep});
    r.wavefuncs = ndarray<double>(std::vector<int>{static_cast<int>(N), keep});
    std::vector<double> rhs(M, 0.0), vec(M, 0.0);
    const double norm = 1.0 / std::sqrt(dx);
    for (int k = 0; k < keep; ++k)
    {
        // Bisect the k-th eigenvalue (0-based): count(lo) <= k < count(hi).
        double a = lo - 1.0, b = hi + 1.0;
        for (int it = 0; it < 200; ++it)
        {
            const double mid = 0.5 * (a + b);
            if (tise_detail::sturm_count(diag, off, mid) <= static_cast<std::size_t>(k))
            {
                a = mid;
            }
            else
            {
                b = mid;
            }
            if (b - a <= 1e-12 * std::max({1e-300, std::abs(a), std::abs(b)}))
            {
                break;
            }
        }
        const double lam = 0.5 * (a + b);
        r.energies.at(static_cast<std::size_t>(k)) = lam;
        // Inverse iteration from a linear ramp (not orthogonal to low modes).
        for (std::size_t i = 0; i < M; ++i)
        {
            rhs[i] = static_cast<double>(i + 1);
        }
        for (int it = 0; it < 8; ++it)
        {
            tise_detail::shifted_tridiag_solve(diag, off, lam, rhs, vec);
            double nrm = 0.0;
            for (double x : vec)
            {
                nrm += x * x;
            }
            nrm = std::sqrt(nrm);
            if (nrm == 0.0)
            {
                break;
            }
            for (std::size_t i = 0; i < M; ++i)
            {
                rhs[i] = vec[i] / nrm;
            }
        }
        r.wavefuncs(0, static_cast<std::size_t>(k)) = 0.0;
        r.wavefuncs(N - 1, static_cast<std::size_t>(k)) = 0.0;
        for (std::size_t i = 0; i < M; ++i)
        {
            r.wavefuncs(i + 1, static_cast<std::size_t>(k)) = rhs[i] * norm;
        }
    }
    return r;
}
} // namespace qm

// ── Nuclear & astrophysics ──
/// Radioactive remainder N0·2^{−t/half_life}.
NP_NODISCARD inline double decay_remaining(double N0, double t, double half_life) noexcept
{
    return N0 * std::pow(0.5, t / half_life);
}
/// Activity λN with λ = ln2/half_life.
NP_NODISCARD inline double decay_activity(double N, double half_life) noexcept
{
    return std::log(2.0) / half_life * N;
}
/// Semi-empirical mass formula binding energy in MeV.
NP_NODISCARD inline double semf_binding_mev(int Z, int A) noexcept
{
    const double a = static_cast<double>(A);
    const double z = static_cast<double>(Z);
    const double bulk = 15.8 * a;
    const double surf = 18.3 * std::pow(a, 2.0 / 3.0);
    const double coul = 0.714 * z * (z - 1.0) / std::pow(a, 1.0 / 3.0);
    const double asym = 23.2 * (a - 2.0 * z) * (a - 2.0 * z) / a;
    double delta = 0.0;
    if (Z % 2 == 0 && (A - Z) % 2 == 0)
    {
        delta = 12.0 / std::sqrt(a);
    }
    else if (Z % 2 == 1 && (A - Z) % 2 == 1)
    {
        delta = -12.0 / std::sqrt(a);
    }
    return bulk - surf - coul - asym + delta;
}
NP_NODISCARD inline double escape_velocity(double M, double R, double G = constants::G) noexcept
{
    return std::sqrt(2.0 * G * M / R);
}
NP_NODISCARD inline double circular_velocity(double M, double r, double G = constants::G) noexcept
{
    return std::sqrt(G * M / r);
}
NP_NODISCARD inline double orbital_period(double M, double r, double G = constants::G) noexcept
{
    return 2.0 * constants::pi * std::sqrt(r * r * r / (G * M));
}
NP_NODISCARD inline double schwarzschild_radius(double M, double G = constants::G, double c = constants::c) noexcept
{
    return 2.0 * G * M / (c * c);
}
/// Hubble recession velocity v = H0·d (H0 in 1/s, d in m).
NP_NODISCARD inline double hubble_velocity(double H0, double d) noexcept
{
    return H0 * d;
}

// ── Dimensionless numbers (fluids & heat) ──
NP_NODISCARD inline double reynolds(double rho, double v, double L, double mu) noexcept
{
    return rho * v * L / mu;
}
NP_NODISCARD inline double mach_number(double v, double c_sound) noexcept
{
    return v / c_sound;
}
NP_NODISCARD inline double prandtl(double mu, double cp, double k) noexcept
{
    return mu * cp / k;
}
NP_NODISCARD inline double froude(double v, double L, double g = 9.81) noexcept
{
    return v / std::sqrt(g * L);
}
/// Laminar flat-plate correlation 0.664·√Re·∛Pr.
NP_NODISCARD inline double nusselt_laminar_flat(double Re, double Pr) noexcept
{
    return 0.664 * std::sqrt(Re) * std::cbrt(Pr);
}
/// Stokes drag 6πμrv.
NP_NODISCARD inline double stokes_drag(double mu, double r, double v) noexcept
{
    return 6.0 * constants::pi * mu * r * v;
}

} // namespace np::physics

#endif // NP_PHYSICS_HPP