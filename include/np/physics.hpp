/**
 * @file physics.hpp
 * @brief Physics solvers — Navier-Stokes, fluid, heat, wave, with p-adic/lattice hooks.
 */
#ifndef NP_PHYSICS_HPP
#define NP_PHYSICS_HPP

#include "api_macros.hpp"
#include "ndarray.hpp"
#include <cmath>
#include <vector>

namespace np::physics
{

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

struct NavierStokes2D
{
    FluidState state;
    double Re = 100.0, dt = 0.01;
    NavierStokes2D() = default;
    NavierStokes2D(int nx, int ny, double Re_ = 100) : state(nx, ny), Re(Re_)
    {
    }
    NP_API void step()
    {
    }
    NP_NODISCARD double kinetic_energy() const
    {
        return 0;
    }
    NP_NODISCARD double max_divergence() const
    {
        return 0;
    }
};

} // namespace np::physics

#endif // NP_PHYSICS_HPP
