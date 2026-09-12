/**
 * @file variety.hpp
 * @brief Deprecated alias for manifold.hpp (correct name).
 *
 * `variety` is now `manifold` (correct term for smooth manifolds;
 * `variety` is reserved for algebraic varieties). This header keeps
 * backward compatibility: `np::variety::*` aliases `np::manifold::*`.
 *
 * New code should include <np/manifold.hpp> and use `np::manifold::`.
 *
 * @author Sergio Randriamihoatra (sergiorandriamihoatra@gmail.com)
 */
#ifndef NP_VARIETY_HPP
#define NP_VARIETY_HPP

#include "api_macros.hpp"
#include "manifold.hpp"

#endif // NP_VARIETY_HPP
