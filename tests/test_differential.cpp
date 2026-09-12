/**
 * @file test_differential.cpp
 * @brief Tests for differential forms, exterior derivative, VM/LLVM JIT.
 */
#include "test_util.hpp"
#include <np/np.hpp>

int main()
{
    using namespace np::differential;

    // ── VM parsing and eval ───────────────────────────────────────────────
    {
        VM vm("x^2 + y^2", {"x", "y"});
        test::check(std::abs(vm.eval({3, 4}) - 25.0) < 1e-9, "VM eval x^2+y^2");
        VM dx = vm.derivative_vm(0); // ∂/∂x = 2x
        test::check(std::abs(dx.eval({3, 4}) - 6.0) < 1e-9, "VM derivative 2x");
        VM dy = vm.derivative_vm(1);
        test::check(std::abs(dy.eval({3, 4}) - 8.0) < 1e-9, "VM derivative 2y");
    }
    {
        VM vm("sin(x) * cos(y)", {"x", "y"});
        double v = vm.eval({0, 0});
        test::check(std::abs(v - 0.0) < 1e-9, "VM sin*cos");
        VM dx = vm.derivative_vm(0);
        // derivative w.r.t x is cos(x)cos(y)
        test::check(std::abs(dx.eval({0, 0}) - 1.0) < 1e-9, "VM derivative cos*cos");
    }
    {
        VM vm("exp(x) + log(y)", {"x", "y"});
        test::check(std::abs(vm.eval({0, 1}) - 1.0) < 1e-9, "VM exp+log");
    }
    {
        // laplacian() builds the tree directly (an earlier revision
        // rebuilt from to_string() fragments, which always threw because
        // differentiated exprs like "x^2'_d0'_d0" don't parse).
        VM vm("x^2 + y^2", {"x", "y"});
        VM lap = kernel::laplacian(vm);
        test::check(std::abs(lap.eval({3, 4}) - 4.0) < 1e-9, "laplacian x^2+y^2 = 4");
        test::check(std::abs(kernel::laplacian_eval(vm, {1, 2}) - 4.0) < 1e-9, "laplacian_eval agrees");
        VM v1("x^3", {"x"});
        test::check(std::abs(kernel::laplacian(v1).eval({2}) - 12.0) < 1e-9, "laplacian x^3 = 6x");
    }

    // ── ScalarField + exterior_derivative (finite difference + VM) ───────
    {
        ScalarField f([](const Point &p) { return p[0] * p[0] + p[1] * p[1]; }, 2);
        auto df = exterior_derivative(f);
        test::check(df.dim == 2, "exterior_derivative dim");
        // df = 2x dx + 2y dy
        test::check(std::abs(df(Point{3, 4}, 0) - 6.0) < 1e-6, "df dx 2x");
        test::check(std::abs(df(Point{3, 4}, 1) - 8.0) < 1e-6, "df dy 2y");
    }
    {
        // Via VM symbolic
        VM vm("x^2 + y^2", {"x", "y"});
        auto df = exterior_derivative_vm(vm, {"x", "y"});
        test::check(df.dim == 2, "exterior_derivative_vm dim");
        test::check(std::abs(df(Point{3, 4}, 0) - 6.0) < 1e-9, "VM df dx");
        test::check(std::abs(df(Point{3, 4}, 1) - 8.0) < 1e-9, "VM df dy");

        // Wedge
        OneForm a(2), b(2);
        a.comps[0] = ScalarField([](const Point &p) { return p[0]; }, 2);
        a.comps[1] = ScalarField([](const Point &p) { return p[1]; }, 2);
        b.comps[0] = ScalarField([](const Point &p) { return -p[1]; }, 2);
        b.comps[1] = ScalarField([](const Point &p) { return p[0]; }, 2);
        auto w = wedge(a, b);
        test::check(w.k == 2 && w.dim == 2, "wedge k=2");
        // coefficient for dx∧dy is a_x b_y - a_y b_x = x*x - y*(-y) = x^2 + y^2
        double c = w.coeffs.at({0, 1})(Point{3, 4});
        test::check(std::abs(c - 25.0) < 1e-9, "wedge coeff");
        // Pullback convention: J[i][j] = d phi_j / d x_i. phi(x,y)=(2x,y),
        // omega = x dx: (phi*omega)(v) = 2px*(2vx), so comps = [4x, 0].
        OneForm o(2);
        o.comps[0] = ScalarField([](const Point &p) { return p[0]; }, 2);
        o.comps[1] = ScalarField([](const Point &p) { return 0.0; }, 2);
        std::function<Point(const Point &)> phi = [](const Point &p) -> Point { return Point{p[0] * 2, p[1]}; };
        std::function<std::vector<std::vector<double>>(const Point &)> dphi = [](const Point &) {
            return std::vector<std::vector<double>>{{2, 0}, {0, 1}};
        };
        auto pb = pullback(o, phi, dphi);
        // omega evaluated at phi(3,4)=(6,4) gives 6, times J[0][0]=2.
        test::check(std::abs(pb(Point{3, 4}, 0) - 12.0) < 1e-9, "pullback convention x");
        test::check(std::abs(pb(Point{3, 4}, 1) - 0.0) < 1e-9, "pullback convention y");
        // lie_derivative of a 0-form is a 0-form (was: OneForm with dim-1
        // components dropped). L_X(x^2+y^2) along (1,0) is 2x.
        ScalarField f2([](const Point &p) { return p[0] * p[0] + p[1] * p[1]; }, 2);
        auto lx = lie_derivative(f2, {1.0, 0.0});
        test::check(std::abs(lx(Point{3, 4}) - 6.0) < 1e-6, "lie scalar type+value");
    }

    // ── VM batch eval on ndarray ──────────────────────────────────────────
    {
        VM vm("x^2", {"x"});
        np::ndarray<double> pts(std::vector<int>{3, 1});
        pts(0, 0) = 1;
        pts(1, 0) = 2;
        pts(2, 0) = 3;
        auto out = vm.eval_batch(pts);
        test::check(out.size() == 3 && std::abs(out[0] - 1) < 1e-9 && std::abs(out[2] - 9) < 1e-9, "VM eval_batch");
    }

#if NP_HAS_LLVM_JIT
    test::check(true, "LLVM JIT available");
#else
    test::check(true, "VM fallback AD (no LLVM)");
#endif

    return test::failures() ? 1 : 0;
}
