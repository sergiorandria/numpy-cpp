/**
 * @file test_memory.cpp
 */
#include "test_util.hpp"
#include <np/np.hpp>
int main()
{
    using namespace np::mem;
    auto a = np::zeros<double>({2, 2});
    auto h = migrate_to_hbm(a);
    test::check(h.size() == 4, "HBM size");
    auto b = migrate_to_host(h);
    test::check(b.size() == 4, "host migrate");
    auto z = zeros_hbm<double>({2, 2});
    test::check(z.size() == 4, "zeros_hbm");
    auto h2 = MemoryFactory::hbm(a);
    test::check(h2.space == MemorySpace::HBM, "factory HBM");
    auto c = MemoryFactory::cxl(a);
    test::check(c.space == MemorySpace::CXL, "factory CXL");
    return test::failures() ? 1 : 0;
}
