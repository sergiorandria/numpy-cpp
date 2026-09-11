/**
 * @file test_memory.cpp
 */
#include "test_util.hpp"
#include <np/np.hpp>
int main()
{
    using namespace np::mem;
    auto a = np::zeros<double>({2, 2});
    auto h = tag_hbm_hint(a);
    test::check(h.size() == 4, "hint size");
    test::check(h.space == MemorySpace::HBM, "hint space tag preserved");
    // The tag aliases host storage (no device placement involved).
    test::check(h.span().data() == h.data.data().data(), "hint spans host buffer");
    auto b = migrate_to_host(h);
    test::check(b.size() == 4, "host migrate");
    auto z = zeros_hinted<double>({2, 2}, MemorySpace::HBM);
    test::check(z.size() == 4, "zeros_hinted");
    auto h2 = MemoryFactory::hbm(a);
    test::check(h2.space == MemorySpace::HBM, "factory HBM");
    auto c = MemoryFactory::cxl(a);
    test::check(c.space == MemorySpace::CXL, "factory CXL");
    auto d = MemoryFactory::device(a);
    test::check(d.space == MemorySpace::Device, "factory Device");
    return test::failures() ? 1 : 0;
}
