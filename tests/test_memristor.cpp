/**
 * @file test_memristor.cpp
 */
#include "test_util.hpp"
#include <np/np.hpp>
int main()
{
    using namespace np::analog;
    auto w = np::eye<float>(2);
    Crossbar cb(w);
    auto x = np::ndarray<float>(std::vector<int>{2});
    x[0] = 1;
    x[1] = 2;
    auto y = cb.dot(x);
    test::check(y.size() == 2, "crossbar dot");
    auto q = cb.quantize(4);
    test::check(q.size() == 4, "quantize");
    auto cb2 = ReRAMFactory::crossbar(w);
    test::check(cb2.weights.size() == 4, "factory");
    return test::failures() ? 1 : 0;
}
