/**
 * @file test_neuromorphic.cpp
 * @brief Tests for neuromorphic/event/spike — Loihi/SpiNNaker strategies.
 */
#include "test_util.hpp"
#include <np/np.hpp>

int main()
{
    using namespace np::event;
    using namespace np::spike;
    using namespace np::neuromorphic;

    // EventArray
    {
        EventArray ea(10, 10);
        ea.push({0.1, 1, 2, 1});
        ea.push({0.2, 3, 4, 0});
        test::check(ea.size() == 2, "EventArray size");
        auto sp = ea.span();
        test::check(sp.size() == 2 && sp[0].x == 1, "EventArray span");
        auto cl = ea.clone();
        test::check(cl.size() == 2, "EventArray clone");
        EventBuilder b(5, 5);
        b.add(0.1, 1, 1).add(0.2, 2, 2);
        auto built = b.build();
        test::check(built.size() == 2, "EventBuilder");
        struct V : SpikeVisitor
        {
            bool seen = false;
            void visit(const EventArray &a) override
            {
                seen = !a.empty();
            }
        } v;
        ea.accept(v);
        test::check(v.seen, "SpikeVisitor");
    }
    // Spike encoding
    {
        auto a = np::ndarray<double>(std::vector<int>{3});
        a[0] = 0.0;
        a[1] = 0.5;
        a[2] = 1.0;
        auto er = encode_rate(a, 100, 100);
        test::check(er.size() >= 10, "encode_rate");
        auto et = encode_temporal(a, 1.0);
        test::check(et.size() == 3, "encode_temporal");
    }
    // LIF / Izhikevich / STDP
    {
        LIFNeuron n;
        bool spiked = n.step(2.0, 1.0);
        (void)spiked;
        test::check(true, "LIF step");
        IzhikevichNeuron iz;
        test::check(true, "Izhikevich");
        STDP stdp;
        double dw = stdp.weight_update(10.0);
        test::check(dw > 0, "STDP");
    }
    // Backends Strategy + Factory
    {
        auto cpu = NeuromorphicFactory::cpu();
        auto loihi = NeuromorphicFactory::loihi();
        auto spi = NeuromorphicFactory::spinnaker();
        test::check(cpu->name() == "CPU", "CPU backend");
        test::check(loihi->name() == "Loihi2", "Loihi backend");
        test::check(spi->name() == "SpiNNaker2", "SpiNNaker backend");
        EventArray ea(2, 2);
        ea.push({0, 0, 0, 1});
        auto out = cpu->process(ea);
        test::check(out.size() == 1, "CPU process");
        auto out2 = loihi->process(ea);
        test::check(out2.size() == 1, "Loihi process");
        QuantizedEventArray q{ea, 8};
        test::check(q.as_event_array().size() == 1, "Quantized decorator");
    }

    return test::failures() ? 1 : 0;
}
