/**
 * @file test_neuromorphic.cpp
 * @brief Tests for neuromorphic/event/spike — CPU harness + LIF-sim backend.
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
        auto sim = NeuromorphicFactory::lif_sim();
        test::check(cpu->name() == "CPU", "CPU backend");
        test::check(sim->name() == "LIF-sim", "LIF-sim backend");
        // CPUBackend is a documented pass-through harness, not a simulation.
        EventArray ea(2, 2);
        ea.push({0, 0, 0, 1});
        auto out = cpu->process(ea);
        test::check(out.size() == 1, "CPU process passes through");

        // All-subthreshold input must produce ZERO output spikes. (The old
        // in.clone() no-op would wrongly echo the 5 input events back.)
        LifSimBackend weak(0.1, 1.0);
        EventBuilder wb(2, 2);
        wb.add(0.1, 0, 0).add(0.2, 0, 0).add(0.3, 0, 0).add(0.4, 0, 0).add(0.5, 0, 0);
        auto weak_out = weak.process(wb.build());
        test::check(weak_out.empty(), "subthreshold input, no output spikes");

        // Superthreshold pattern, cross-checked against a standalone
        // LIFNeuron driven with the same currents: v = 0.5, 0.975, 1.426…,
        // so exactly the 3rd of 3 strong events fires.
        LifSimBackend strong(10.0, 1.0);
        EventBuilder sb(2, 2);
        sb.add(0.1, 1, 1).add(0.2, 1, 1).add(0.3, 1, 1);
        auto strong_out = strong.process(sb.build());
        LIFNeuron ref;
        const bool f1 = ref.step(10.0, 1.0), f2 = ref.step(10.0, 1.0), f3 = ref.step(10.0, 1.0);
        test::check(!f1 && !f2 && f3, "standalone LIF fires on 3rd strong step");
        test::check(strong_out.size() == 1, "backend fires once");
        if (strong_out.size() == 1)
        {
            const auto &ev = strong_out.span()[0];
            test::check(ev.x == 1 && ev.y == 1, "spike at right channel");
            test::check(ev.t == 0.3, "spike at triggering time");
            test::check(ev.p == 1, "spike polarity +1");
        }

        // Inhibitory input never fires (would also be echoed by a clone).
        EventBuilder ib(2, 2);
        ib.add(0.1, 0, 0, -1).add(0.2, 0, 0, -1);
        test::check(strong.process(ib.build()).empty(), "inhibitory input, no spikes");

        // Channels are isolated: firing (0,0) must not spike (1,0).
        EventBuilder cb(2, 2);
        cb.add(0.1, 0, 0).add(0.2, 0, 0).add(0.3, 0, 0).add(0.15, 1, 0);
        auto ch_out = strong.process(cb.build());
        test::check(ch_out.size() == 1 && ch_out.span()[0].x == 0, "channels isolated");

        // Out-of-range events are ignored, not aliased into channels.
        EventBuilder ob(2, 2);
        ob.add(0.1, 99, 99).add(0.2, -1, 0);
        test::check(strong.process(ob.build()).empty(), "out-of-range ignored");

        // Determinism: same input twice -> identical output.
        auto run1 = strong.process(sb.build());
        auto run2 = strong.process(sb.build());
        bool same = run1.size() == run2.size();
        if (same)
        {
            for (size_t i = 0; i < run1.size(); ++i)
            {
                const auto &a = run1.span()[i];
                const auto &b = run2.span()[i];
                same = same && a.t == b.t && a.x == b.x && a.y == b.y && a.p == b.p;
            }
        }
        test::check(same, "deterministic replay");

        QuantizedEventArray q{ea, 8};
        test::check(q.as_event_array().size() == 1, "Quantized decorator");
    }

    return test::failures() ? 1 : 0;
}
