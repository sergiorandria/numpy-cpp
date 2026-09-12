/**
 * @example neuromorphic_snn.cpp
 * Spiking neural network via np::neuromorphic software simulation
 * (CPU harness + per-channel LIF backend). No Loihi/SpiNNaker hardware
 * involved — see neuromorphic.hpp doc-block.
 */
#include <iostream>
#include <np/np.hpp>

int main()
{
    using namespace np::event;
    using namespace np::spike;
    using namespace np::neuromorphic;

    // 1. Encode ndarray -> EventArray (rate)
    auto img = np::ndarray<float>(std::vector<int>{4});
    img[0] = 0.0f;
    img[1] = 0.5f;
    img[2] = 0.8f;
    img[3] = 1.0f;
    auto spikes = encode_rate(img, 100, 100);
    std::cout << "spikes " << spikes.size() << "\n";

    // 2. LIF network
    LIFNeuron lif;
    int out_spikes = 0;
    for (auto &ev : spikes.span())
        if (lif.step(2.0))
            ++out_spikes;
    std::cout << "LIF out " << out_spikes << "\n";

    // 3. Backend Strategy (pass-through CPU harness vs real LIF simulation).
    // The backend IS the LIF pipeline now: same per-channel dynamics as the
    // manual loop in step 2, driven by event polarity instead of a constant.
    auto cpu = NeuromorphicFactory::cpu();
    LifSimBackend sim(10.0, 1.0);
    EventBuilder b(10, 10);
    // Three strong excitatory events on (1,1): fires once, on the 3rd.
    // One weak event on (2,2): below threshold, never fires.
    b.add(0.1, 1, 1).add(0.2, 1, 1).add(0.3, 1, 1).add(0.15, 2, 2);
    auto ea = b.build();
    std::cout << cpu->name() << " " << cpu->process(ea).size() << " (pass-through echo)\n";
    auto sim_out = sim.process(ea);
    std::cout << sim.name() << " " << sim_out.size() << " (simulated spikes)\n";
    for (auto &ev : sim_out.span())
        std::cout << "  spike t=" << ev.t << " (" << ev.x << "," << ev.y << ") p=" << ev.p << "\n";

    // 4. STDP
    STDP stdp;
    std::cout << "STDP dt=10 " << stdp.weight_update(10) << "\n";
    return 0;
}
