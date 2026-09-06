/**
 * @example neuromorphic_snn.cpp
 * Spiking neural network on Loihi2 / CPU via np::neuromorphic
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

    // 3. Backend Strategy (Loihi2 vs CPU)
    auto cpu = NeuromorphicFactory::cpu();
    auto loihi = NeuromorphicFactory::loihi();
    EventBuilder b(10, 10);
    b.add(0.1, 1, 1).add(0.2, 2, 2);
    auto ea = b.build();
    std::cout << cpu->name() << " " << cpu->process(ea).size() << "\n";
    std::cout << loihi->name() << " " << loihi->process(ea).size() << "\n";

    // 4. STDP
    STDP stdp;
    std::cout << "STDP dt=10 " << stdp.weight_update(10) << "\n";
    return 0;
}
