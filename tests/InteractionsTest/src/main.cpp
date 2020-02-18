
#include <bindings/Interactions.hpp>

int main(int argc, char* argv[])
{

    std::ifstream params("input/InteractiveTest.params");
    std::stringstream buffer;
    buffer << params.rdbuf();

    std::map<std::string, std::string> args_dict = {
        { "NumThreads", "1" },
        { "LogFile", "InteractionsTest.log" }
    };

    PV::Interactions* interact = new PV::Interactions(args_dict, buffer.str());
    auto result = interact->begin();
    pvAssert(result == PV::Interactions::SUCCESS);

    PVLayerLoc loc;

    result = interact->getLayerShape("Input", &loc);
    pvAssert(result == PV::Interactions::SUCCESS);

    int numNeurons = loc.nbatch * loc.nx * loc.ny * loc.nf;
    int offset = loc.kb0 * (loc.nxGlobal * loc.nyGlobal * loc.nf)
        + loc.ky0 * (loc.nxGlobal * loc.nf)
        + loc.kx0 * loc.nf;

    float multiplier = 1.0f;
    int timestep = 0;
    while (!interact->isFinished()) {
        std::vector<float> src_data, dst_data;
        // create test data for each timestep
        for (int i = 0; i < numNeurons; ++i) {
            src_data.push_back(offset + i + timestep);
        }
        timestep++;

        // put the test data in the input layer
        result = interact->setLayerState("Input", &src_data);
        pvAssert(result == PV::Interactions::SUCCESS);

        // advance by one timestep
        result = interact->step(nullptr);
        pvAssert(result == PV::Interactions::SUCCESS);

        // Check the state of the input layer and make sure it did not change
        result = interact->getLayerState("Input", &dst_data);
        pvAssert(result == PV::Interactions::SUCCESS);
        for (int i = 0; i < numNeurons; ++i) {
            pvAssert(src_data[i] == dst_data[i]);
        }

        // Get the output data and make sure we get the input times the weights
        result = interact->getLayerActivity("Output", &dst_data);
        pvAssert(result == PV::Interactions::SUCCESS);
        pvAssert(src_data.size() == dst_data.size());
        for (int i = 0; i < numNeurons; ++i) {
            pvAssert(src_data[i] * multiplier == dst_data[i]);
        }
        multiplier *= 2.0f;

        // Double the value of nonzero weights every time
        std::vector<float> weights;
        result = interact->getConnectionWeights("InputToOutput", &weights);
        pvAssert(result == PV::Interactions::SUCCESS);
        for (int i = 0; i < weights.size(); ++i) {
            if (weights[i] != 0.0f) {
                weights[i] *= 2.0f;
            }
        }
        result = interact->setConnectionWeights("InputToOutput", &weights);
        pvAssert(result == PV::Interactions::SUCCESS);
    }
    result = interact->finish();
    pvAssert(result == PV::Interactions::SUCCESS);

    delete interact;

    return EXIT_SUCCESS;
}
