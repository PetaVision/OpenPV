
#include <bindings/Interactions.hpp>

int main(int argc, char *argv[]) {

   std::ifstream params("input/InteractiveTest.params");
   std::stringstream buffer;
   buffer << params.rdbuf();

   std::map<std::string, std::string> args_dict = {
      {"NumThreads", "1"},
      {"LogFile",    "InteractionsTest.log"}
   };

   PV::Interactions *interact = new PV::Interactions(args_dict, buffer.str());
   pvAssert(interact->begin() == PV::Interactions::SUCCESS);

   PVLayerLoc loc;
   pvAssert(interact->getLayerShape("Input", &loc) == PV::Interactions::SUCCESS);

   int numNeurons = loc.nbatch * loc.nx * loc.ny * loc.nf;
   int offset     = loc.kb0 * (loc.nxGlobal * loc.nyGlobal * loc.nf)
                  + loc.ky0 * (loc.nxGlobal * loc.nf)
                  + loc.kx0 * loc.nf;

   float multiplier = 1.0f;
   int   timestep   = 0;
   while (!interact->isFinished()) {
      std::vector<float> src_data, dst_data;
      // create test data for each timestep
      for (int i = 0; i < numNeurons; ++i) {
         src_data.push_back(offset + i + timestep);
      }
      timestep++;

      // put the test data in the input layer
      pvAssert(interact->setLayerState("Input", &src_data) == PV::Interactions::SUCCESS);

      // advance by one timestep
      pvAssert(interact->step(nullptr) == PV::Interactions::SUCCESS); 

      // Check the state of the input layer and make sure it did not change
      pvAssert(interact->getLayerState("Input", &dst_data) == PV::Interactions::SUCCESS);
      for (int i = 0; i < numNeurons; ++i) {
         pvAssert(src_data[i] == dst_data[i]);
      }

      // Get the output data and make sure we get the input times the weights
      pvAssert(interact->getLayerActivity("Output", &dst_data) == PV::Interactions::SUCCESS);
      pvAssert(src_data.size() == dst_data.size());
      for (int i = 0; i < numNeurons; ++i) {
         pvAssert(src_data[i] * multiplier == dst_data[i]);
      }
      multiplier *= 2.0f;
   
      // Double the value of nonzero weights every time
      std::vector<float> weights;
      pvAssert(interact->getConnectionWeights("InputToOutput", &weights) == PV::Interactions::SUCCESS);
      for (int i = 0; i < weights.size(); ++i) {
         if (weights[i] != 0.0f) {
            weights[i] *= 2.0f;
         }
      }
      pvAssert(interact->setConnectionWeights("InputToOutput", &weights) == PV::Interactions::SUCCESS);

   }


   pvAssert(interact->finish() == PV::Interactions::SUCCESS);

   delete interact;

   return EXIT_SUCCESS;
}
