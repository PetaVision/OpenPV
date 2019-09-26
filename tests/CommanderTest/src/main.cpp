
#include <bindings/Commander.hpp>

void err(std::string const msg) {
   pvAssert(false);
}

int main(int argc, char *argv[]) {

   std::ifstream params("input/CommanderTest.params");
   std::stringstream buffer;
   buffer << params.rdbuf();

   std::map<std::string, std::string> args_dict = {
      {"NumThreads", "1"},
      {"LogFile",    "CommanderTest.log"}
   };

   PV::Commander *cmd = new PV::Commander(args_dict, buffer.str(), &err);
   if (cmd->isRoot() == false) {
      cmd->waitForCommands();
   }
   else {
      cmd->begin();

      int nb, ny, nx, nf;
      cmd->getLayerShape("Input", &nb, &ny, &nx, &nf); 

      int   numNeurons = nb * ny * nx * nf;
      float multiplier = 1.0f;
      int   timestep   = 0;

      while (!cmd->isFinished()) {
         std::vector<float> src_data, dst_data;
         // create test data for each timestep
         for (int i = 0; i < numNeurons; ++i) {
            src_data.push_back(i + timestep);
         }
         timestep++;

         // put the test data in the input layer
         cmd->setLayerState("Input", &src_data);

         // advance by one timestep
         cmd->advance(1);

         // Test getProbeValues
         if (timestep > 1) {
            std::vector<double> values;
            cmd->getProbeValues("Probe", &values);
            pvAssert(values.size() == nb);
            for (int b = 0; b < nb; ++b) {
               double sum = 0.0;
               int offset = b * numNeurons / nb;
               for (int i = 0; i < numNeurons / nb; ++i) {
                  sum += pow((double)src_data[i + offset], 2.0);
               }
               pvAssert(sum == values[b]);
            }
         }

         // Get the output data and make sure we get the input times the weights
         cmd->getLayerActivity("Output", &dst_data, nullptr, nullptr, nullptr, nullptr);
         pvAssert(src_data.size() == dst_data.size());
         for (int i = 0; i < numNeurons; ++i) {
            pvAssert(src_data[i] * multiplier == dst_data[i]);
         }
         multiplier *= 2.0f;

         // Check the state of the input layer and make sure it did not change
         cmd->getLayerState("Input", &dst_data, nullptr, nullptr, nullptr, nullptr);
         for (int i = 0; i < numNeurons; ++i) {
            pvAssert(src_data[i] == dst_data[i]);
         }
      
         // Double the value of nonzero weights every time
         float *weights = nullptr;
         int nwp = 0, nyp = 0, nxp = 0, nfp = 0; 
         cmd->getConnectionWeights("InputToOutput", &weights, &nwp, &nyp, &nxp, &nfp);
         for (int i = 0; i < nwp*nyp*nxp*nfp; ++i) {
            if (weights[i] != 0.0f) {
               weights[i] *= 2.0f;
            }
         }
         std::vector<float> w(weights, weights+nwp*nyp*nxp*nfp);
         cmd->setConnectionWeights("InputToOutput", &w);
      }
      
      cmd->finish();
   }

   delete cmd;

   return EXIT_SUCCESS;
}
