/*
 * pv.cpp
 *
 */

#include "FailBeforeExpectedStartTimeLayer.hpp"
#include "columns/HyPerCol.hpp"
#include "columns/PV_Init.hpp"
#include "copyOutput.hpp"
#include "utils/PVLog.hpp"
#include <vector>

using namespace PV;

int main(int argc, char *argv[]) {
   int status = PV_SUCCESS;
   PV_Init pv_init{&argc, &argv, false /*do not allow unrecognized arguments*/};
   status = pv_init.registerKeyword(
         "FailBeforeExpectedStartTimeLayer", Factory::create<FailBeforeExpectedStartTimeLayer>);
   FatalIf(status != PV_SUCCESS, "Unable to add FailBeforeExpectedStartTimeLayer\n");
   if (!pv_init.getStringArgument("CheckpointReadDirectory").empty()) {
      if (pv_init.getCommunicator()->commRank() == 0) {
         ErrorLog() << argv[0] << " cannot be run with the -c argument.\n";
      }
      status = PV_FAILURE;
   }
   if (pv_init.getBooleanArgument("Restart")) {
      if (pv_init.getCommunicator()->commRank() == 0) {
         ErrorLog() << argv[0] << " cannot be run with the -r flag.\n";
      }
      status = PV_FAILURE;
   }
   if (status != PV_SUCCESS) {
      MPI_Barrier(pv_init.getCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }

   HyPerCol *hc;
   FailBeforeExpectedStartTimeLayer *outputLayer;

   hc = new HyPerCol(&pv_init);
   FatalIf(hc == nullptr, "failed to create HyPerCol.\n");
   outputLayer = dynamic_cast<FailBeforeExpectedStartTimeLayer *>(hc->getObjectFromName("Output"));
   FatalIf(
         outputLayer == nullptr,
         "Params file does not have a FailBeforeExpectedStartTimeLayer called \"Output\".\n");
   outputLayer->setExpectedStartTime(0.0);
   status = hc->run(10.0, 1.0);
   FatalIf(status != PV_SUCCESS, "HyPerCol::run failed with arguments (0.0, 10.0, 1.0).\n");
   std::vector<float> withoutRestart = copyOutput(outputLayer);
   delete hc;

   hc = new HyPerCol(&pv_init);
   FatalIf(hc == nullptr, "failed to create HyPerCol.\n");
   outputLayer = dynamic_cast<FailBeforeExpectedStartTimeLayer *>(hc->getObjectFromName("Output"));
   FatalIf(
         outputLayer == nullptr,
         "Params file does not have a FailBeforeExpectedStartTimeLayer called \"Output\".\n");
   outputLayer->setExpectedStartTime(0.0);
   status = hc->run(5.0, 1.0);
   FatalIf(status != PV_SUCCESS, "HyPerCol::run failed with arguments (0.0, 5.0, 1.0).\n");
   delete hc;

   pv_init.setBooleanArgument("Restart", true);
   hc = new HyPerCol(&pv_init);
   FatalIf(hc == nullptr, "failed to create HyPerCol.\n");
   outputLayer = dynamic_cast<FailBeforeExpectedStartTimeLayer *>(hc->getObjectFromName("Output"));
   FatalIf(
         outputLayer == nullptr,
         "Params file does not have a FailBeforeExpectedStartTimeLayer called \"Output\".\n");
   outputLayer->setExpectedStartTime(6.0);
   status = hc->run();
   FatalIf(status != PV_SUCCESS, "HyPerCol::run failed with restart flag set to true.\n");
   std::vector<float> afterRestart = copyOutput(outputLayer);
   delete hc;

   size_t const numNeurons = afterRestart.size();
   FatalIf(
         withoutRestart.size() != numNeurons,
         "Number of neurons differ between runs (%zu versus %zu)\n",
         withoutRestart.size(),
         numNeurons);

   for (size_t n = 0; n < numNeurons; n++) {
      if (afterRestart.at(n) != withoutRestart.at(n)) {
         ErrorLog() << "Index " << n << ": values differ (without restart: " << withoutRestart.at(n)
                    << ", after restart: " << afterRestart.at(n) << "\n";
         status = PV_FAILURE;
      }
   }

   return status == PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}
