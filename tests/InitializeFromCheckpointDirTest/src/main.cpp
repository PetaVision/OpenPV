/*
 * pv.cpp
 *
 */

#include "columns/HyPerCol.hpp"
#include "columns/PV_Init.hpp"
#include "layers/HyPerLayer.hpp"

int main(int argc, char *argv[]) {
   int status = PV_SUCCESS;
   PV::PV_Init pv_initObj{&argc, &argv, false /*do not allow unrecognized arguments*/};
   if (pv_initObj.getParams() != nullptr) {
      status = PV_FAILURE;
      if (pv_initObj.getCommunicator()->commRank() == 0) {
         ErrorLog() << argv[0]
                    << " must be run without params files. The necessary files are hard-coded.\n";
      }
   }
   if (pv_initObj.getIntegerArgument("NumRows") != 0) {
      status = PV_FAILURE;
      if (pv_initObj.getCommunicator()->commRank() == 0) {
         ErrorLog() << argv[0]
                    << " must be run without setting rows. The necessary value is computed.\n";
      }
   }
   if (pv_initObj.getIntegerArgument("NumColumns") != 0) {
      status = PV_FAILURE;
      if (pv_initObj.getCommunicator()->commRank() == 0) {
         ErrorLog() << argv[0]
                    << " must be run without setting columns. The necessary value is computed.\n";
      }
   }
   if (pv_initObj.getIntegerArgument("BatchWidth") != 0) {
      status = PV_FAILURE;
      if (pv_initObj.getCommunicator()->commRank() == 0) {
         ErrorLog()
               << argv[0]
               << " must be run without setting batchwidth. The necessary value is computed.\n";
      }
   }

   MPI_Barrier(pv_initObj.getCommunicator()->globalCommunicator());
   if (status != PV_SUCCESS) {
      exit(EXIT_FAILURE);
   }

   if (pv_initObj.getCommunicator()->globalCommRank() == 0) {
      char const *rmcommand = "rm -rf checkpoints output";
      status                = system(rmcommand);
      if (status != 0) {
         Fatal().printf(
               "deleting old checkpoints and output directories failed: \"%s\" returned %d\n",
               rmcommand,
               status);
      }
   }

   pv_initObj.setMPIConfiguration(-1, -1, pv_initObj.getWorldSize());

   pv_initObj.setParams("input/BaseRun.params");
   PV::HyPerCol *hc1 = new PV::HyPerCol(&pv_initObj);
   status            = hc1->run();
   FatalIf(status != PV_SUCCESS, "buildandrun with BaseRun.params failed.\n");
   double const stopTime1 = hc1->getStopTime();

   pv_initObj.setParams("input/InitializeFromCheckpointDirTest.params");
   PV::HyPerCol *hc2 = new PV::HyPerCol(&pv_initObj);
   status            = hc2->run();
   FatalIf(status != PV_SUCCESS, "run with InitializeFromCheckpointDirTest.params failed.\n");
   double const stopTime2 = hc2->getStopTime();

   PV::HyPerLayer *outputLayer = dynamic_cast<PV::HyPerLayer *>(hc2->getObjectFromName("Output"));
   FatalIf(outputLayer == nullptr, "No layer named \"Output\".");

   double const totalTime = stopTime1 + stopTime2;
   PVLayerLoc const *loc  = outputLayer->getLayerLoc();
   for (int b = 0; b < loc->nbatch; b++) {
      int const globalBatchIndex = loc->kb0 + b;
      int const N                = outputLayer->getNumExtended();
      float const *A             = &outputLayer->getLayerData()[b * N];
      float const correct        = totalTime * (double)(globalBatchIndex + 1);
      for (int k = 0; k < N; k++) {
         if (A[k] != correct) {
            status = PV_FAILURE;
            ErrorLog() << "Batch index " << globalBatchIndex << ", neuron " << k << ": expected "
                       << correct << "; received " << A[k] << "\n";
         }
      }
   }

   return status == PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}
