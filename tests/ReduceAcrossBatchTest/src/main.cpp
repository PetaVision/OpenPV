/*
 * main.cpp for ReduceAcrossBatchTest.cpp
 *
 * This test depends on the ReduceAcrossBatchTest.params file, and the
 * working of the test is described there.
 */

#include <columns/PV_Init.hpp>
#include <columns/buildandrun.hpp>

int checkWeights(HyPerCol *hc, int argc, char *argv[]);

int main(int argc, char *argv[]) {
   int status = PV_SUCCESS;
   PV::PV_Init pv_initObj(&argc, &argv, false /*do not allow unrecognized arguments*/);
   if (pv_initObj.getIntegerArgument("NumRows") > 0) {
      if (pv_initObj.getWorldRank() == 0) {
         Fatal() << argv[0] << " should be run without the -rows option.\n";
      }
      status = PV_FAILURE;
   }
   if (pv_initObj.getIntegerArgument("NumColumns") > 0) {
      if (pv_initObj.getWorldRank() == 0) {
         Fatal() << argv[0] << " should be run without the -columns option.\n";
      }
      status = PV_FAILURE;
   }
   if (pv_initObj.getIntegerArgument("BatchWidth") > 0) {
      if (pv_initObj.getWorldRank() == 0) {
         Fatal() << argv[0] << " should be run without the -batchwidth option.\n";
      }
      status = PV_FAILURE;
   }
   if (status != PV_SUCCESS) {
      MPI_Barrier(pv_initObj.getCommunicator()->globalCommunicator());
      return EXIT_FAILURE;
   }
   // Set batch width to the number of processes, and rows and columns to 1.
   pv_initObj.setMPIConfiguration(1 /*rows*/, 1 /*columns*/, 0 /*compute the batchWidth*/);
   status = buildandrun(&pv_initObj, NULL, &checkWeights);
   return status == PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

int checkWeights(HyPerCol *hc, int argc, char *argv[]) {
   HyPerConn *conn = dynamic_cast<HyPerConn *>(hc->getObjectFromName("InputToOutput"));
   FatalIf(conn == nullptr, "No HyPerConn named \"InputToOutput\" in column.\n");
   HyPerLayer *correctValuesLayer = dynamic_cast<HyPerLayer *>(hc->getObjectFromName("SumInputs"));
   FatalIf(correctValuesLayer == nullptr, "No layer named \"SumInputs\" in column.\n");

   int const N = correctValuesLayer->getNumExtended();
   FatalIf(
         conn->getNumDataPatches() != N,
         "connection InputToOutput and layer SumInputs have different sizes.\n");
   float const *weights       = conn->get_wDataStart(0);
   float const *correctValues = correctValuesLayer->getLayerData(0);
   int status                 = PV_SUCCESS;
   for (int k = 0; k < N; k++) {
      if (weights[k] != correctValues[k]) {
         status = PV_FAILURE;
         ErrorLog() << "Weight index " << k << ": expected " << correctValues[k] << "; value was "
                    << weights[k] << "\n";
      }
   }
   return PV_SUCCESS;
}
