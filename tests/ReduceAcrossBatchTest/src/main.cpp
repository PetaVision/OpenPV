/*
 * main.cpp for ReduceAcrossBatchTest.cpp
 *
 * This test depends on the ReduceAcrossBatchTest.params file, and the
 * working of the test is described there.
 */

#include <columns/ComponentBasedObject.hpp>
#include <columns/PV_Init.hpp>
#include <columns/buildandrun.hpp>
#include <components/WeightsPair.hpp>

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
   int numProcesses = pv_initObj.getWorldSize();
   pv_initObj.setMPIConfiguration(1 /*rows*/, 1 /*columns*/, numProcesses /*batchWidth*/);
   status = buildandrun(&pv_initObj, NULL, &checkWeights);
   return status == PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

int checkWeights(HyPerCol *hc, int argc, char *argv[]) {
   auto *conn = dynamic_cast<ComponentBasedObject *>(hc->getObjectFromName("InputToOutput"));
   FatalIf(conn == nullptr, "No connection named \"InputToOutput\" in column.\n");
   HyPerLayer *correctValuesLayer = dynamic_cast<HyPerLayer *>(hc->getObjectFromName("SumInputs"));
   FatalIf(correctValuesLayer == nullptr, "No layer named \"SumInputs\" in column.\n");
   auto *correctValuesPublisher = correctValuesLayer->getComponentByType<BasePublisherComponent>();
   FatalIf(
         correctValuesPublisher == nullptr,
         "%s does not have a BasePublisherComponent.\n",
         correctValuesLayer->getDescription_c());

   int const N       = correctValuesLayer->getNumExtended();
   auto *weightsPair = conn->getComponentByType<WeightsPair>();
   FatalIf(
         weightsPair == nullptr,
         "%s does not have a WeightsPair component.\n",
         conn->getDescription_c());
   auto *preWeights = weightsPair->getPreWeights();
   FatalIf(
         preWeights->getNumDataPatches() != N,
         "connection InputToOutput and layer SumInputs have different sizes.\n");
   float const *weights       = preWeights->getData(0);
   float const *correctValues = correctValuesPublisher->getLayerData(0);
   int status                 = PV_SUCCESS;
   for (int k = 0; k < N; k++) {
      if (weights[k] != correctValues[k]) {
         status = PV_FAILURE;
         ErrorLog() << "Weight index " << k << ": expected " << correctValues[k] << "; value was "
                    << weights[k] << "\n";
      }
   }
   return status;
}
