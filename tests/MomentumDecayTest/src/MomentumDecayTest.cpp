/*
 * MomentumDecayTest.cpp
 *
 */

#include <columns/buildandrun.hpp>
#include <connections/MomentumConn.hpp>
#include <io/SharedWeightsFile.hpp>
#include <structures/WeightData.hpp>

int checkWeights(HyPerCol *hc, int argc, char *argv[]);
std::shared_ptr<WeightData const> getCorrectWeightData(HyPerCol *hc);
std::shared_ptr<WeightData const> getObservedWeightData(HyPerCol *hc);

int main(int argc, char *argv[]) {
   int status = buildandrun(argc, argv, nullptr, checkWeights);
   return status == PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

int checkWeights(HyPerCol *hc, int argc, char *argv[]) {
   auto correctWeightData = getCorrectWeightData(hc);
   auto observedWeightData = getObservedWeightData(hc);

   FatalIf(
         observedWeightData->getNumArbors() != correctWeightData->getNumArbors(),
         "NumArbors mismatch\n");
   int numArbors = observedWeightData->getNumArbors();
   FatalIf(
         observedWeightData->getPatchSizeX() != correctWeightData->getPatchSizeX(),
         "PatchSizeX mismatch\n");
   FatalIf(
         observedWeightData->getPatchSizeY() != correctWeightData->getPatchSizeY(),
         "PatchSizeY mismatch\n");
   FatalIf(
         observedWeightData->getPatchSizeF() != correctWeightData->getPatchSizeF(),
         "PatchSizeF mismatch\n");
   FatalIf(
         observedWeightData->getNumDataPatchesX() != correctWeightData->getNumDataPatchesX(),
         "NumDataPatchesX mismatch\n");
   FatalIf(
         observedWeightData->getNumDataPatchesY() != correctWeightData->getNumDataPatchesY(),
         "NumDataPatchesY mismatch\n");
   FatalIf(
         observedWeightData->getNumDataPatchesF() != correctWeightData->getNumDataPatchesF(),
         "NumDataPatchesF mismatch\n");
   FatalIf(
         observedWeightData->getNumValuesPerArbor() != correctWeightData->getNumValuesPerArbor(),
         "NumValuesPerArbor mismatch\n");
   long int N = observedWeightData->getNumValuesPerArbor();

   int status = PV_SUCCESS;
   for (int a = 0; a < observedWeightData->getNumArbors(); ++a) {
      float const *correctData = correctWeightData->getData(a);
      float const *computedData = observedWeightData->getData(a);
      for (long int n = 0L; n < N; ++n) {
         float discrep = computedData[n] - correctData[n];
         if (std::abs(discrep) > 5e-7) {
            ErrorLog().printf(
                  "Arbor %d, weight %ld differs %f versus %f (discrepancy %g)\n",
                  a, n, (double)computedData[n], (double)correctData[n], (double)discrep);
            status = PV_FAILURE;
         }
      }
   }

   return status;
}

std::shared_ptr<WeightData const> getCorrectWeightData(HyPerCol *hc) {
   std::string const &paramsFilename = hc->getPV_InitObj()->getStringArgument("ParamsFile");
   std::string inputDir = dirName(paramsFilename);
   auto *communicator = hc->getCommunicator();
   auto ioMPIBlock = communicator->getIOMPIBlock();
   auto fileManager = std::make_shared<FileManager>(ioMPIBlock, inputDir);
   std::string paramsName = stripExtension(paramsFilename);
   std::string correctWeightsFilename = paramsName + "_correct.pvp";
   auto correctWeightData = std::make_shared<WeightData>(
         1 /*numAxonalArbors*/,
         1 /*patchSizeX*/, 1 /*patchSizeY*/, 3 /*patchSizeF*/,
         1 /*numDataPatchesX*/, 1 /*numDataPatchesY*/, 1 /*numDataPatchesF*/);
   SharedWeightsFile correctWeightsFile(
         fileManager,
         correctWeightsFilename,
         correctWeightData,
         false /*compressedFlag*/,
         true /*readOnlyFlag*/,
         false /*clobberFlag*/,
         false /*verifyWrites*/);
   correctWeightsFile.read();
   return correctWeightData;
}

std::shared_ptr<WeightData const> getObservedWeightData(HyPerCol *hc) {
   auto *connection = hc->getTable()->findObject<MomentumConn>("PreToPost");
   FatalIf(connection == nullptr, "Unable to find MomentumConn \"PreToPost\"\n");
   auto *weightsPair = connection->getComponentByType<WeightsPair>();
   FatalIf(weightsPair == nullptr, "Unable to find a WeightsPair in MomentumConn \"PreToPost\"\n");
   auto *weights = weightsPair->getPreWeights();
   return weights->getData();
}
