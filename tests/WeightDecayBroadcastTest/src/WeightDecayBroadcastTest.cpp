/*
 * WeightDecayL2BroadcastTest.cpp
 *
 */

#include <columns/buildandrun.hpp>
#include <connections/HyPerConn.hpp>
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
   float maxDiscrep = 0.0f;
   float const tolerance = 5.0e-7f;
   for (int a = 0; a < observedWeightData->getNumArbors(); ++a) {
      float const *correctData = correctWeightData->getData(a);
      float const *computedData = observedWeightData->getData(a);
      for (long int n = 0L; n < N; ++n) {
         float discrep = computedData[n] - correctData[n];
         maxDiscrep    = std::max(maxDiscrep, std::fabs(discrep));
         if (std::fabs(discrep) > tolerance) {
            ErrorLog().printf(
                  "Arbor %d, weight %ld differs %f versus %f (discrepancy %g)\n",
                  a, n, (double)computedData[n], (double)correctData[n], (double)discrep);
            status = PV_FAILURE;
         }
      }
   }
   auto *communicator = hc->getCommunicator();
   auto globalMPIBlock = communicator->getGlobalMPIBlock();
   MPI_Allreduce(MPI_IN_PLACE, &maxDiscrep, 1, MPI_FLOAT, MPI_MAX, globalMPIBlock->getComm());
   if (status == PV_SUCCESS) {
      InfoLog().printf("Test passed. Maximum weight discrepancy was %f\n", (double)maxDiscrep);
   }
   else {
      ErrorLog().printf(
            "Test failed. Maximum weight discrepancy was %f, greater than threshold of %f\n",
            (double)maxDiscrep, (double)tolerance);
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
   auto correctWeightDataAll = std::make_shared<WeightData>(
         "correctWeightDataAll",
         1 /*numAxonalArbors*/,
         4 /*patchSizeX*/, 4 /*patchSizeY*/, 3 /*patchSizeF*/,
         1 /*numDataPatchesX*/, 1 /*numDataPatchesY*/, 16 /*numDataPatchesF*/);
   SharedWeightsFile correctWeightsFile(
         fileManager,
         correctWeightsFilename,
         correctWeightDataAll,
         false /*compressedFlag*/,
         true /*readOnlyFlag*/,
         false /*clobberFlag*/,
         false /*verifyWrites*/);
   correctWeightsFile.read();

   // Now each process has all the weights, but the observed weights will contain only
   // the section corresponding to that process.
   int numRows    = ioMPIBlock->getNumRows();
   int numColumns = ioMPIBlock->getNumColumns();
   int nxpGlobal  = correctWeightDataAll->getPatchSizeX();
   int nxpLocal   = nxpGlobal / numColumns;
   int nypGlobal  = correctWeightDataAll->getPatchSizeY();
   int nypLocal   = nypGlobal / numRows;
   FatalIf(
         nxpLocal * numColumns != correctWeightDataAll->getPatchSizeX(),
         "number of MPI columns (%d) does not divide nxp = %d in file for correct weights \"%s\"\n",
         numColumns, correctWeightsFilename.c_str());
   FatalIf(
         nypLocal * numRows != correctWeightDataAll->getPatchSizeY(),
         "number of MPI columns (%d) does not divide nxp = %d in file for correct weights \"%s\"\n",
         numRows, correctWeightsFilename.c_str());
   auto correctWeightDataLocal = std::make_shared<WeightData>(
         "correctWeightDataLocal",
         1 /*numAxonalArbors*/,
         nxpLocal, nypLocal, 3 /*patchSizeF*/,
         1 /*numDataPatchesX*/, 1 /*numDataPatchesY*/, 16 /*numDataPatchesF*/);
   int numPatches   = (int)correctWeightDataAll->getNumDataPatchesOverall();
   int nfp          = correctWeightDataAll->getPatchSizeF();
   int columnStart  = nxpLocal * ioMPIBlock->getColumnIndex();
   int rowStart     = nypLocal * ioMPIBlock->getRowIndex();
   for (int kp = 0; kp < numPatches; ++kp) {
      float *globalPointer = correctWeightDataAll->getDataFromDataIndex(0 /*arbor*/, kp);
      float *localPointer  = correctWeightDataLocal->getDataFromDataIndex(0 /*arbor*/, kp);
      for (int ky = 0; ky < nypLocal; ++ky) {
         for (int kx = 0; kx < nxpLocal; ++kx) {
            for (int kf = 0; kf < nfp; ++kf) {
               long globalIndex =
                     kIndex(kx + columnStart, ky + rowStart, kf, nxpGlobal, nypGlobal, nfp);
               long localIndex = kIndex(kx, ky, kf, nxpLocal, nypLocal, nfp);
               localPointer[localIndex] = globalPointer[globalIndex];
            }
         }
      }
   }
   return correctWeightDataLocal;
}

std::shared_ptr<WeightData const> getObservedWeightData(HyPerCol *hc) {
   auto *connection = hc->getTable()->findObject<HyPerConn>("PreToPost");
   FatalIf(connection == nullptr, "Unable to find HyPerConn \"PreToPost\"\n");
   auto *weightsPair = connection->getComponentByType<WeightsPair>();
   FatalIf(weightsPair == nullptr, "Unable to find a WeightsPair in HyPerConn \"PreToPost\"\n");
   auto *weights = weightsPair->getPreWeights();
   return weights->getData();
}
