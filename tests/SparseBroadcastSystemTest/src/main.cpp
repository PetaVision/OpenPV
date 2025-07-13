/*
 * main.cpp
 *
 */

#include <columns/buildandrun.hpp>
#include <columns/PV_Init.hpp>
#include <io/SparseBroadcastLayerFile.hpp>

#include <cstdlib>

int checkValues(PV_Init &pv_initObj);

bool operator!=(SparseList<float>::Entry const &lhs, SparseList<float>::Entry const &rhs) {
   bool indicesMatch = (lhs.index != rhs.index);
   bool valuesMatch  = (lhs.value != rhs.value);
   return (indicesMatch or valuesMatch);
}

int main(int argc, char *argv[]) {
   PV_Init pv_initObj(&argc, &argv, false /*allowUnrecognizedArgumentsFlag*/);
   int status = buildandrun(&pv_initObj);

   if (status == PV_SUCCESS) {
      status = checkValues(pv_initObj);
   }

   return status == PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

int checkValues(PV_Init &pv_initObj) {
   auto globalMPIBlock = pv_initObj.getCommunicator()->getGlobalMPIBlock();
   std::string const &paramsFilename = pv_initObj.getStringArgument("ParamsFile");
   PVParams params(paramsFilename.c_str(), globalMPIBlock->getComm());
   auto columnParamsIO = params.makeParamsIO("column");
   int nbatchGlobal = columnParamsIO->readValue<int>("nbatch");
   int globalMPIBatchDimension = globalMPIBlock->getBatchDimension();
   FatalIf(
         nbatchGlobal % globalMPIBatchDimension != 0,
         "The HyPerCol nbatch (%d) be a multiple of MPI Batch Dimension (%d).\n",
         nbatchGlobal, globalMPIBatchDimension);
   int nbatchLocal = nbatchGlobal / globalMPIBatchDimension;

   std::string layerName("SparseBroadcast2");
   auto layerParamsIO = params.makeParamsIO(layerName);
   int nf = layerParamsIO->readValue<int>("nf", false /*warnIfAbsentFlag*/);

   // load data from output file
   auto outputMPIBlock = pv_initObj.getCommunicator()->getIOMPIBlock();
   auto outputFileManager = std::make_shared<FileManager>(outputMPIBlock, "output");
   std::string outputFilename(layerName + ".pvp");
   SparseBroadcastLayerFile outputFile(
         outputFileManager,
         outputFilename,
         nf,
         nbatchLocal,
         true /*readOnlyFlag*/,
         false /*clobberFlag*/,
         false /*verifyWrites*/);
   std::vector<SparseList<float>> outputData(nbatchLocal);
   for (int b = 0; b < nbatchLocal; ++b) {
      outputFile.setListLocation(&outputData[b], b);
   }
   outputFile.read();

   std::vector<std::vector<SparseList<float>::Entry>> correctValues(nbatchGlobal);
   correctValues[0] = { {0, 2.0f}, {1, 5.0f}, {3, 6.0f}, {5, -8.0f}, {6, -1.0f} };
   correctValues[1] = { {4, -8.0f}, {7, 14.0f}, {9, -6.0f}, {12, -5.0f}, {14, 4.5f} };
   correctValues[2] = { {2, 15.0f}, {8, 5.5f}, {10, 26.0f}, {11, -6.0f}, {13, -28.0f} };
   correctValues[3] = { {4, -18.0f}, {6, -8.0f}, {7, 34.0f}, {13, 38.0f}, {15, -20.0f} };


   int status = PV_SUCCESS;
   int batchOffset = outputMPIBlock->getBatchIndex() * nbatchLocal;
   for (int b = 0; b < nbatchLocal; ++b) {
      int globalBatchIndex = b + batchOffset;
      FatalIf(
            globalBatchIndex < 0 or globalBatchIndex >= nbatchGlobal,
            "globalBatchIndex %d out of bounds (nbatchGlobal = %d)\n",
            globalBatchIndex, nbatchGlobal);
      auto correctBatchElement = correctValues[globalBatchIndex];
      auto observedBatchElement = outputData[b].getContents();
      auto N = correctBatchElement.size();
      if (observedBatchElement.size() != N) {
         ErrorLog().printf(
               "Output file \"%s\", batch element %d (global batch element %d) "
               "has %zu nonzero entries instead of the correct number %zu\n",
               outputFilename.c_str(), b, globalBatchIndex, observedBatchElement.size(), N);
         status = PV_FAILURE;
         continue;
      }
      for (decltype(N) n = 0; n < N; ++n) {
         auto const &correctEntry  = observedBatchElement[n];
         auto const &observedEntry = observedBatchElement[n];
         if (observedEntry != correctEntry) {
            ErrorLog().printf(
                  "Output file \"%s\", batch element %d (global batch element %d), entry %zu: "
                  "observed sparse entry (%d, %f) does not match correct entry (%d, %f)\n",
                  outputFilename.c_str(), b, globalBatchIndex, static_cast<std::size_t>(n),
                  observedEntry.index, static_cast<double>(observedEntry.value),
                  correctEntry.index, static_cast<double>(correctEntry.value));
         }
      }
   }
   return status;
}
