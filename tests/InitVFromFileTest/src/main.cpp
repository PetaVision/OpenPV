/*
 * main.cpp
 */

#include <columns/buildandrun.hpp>
#include <io/BroadcastLayerFile.hpp>
#include <io/LayerFile.hpp>
#include <io/SparseBroadcastLayerFile.hpp>
#include <io/SparseLayerFile.hpp>
#include <utils/BufferUtilsMPI.hpp>
#include <utils/BufferUtilsPvp.hpp>

int checkDenseNonbroadcast(PV_Init &pv_initObj);
int checkDenseBroadcast(PV_Init &pv_initObj);
int checkSparseNonbroadcast(PV_Init &pv_initObj);
int checkSparseBroadcast(PV_Init &pv_initObj);

PVLayerLoc createNonbroadcastLayerLoc(PV_Init &pv_initObj, std::string const &layerName);
PVLayerLoc createBroadcastLayerLoc(PV_Init &pv_initObj, std::string const &layerName);

int main(int argc, char *argv[]) {
   PV_Init pv_initObj(&argc, &argv, false /*allowUnrecognizedArgumentsFlag*/);
   int status = buildandrun(&pv_initObj);

   if (status == PV_SUCCESS) {
      status = checkDenseNonbroadcast(pv_initObj);
   }
   if (status == PV_SUCCESS) {
      status = checkDenseBroadcast(pv_initObj);
   }
   if (status == PV_SUCCESS) {
      status = checkSparseNonbroadcast(pv_initObj);
   }
   if (status == PV_SUCCESS) {
      status = checkSparseBroadcast(pv_initObj);
   }

   return status == PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

PVLayerLoc createBroadcastLayerLoc(PV_Init &pv_initObj, std::string const &layerName) {
   auto globalMPIBlock = pv_initObj.getCommunicator()->getGlobalMPIBlock();
   auto const &paramsFilename = pv_initObj.getStringArgument("ParamsFile");
   PVParams params(paramsFilename.c_str(), globalMPIBlock->getComm());
   auto columnParamsIO = params.makeParamsIO("column");
   int nbatchGlobal    = columnParamsIO->readValue<int>("nbatch");
   int globalMPIBatchDimension = globalMPIBlock->getGlobalBatchDimension();
   FatalIf(
         nbatchGlobal % globalMPIBatchDimension != 0,
         "The HyPerCol nbatch (%d) be a multiple of MPI Batch Dimension (%d).\n",
         nbatchGlobal, globalMPIBatchDimension);
   int nbatchLocal = nbatchGlobal / nbatchLocal;
   auto layerParamsIO = params.makeParamsIO(layerName);
   int nf           = layerParamsIO->readValue<int>("nf", false /*warnIfAbsentFlag*/);

   PVLayerLoc loc;
   loc.nbatchGlobal = nbatchGlobal;
   loc.nxGlobal     = globalMPIBlock->getNumColumns();
   loc.nyGlobal     = globalMPIBlock->getNumRows();
   loc.nbatch       = nbatchGlobal / globalMPIBatchDimension;
   loc.nx           = 1;
   loc.ny           = 1;
   loc.nf           = nf;
   loc.kb0          = loc.nbatch * globalMPIBlock->getBatchIndex();
   loc.kx0          = globalMPIBlock->getColumnIndex();
   loc.ky0          = globalMPIBlock->getRowIndex();
   loc.bcast        = 0;
   loc.halo.lt      = 0;
   loc.halo.rt      = 0;
   loc.halo.dn      = 0;
   loc.halo.up      = 0;
   return loc;
}

PVLayerLoc createNonbroadcastLayerLoc(PV_Init &pv_initObj, std::string const &layerName) {
   auto globalMPIBlock = pv_initObj.getCommunicator()->getGlobalMPIBlock();
   auto const &paramsFilename = pv_initObj.getStringArgument("ParamsFile");
   PVParams params(paramsFilename.c_str(), globalMPIBlock->getComm());
   auto columnParamsIO = params.makeParamsIO("column");
   auto layerParamsIO  = params.makeParamsIO(layerName);
   int nbatchGlobal = columnParamsIO->readValue<int>("nbatch");
   int globalMPIBatchDimension = globalMPIBlock->getGlobalBatchDimension();
   FatalIf(
         nbatchGlobal % globalMPIBatchDimension != 0,
         "The HyPerCol nbatch (%d) must be a multiple of MPI Batch Dimension (%d).\n",
         nbatchGlobal, globalMPIBatchDimension);
   int nbatchLocal     = nbatchGlobal / globalMPIBatchDimension;
   double nxFromColumn = columnParamsIO->readValue<double>("nx");
   double nxScale      = layerParamsIO->readValue<double>("nxScale", false /*warnIfAbsentFlag*/);
   int nxGlobal        = static_cast<int>(std::nearbyint(nxFromColumn * nxScale));
   double nyFromColumn = columnParamsIO->readValue<double>("ny");
   double nyScale      = layerParamsIO->readValue<double>("nyScale", false /*warnIfAbsentFlag*/);
   int nyGlobal        = static_cast<int>(std::nearbyint(nyFromColumn * nyScale));
   int nf              = layerParamsIO->readValue<int>("nf", false /*warnIfAbsentFlag*/);

   PVLayerLoc loc;
   loc.nbatchGlobal = nbatchGlobal;
   loc.nxGlobal     = nxGlobal;
   loc.nyGlobal     = nyGlobal;
   loc.nbatch       = nbatchGlobal / globalMPIBatchDimension;
   loc.nx           = nxGlobal / globalMPIBlock->getNumColumns();
   loc.ny           = nyGlobal / globalMPIBlock->getNumRows();
   loc.nf           = nf;
   loc.kb0          = loc.nbatch * globalMPIBlock->getBatchIndex();
   loc.kx0          = loc.nx * globalMPIBlock->getColumnIndex();
   loc.ky0          = loc.ny * globalMPIBlock->getRowIndex();
   loc.bcast        = 0;
   loc.halo.lt      = 0;
   loc.halo.rt      = 0;
   loc.halo.dn      = 0;
   loc.halo.up      = 0;
   return loc;
}

int checkDenseNonbroadcast(PV_Init &pv_initObj) {
   std::string layerName("DenseNonbroadcast");
   PVLayerLoc loc = createNonbroadcastLayerLoc(pv_initObj, layerName);
   int numNeurons = loc.nx * loc.ny * loc.nf;

   // load data from input file
   auto inputMPIBlock = pv_initObj.getCommunicator()->getGlobalMPIBlock();
   auto inputFileManager = std::make_shared<FileManager>(inputMPIBlock, "input");
   std::string inputFilename(layerName + ".pvp");
   LayerFile inputFile(
         inputFileManager,
         inputFilename,
         loc,
         false /*dataExtendedFlag*/,
         false /*fileExtendedFlag*/,
         true /*readOnlyFlag*/,
         false /*clobberFlag*/,
         false /*verifyWrites*/);
   std::vector<std::vector<float>> inputData(loc.nbatch);
   for (int b = 0; b < loc.nbatch; ++b) {
      inputData[b].resize(numNeurons);
      inputFile.setDataLocation(inputData[b].data(), b);
   }
   inputFile.read();
   
   // load data from output file
   auto outputMPIBlock = pv_initObj.getCommunicator()->getIOMPIBlock();
   auto outputFileManager = std::make_shared<FileManager>(outputMPIBlock, "output");
   std::string outputFilename(layerName + ".pvp");
   LayerFile outputFile(
         outputFileManager,
         outputFilename,
         loc,
         false /*dataExtendedFlag*/,
         false /*fileExtendedFlag*/,
         true /*readOnlyFlag*/,
         false /*clobberFlag*/,
         false /*verifyWrites*/);
   std::vector<std::vector<float>> outputData(loc.nbatch);
   for (int b = 0; b < loc.nbatch; ++b) {
      outputData[b].resize(numNeurons);
      outputFile.setDataLocation(outputData[b].data(), b);
   }
   outputFile.read();

   // compare inputData and outputData
   int status = PV_SUCCESS;
   for (int b = 0; b < loc.nbatch; ++b) {
      if (static_cast<int>(inputData[b].size()) != numNeurons) {
         ErrorLog().printf(
               "Rank %d: batch element %d input size %zu does not agree with params nf %d\n",
               pv_initObj.getCommunicator()->globalCommRank(),
               b,
               inputData[b].size(),
               numNeurons);
         status = PV_FAILURE;
         break;
      }
      if (outputData[b].size() != inputData[b].size()) {
         ErrorLog().printf(
               "Rank %d: batch element %d input and output sizes do not agree: %zu versus %zu\n",
               pv_initObj.getCommunicator()->globalCommRank(),
               b,
               inputData[b].size(),
               outputData[b].size());
         status = PV_FAILURE;
         break;
      }
      auto N = inputData[b].size();
      for (decltype(N) n = 0; n < N; ++n) {
         if (outputData[b][n] != inputData[b][n]) {
            ErrorLog().printf(
                  "Rank %d: batch element %d, neuron %zu values do not agree: %f versus %f\n",
                  pv_initObj.getCommunicator()->globalCommRank(),
                  b, n,
                  static_cast<double>(inputData[b][n]),
                  static_cast<double>(outputData[b][n]));
            status = PV_FAILURE;
         }
      }
   }
   return status;
}

int checkDenseBroadcast(PV_Init &pv_initObj) {
   std::string layerName("DenseBroadcast");
   PVLayerLoc loc = createBroadcastLayerLoc(pv_initObj, layerName);
   int numNeurons = loc.nx * loc.ny * loc.nf;

   // load data from input file
   auto inputMPIBlock = pv_initObj.getCommunicator()->getGlobalMPIBlock();
   auto inputFileManager = std::make_shared<FileManager>(inputMPIBlock, "input");
   std::string inputFilename(layerName + ".pvp");
   BroadcastLayerFile inputFile(
         inputFileManager,
         inputFilename,
         loc.nf,
         loc.nbatch,
         true /*readOnlyFlag*/,
         false /*clobberFlag*/,
         false /*verifyWrites*/);
   std::vector<std::vector<float>> inputData(loc.nbatch);
   for (int b = 0; b < loc.nbatch; ++b) {
      inputData[b].resize(loc.nf);
      inputFile.setDataLocation(inputData[b].data(), b);
   }
   inputFile.read();
   
   // load data from output file
   auto outputMPIBlock = pv_initObj.getCommunicator()->getIOMPIBlock();
   auto outputFileManager = std::make_shared<FileManager>(outputMPIBlock, "output");
   std::string outputFilename(layerName + ".pvp");
   BroadcastLayerFile outputFile(
         inputFileManager,
         inputFilename,
         loc.nf,
         loc.nbatch,
         true /*readOnlyFlag*/,
         false /*clobberFlag*/,
         false /*verifyWrites*/);
   std::vector<std::vector<float>> outputData(loc.nbatch);
   for (int b = 0; b < loc.nbatch; ++b) {
      outputData[b].resize(numNeurons);
      outputFile.setDataLocation(outputData[b].data(), b);
   }
   outputFile.read();

   // compare inputData and outputData
   int status = PV_SUCCESS;
   for (int b = 0; b < loc.nbatch; ++b) {
      if (static_cast<int>(inputData[b].size()) != loc.nf) {
         ErrorLog().printf(
               "Rank %d: batch element %d input size %zu does not agree with params nf %d\n",
               pv_initObj.getCommunicator()->globalCommRank(),
               b,
               inputData[b].size(),
               loc.nf);
         status = PV_FAILURE;
         break;
      }
      if (outputData[b].size() != inputData[b].size()) {
         ErrorLog().printf(
               "Rank %d: batch element %d input and output sizes do not agree: %zu versus %zu\n",
               pv_initObj.getCommunicator()->globalCommRank(),
               b,
               inputData[b].size(),
               outputData[b].size());
         status = PV_FAILURE;
         break;
      }
      auto N = inputData[b].size();
      for (decltype(N) n = 0; n < N; ++n) {
         if (outputData[b][n] != inputData[b][n]) {
            ErrorLog().printf(
                  "Rank %d: batch element %d, neuron %zu values do not agree: %f versus %f\n",
                  pv_initObj.getCommunicator()->globalCommRank(),
                  b, n,
                  static_cast<double>(inputData[b][n]),
                  static_cast<double>(outputData[b][n]));
            status = PV_FAILURE;
         }
      }
   }
   return status;
}

int checkSparseNonbroadcast(PV_Init &pv_initObj) {
   std::string layerName("SparseNonbroadcast");
   PVLayerLoc loc = createNonbroadcastLayerLoc(pv_initObj, layerName);
   int numNeurons = loc.nx * loc.ny * loc.nf;

   // load data from input file
   auto inputMPIBlock = pv_initObj.getCommunicator()->getGlobalMPIBlock();
   auto inputFileManager = std::make_shared<FileManager>(inputMPIBlock, "input");
   std::string inputFilename(layerName + ".pvp");
   SparseLayerFile inputFile(
         inputFileManager,
         inputFilename,
         loc,
         false /*dataExtendedFlag*/,
         false /*fileExtendedFlag*/,
         true /*readOnlyFlag*/,
         false /*clobberFlag*/,
         false /*verifyWrites*/);
   std::vector<SparseList<float>> inputData(loc.nbatch);
   for (int b = 0; b < loc.nbatch; ++b) {
      inputFile.setListLocation(&inputData[b], b);
   }
   inputFile.read();
   
   // load data from output file
   auto outputMPIBlock = pv_initObj.getCommunicator()->getIOMPIBlock();
   auto outputFileManager = std::make_shared<FileManager>(outputMPIBlock, "output");
   std::string outputFilename(layerName + ".pvp");
   SparseLayerFile outputFile(
         outputFileManager,
         outputFilename,
         loc,
         false /*dataExtendedFlag*/,
         false /*fileExtendedFlag*/,
         true /*readOnlyFlag*/,
         false /*clobberFlag*/,
         false /*verifyWrites*/);
   std::vector<SparseList<float>> outputData(loc.nbatch);
   for (int b = 0; b < loc.nbatch; ++b) {
      outputFile.setListLocation(&outputData[b], b);
   }
   outputFile.read();

   // compare inputData and outputData
   int status = PV_SUCCESS;
   for (int b = 0; b < loc.nbatch; ++b) {
      auto inputContents = inputData[b].getContents();
      auto outputContents = outputData[b].getContents();
      if (outputContents.size() != inputContents.size()) {
         ErrorLog().printf(
               "Rank %d: batch element %d input and output sizes do not agree: %zu versus %zu\n",
               pv_initObj.getCommunicator()->globalCommRank(),
               b,
               inputContents.size(),
               outputContents.size());
         status = PV_FAILURE;
         break;
      }
      auto N = inputContents.size();
      for (decltype(N) n = 0; n < N; ++n) {
         bool indexMatch = outputContents[n].index == inputContents[n].index;
         bool valueMatch = outputContents[n].value == inputContents[n].value;
         if (!indexMatch or !valueMatch) {
            ErrorLog().printf(
                  "Rank %d: batch element %d, sparse entry %zu does not agree: "
                  "(%d,%f) versus (%d,%f)\n",
                  pv_initObj.getCommunicator()->globalCommRank(),
                  b, n,
                  inputContents[n].index, static_cast<double>(inputContents[n].value),
                  outputContents[n].index, static_cast<double>(outputContents[n].value));
            status = PV_FAILURE;
         }
      }
   }
   return status;
}

int checkSparseBroadcast(PV_Init &pv_initObj) {
   std::string layerName("SparseBroadcast");
   PVLayerLoc loc = createBroadcastLayerLoc(pv_initObj, layerName);
   int numNeurons = loc.nx * loc.ny * loc.nf;

   // load data from input file
   auto inputMPIBlock = pv_initObj.getCommunicator()->getGlobalMPIBlock();
   auto inputFileManager = std::make_shared<FileManager>(inputMPIBlock, "input");
   std::string inputFilename(layerName + ".pvp");
   SparseBroadcastLayerFile inputFile(
         inputFileManager,
         inputFilename,
         loc.nf,
         loc.nbatch,
         true /*readOnlyFlag*/,
         false /*clobberFlag*/,
         false /*verifyWrites*/);
   std::vector<SparseList<float>> inputData(loc.nbatch);
   for (int b = 0; b < loc.nbatch; ++b) {
      inputFile.setListLocation(&inputData[b], b);
   }
   inputFile.read();
   
   // load data from output file
   auto outputMPIBlock = pv_initObj.getCommunicator()->getIOMPIBlock();
   auto outputFileManager = std::make_shared<FileManager>(outputMPIBlock, "output");
   std::string outputFilename(layerName + ".pvp");
   SparseBroadcastLayerFile outputFile(
         outputFileManager,
         outputFilename,
         loc.nf,
         loc.nbatch,
         true /*readOnlyFlag*/,
         false /*clobberFlag*/,
         false /*verifyWrites*/);
   std::vector<SparseList<float>> outputData(loc.nbatch);
   for (int b = 0; b < loc.nbatch; ++b) {
      outputFile.setListLocation(&outputData[b], b);
   }
   outputFile.read();

   // compare inputData and outputData
   int status = PV_SUCCESS;
   for (int b = 0; b < loc.nbatch; ++b) {
      auto inputContents = inputData[b].getContents();
      auto outputContents = outputData[b].getContents();
      if (outputContents.size() != inputContents.size()) {
         ErrorLog().printf(
               "Rank %d: batch element %d input and output sizes do not agree: %zu versus %zu\n",
               pv_initObj.getCommunicator()->globalCommRank(),
               b,
               inputContents.size(),
               outputContents.size());
         status = PV_FAILURE;
         break;
      }
      auto N = inputContents.size();
      for (decltype(N) n = 0; n < N; ++n) {
         bool indexMatch = outputContents[n].index == inputContents[n].index;
         bool valueMatch = outputContents[n].value == inputContents[n].value;
         if (!indexMatch or !valueMatch) {
            ErrorLog().printf(
                  "Rank %d: batch element %d, sparse entry %zu does not agree: "
                  "(%d,%f) versus (%d,%f)\n",
                  pv_initObj.getCommunicator()->globalCommRank(),
                  b, n,
                  inputContents[n].index, static_cast<double>(inputContents[n].value),
                  outputContents[n].index, static_cast<double>(outputContents[n].value));
            status = PV_FAILURE;
         }
      }
   }
   return status;
}
