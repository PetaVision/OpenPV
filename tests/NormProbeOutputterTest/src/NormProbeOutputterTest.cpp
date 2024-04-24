#include <checkpointing/Checkpointer.hpp>
#include <columns/Communicator.hpp>
#include <columns/HyPerCol.hpp>
#include <columns/PV_Init.hpp>
#include <include/pv_common.h>
#include <io/FileManager.hpp>
#include <io/FileStreamBuilder.hpp>
#include <io/PVParams.hpp>
#include <probes/NormProbeOutputter.hpp>
#include <probes/ProbeData.hpp>
#include <probes/ProbeDataBuffer.hpp>
#include <utils/PVLog.hpp>

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>

int const gLocalBatchWidth = 4;
int const gNumTimestamps   = 10;

using namespace PV;

struct OutputData {
   double mTimestamp;
   int mBatchIndex;
   int mNumNeurons;
   double mNorm;
};

int checkOutputFiles(HyPerCol *hypercol);
int compare(
      int timestamp,
      int batchIndex,
      int numNeurons,
      double correctNorm,
      OutputData const &observedData);
void deleteOldFiles(HyPerCol *hypercol);
int findNumNeurons(HyPerCol const *hypercol);
void generateData(ProbeDataBuffer<double> &data, Communicator *comm);
double generateNorm(int timeIndex, int batchWidth, int batchIndex);
PVParams generateProbeParams(std::string const &probeName, Communicator *comm);
NormProbeOutputter initNormProbeOutputter(PV_Init &pv_initObj);
void printGlobalStats(PV_Init &pv_init, int numNeurons);
int run(PV_Init &pv_init);

int main(int argc, char **argv) {
   auto pv_init = PV_Init(&argc, &argv, false);
   int status   = run(pv_init);

   return status == PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

int checkOutputFiles(HyPerCol *hypercol) {
   int status            = PV_SUCCESS;
   auto *comm            = hypercol->getPV_InitObj()->getCommunicator();
   auto const ioMPIBlock = comm->getIOMPIBlock();
   if (ioMPIBlock->getRank() != 0) {
      return PV_SUCCESS;
   } // Only root process does I/O.
   auto fileManager =
         std::make_shared<FileManager>(ioMPIBlock, std::string(hypercol->getOutputPath()));

   int const globalBatchWidth     = ioMPIBlock->getGlobalBatchDimension() * gLocalBatchWidth;
   int const globalBatchProcIndex = (ioMPIBlock->getStartBatch() + ioMPIBlock->getBatchIndex());

   int const batchStart = globalBatchProcIndex * gLocalBatchWidth;
   int const batchStop  = batchStart + ioMPIBlock->getBatchDimension() * gLocalBatchWidth;

   int numNeurons = findNumNeurons(hypercol);
   for (int b = batchStart; b < batchStop; ++b) {
      std::string outputPath("NormProbeOutputter_batchElement_");
      outputPath.append(std::to_string(b)).append(".txt");
      auto fileStream = FileStreamBuilder(fileManager, outputPath, true, true, false, false).get();
      std::string fullOutputPath = fileStream->getFileName();
      fileStream                 = nullptr; // close the file.
      // We'll reopen it using cstdio's fopen(), in order to read the values using fscanf().
      FILE *fp = fopen(fullOutputPath.c_str(), "r");
      FatalIf(fp == nullptr, "Unable to open %s for reading\n", fullOutputPath.c_str());
      for (int t = 0; t < gNumTimestamps; ++t) {
         double correctNorm = generateNorm(t, globalBatchWidth, b);
         OutputData observedValues;
         int numread = fscanf(
               fp,
               "%lf, %d, %d, %lf\n",
               &observedValues.mTimestamp,
               &observedValues.mBatchIndex,
               &observedValues.mNumNeurons,
               &observedValues.mNorm);
         if (numread != 4) {
            ErrorLog().printf(
                  "File \"%s\" does not have the expected format.\n", fullOutputPath.c_str());
            status = PV_FAILURE;
         }
         else if (compare(t, b, numNeurons, correctNorm, observedValues) != PV_SUCCESS) {
            status = PV_FAILURE;
         }
      }
      fclose(fp);
      fp = nullptr;
   }

   return status;
}

int compare(
      int timestamp,
      int batchIndex,
      int numNeurons,
      double correctNorm,
      OutputData const &observedData) {
   int status = PV_SUCCESS;

   double timeDbl = static_cast<double>(timestamp);
   if (observedData.mTimestamp != timeDbl) {
      ErrorLog().printf(
            "expected timestamp %d; observed timestamp %f, discrepancy %g\n",
            timestamp,
            observedData.mTimestamp,
            observedData.mTimestamp - timeDbl);
      status = PV_FAILURE;
   }

   if (observedData.mBatchIndex != batchIndex) {
      ErrorLog().printf(
            "timestamp %d, expected batch index %d; observed batch index %d\n",
            timestamp,
            batchIndex,
            observedData.mBatchIndex);
      status = PV_FAILURE;
   }

   if (observedData.mNumNeurons != numNeurons) {
      ErrorLog().printf(
            "timestamp %d, batch index %d, expected N = %d; observed N = %d\n",
            timestamp,
            batchIndex,
            numNeurons,
            observedData.mNumNeurons);
      status = PV_FAILURE;
   }

   if (observedData.mNorm != correctNorm) {
      ErrorLog().printf(
            "timestamp %d, batch index %d, expected value = %f; observed value = %f; discrepancy "
            "%g\n",
            timestamp,
            batchIndex,
            correctNorm,
            observedData.mNorm,
            observedData.mNorm - correctNorm);
      status = PV_FAILURE;
   }

   return status;
}

void deleteOldFiles(HyPerCol *hypercol) {
   // Delete old NormProbeOutputter output files, if they exist. That way we know, at the end of
   // the run, that if the files exist, they were produced by this run.
   // It would be better to do this without using the blunt hammer of calling system(rm -rf ...).
   auto const ioMPIBlock = hypercol->getPV_InitObj()->getCommunicator()->getIOMPIBlock();
   FileManager fileManager(ioMPIBlock, std::string(hypercol->getOutputPath()));
   if (ioMPIBlock->getRank() == 0) {
      std::string outputBlockDir = fileManager.makeBlockFilename(".");
      std::string rmrfcommand("rm -rf ");
      rmrfcommand.append(outputBlockDir).append("/");
      rmrfcommand.append("NormProbeOutputter_batchElement_*.txt");
      int rmrfstatus = system(rmrfcommand.c_str());
      FatalIf(
            rmrfstatus,
            "Global rank command \"system(%s)\") returned %d\n",
            rmrfcommand.c_str(),
            rmrfstatus);
   }
}

int findNumNeurons(HyPerCol const *hypercol) {
   int nx = hypercol->getNxGlobal();
   int ny = hypercol->getNxGlobal();
   return nx * ny;
}

void generateData(ProbeDataBuffer<double> &data, Communicator *comm) {
   data.clear();
   auto localMPIBlock = comm->getLocalMPIBlock();
   if (localMPIBlock->getRank() != 0) {
      return;
   }
   auto ioMPIBlock                = comm->getIOMPIBlock();
   int const globalBatchWidth     = ioMPIBlock->getGlobalBatchDimension() * gLocalBatchWidth;
   int const globalBatchProcIndex = (ioMPIBlock->getStartBatch() + ioMPIBlock->getBatchIndex());
   for (int t = 0; t < gNumTimestamps; ++t) {
      double timestamp = static_cast<double>(t);
      ProbeData<double> norms(timestamp, gLocalBatchWidth);
      for (int b = 0; b < gLocalBatchWidth; ++b) {
         int globalBatchIndex = b + gLocalBatchWidth * globalBatchProcIndex;
         norms.getValue(b)    = generateNorm(t, globalBatchWidth, globalBatchIndex);
      }
      data.store(norms);
   }
}

double generateNorm(int timeIndex, int batchWidth, int batchIndex) {
   return static_cast<double>(timeIndex * batchWidth + batchIndex);
}

PVParams generateProbeParams(std::string const &probeName, Communicator *comm) {
   std::string paramsString;
   paramsString.append("debugParsing = false;\n");
   paramsString.append("NormProbeOutputter \"").append(probeName).append("\" = {\n");
   paramsString.append("   textOutputFlag  = true;\n");
   paramsString.append("   probeOutputFile = \"NormProbeOutputter.txt\";\n");
   paramsString.append("   message         = \"NormProbeOutputter\";\n");
   paramsString.append("};\n");
   PVParams probeParams(paramsString.c_str(), paramsString.size(), 1UL, comm->globalCommunicator());
   return probeParams;
}

NormProbeOutputter initNormProbeOutputter(PV_Init &pv_initObj) {
   Communicator *comm = pv_initObj.getCommunicator();
   std::string probeName("probe");
   PVParams probeParams = generateProbeParams(probeName, comm);
   NormProbeOutputter normProbeOutputter(probeName.c_str(), &probeParams, comm);
   normProbeOutputter.ioParamsFillGroup(PARAMS_IO_READ);

   // create the output files.
   Checkpointer checkpointer("column", comm, pv_initObj.getArguments());
   normProbeOutputter.initOutputStreams(&checkpointer, gLocalBatchWidth);
   // At this point, the output files should exist and be empty.

   return normProbeOutputter;
}

void printGlobalStats(PV_Init &pv_init, int numNeurons) {
   ProbeDataBuffer<double> normsBuffer;
   generateData(normsBuffer, pv_init.getCommunicator());

   auto normProbeOutputter = initNormProbeOutputter(pv_init);
   normProbeOutputter.printGlobalNormsBuffer(normsBuffer, numNeurons);
}

int run(PV_Init &pv_init) {
   int status = PV_SUCCESS;

   HyPerCol *hypercol = new HyPerCol(&pv_init);
   // HyPerCol::processParams() creates the output directory and its block subdirectories if needed.
   hypercol->processParams("NormProbeOutputterTest.params");

   deleteOldFiles(hypercol);

   int numNeurons = findNumNeurons(hypercol);
   printGlobalStats(pv_init, numNeurons);

   status = checkOutputFiles(hypercol);

   delete hypercol;
   if (status == PV_SUCCESS) {
      InfoLog() << "Test passed.\n";
   }
   return status;
}
