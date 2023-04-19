#include <checkpointing/Checkpointer.hpp>
#include <columns/Communicator.hpp>
#include <columns/HyPerCol.hpp>
#include <columns/PV_Init.hpp>
#include <include/pv_common.h>
#include <io/FileManager.hpp>
#include <io/FileStreamBuilder.hpp>
#include <io/PVParams.hpp>
#include <probes/ColumnEnergyOutputter.hpp>
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
   double mEnergy;
};

int checkOutputFiles(HyPerCol *hypercol);
int compare(int timestamp, int batchIndex, double correctEnergy, OutputData const &observedData);
void deleteOldFiles(HyPerCol *hypercol);
void generateData(ProbeDataBuffer<double> &data, Communicator *comm);
double generateNorm(int timeIndex, int batchWidth, int batchIndex);
PVParams generateProbeParams(std::string const &probeName, Communicator *comm);
ColumnEnergyOutputter initColumnEnergyOutputter(PV_Init &pv_initObj);
void printGlobalStats(PV_Init &pv_init);
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

   for (int b = batchStart; b < batchStop; ++b) {
      std::string outputPath("ColumnEnergyOutputter_batchElement_");
      outputPath.append(std::to_string(b)).append(".txt");
      auto fileStream = FileStreamBuilder(fileManager, outputPath, true, true, false, false).get();
      std::string fullOutputPath = fileStream->getFileName();
      fileStream                 = nullptr; // close the file.
      // We'll reopen it using cstdio's fopen(), in order to read the values using fscanf().
      std::FILE *fp = fopen(fullOutputPath.c_str(), "r");
      FatalIf(fp == nullptr, "Unable to open %s for reading\n", fullOutputPath.c_str());
      char const *expectedHeader = "time,index,energy\n";
      char headerLine[20];
      char *fgetsResult = std::fgets(headerLine, 20, fp);
      FatalIf(fgetsResult == nullptr, "Unable to read headerline of %s\n", fullOutputPath.c_str());
      if (strcmp(expectedHeader, headerLine)) {
         ErrorLog() << "Bad header line\n";
         status = PV_FAILURE;
      }
      InfoLog() << "headerLine --> [" << headerLine << "] <-- headerLine\n";
      for (int t = 0; t < gNumTimestamps; ++t) {
         double correctEnergy = generateNorm(t, globalBatchWidth, b);
         OutputData observedValues;
         int numread = fscanf(
               fp,
               "%lf, %d, %lf\n",
               &observedValues.mTimestamp,
               &observedValues.mBatchIndex,
               &observedValues.mEnergy);
         if (compare(t, b, correctEnergy, observedValues) != PV_SUCCESS) {
            status = PV_FAILURE;
         }
      }
      fclose(fp);
      fp = nullptr;
   }

   return status;
}

int compare(int timestamp, int batchIndex, double correctEnergy, OutputData const &observedData) {
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

   if (observedData.mEnergy != correctEnergy) {
      ErrorLog().printf(
            "timestamp %d, batch index %d, expected value = %f; observed value = %f; discrepancy "
            "%g\n",
            timestamp,
            batchIndex,
            correctEnergy,
            observedData.mEnergy,
            observedData.mEnergy - correctEnergy);
      status = PV_FAILURE;
   }

   return status;
}

void deleteOldFiles(HyPerCol *hypercol) {
   // Delete old ColumnEnergyOutputter output files, if they exist. That way we know,
   // at the end of the run, that if the files exist, they were produced by this run.
   // It would be better to do this without using the blunt hammer of calling system(rm -rf ...).
   auto const ioMPIBlock = hypercol->getPV_InitObj()->getCommunicator()->getIOMPIBlock();
   FileManager fileManager(ioMPIBlock, std::string(hypercol->getOutputPath()));
   if (ioMPIBlock->getRank() == 0) {
      std::string outputBlockDir = fileManager.makeBlockFilename(".");
      std::string rmrfcommand("rm -rf ");
      rmrfcommand.append(outputBlockDir).append("/");
      rmrfcommand.append("ColumnEnergyOutputter_batchElement_*.txt");
      int rmrfstatus = system(rmrfcommand.c_str());
      FatalIf(
            rmrfstatus,
            "Global rank command \"system(%s)\") returned %d\n",
            rmrfcommand.c_str(),
            rmrfstatus);
   }
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
   paramsString.append("ColumnEnergyOutputter \"").append(probeName).append("\" = {\n");
   paramsString.append("   textOutputFlag  = true;\n");
   paramsString.append("   probeOutputFile = \"ColumnEnergyOutputter.txt\";\n");
   paramsString.append("   message         = \"ColumnEnergyOutputter\";\n");
   paramsString.append("};\n");
   PVParams probeParams(paramsString.c_str(), paramsString.size(), 1UL, comm->globalCommunicator());
   return probeParams;
}

ColumnEnergyOutputter initColumnEnergyOutputter(PV_Init &pv_initObj) {
   Communicator *comm = pv_initObj.getCommunicator();
   std::string probeName("probe");
   PVParams probeParams = generateProbeParams(probeName, comm);
   ColumnEnergyOutputter columnEnergyOutputter(probeName.c_str(), &probeParams, comm);
   columnEnergyOutputter.ioParamsFillGroup(PARAMS_IO_READ);

   // create the output files.
   Checkpointer checkpointer("column", comm, pv_initObj.getArguments());
   columnEnergyOutputter.initOutputStreams(&checkpointer, gLocalBatchWidth);
   // At this point, the output files should exist and be empty.

   return columnEnergyOutputter;
}

void printGlobalStats(PV_Init &pv_init) {
   ProbeDataBuffer<double> normsBuffer;
   generateData(normsBuffer, pv_init.getCommunicator());

   auto columnEnergyOutputter = initColumnEnergyOutputter(pv_init);
   columnEnergyOutputter.printColumnEnergiesBuffer(normsBuffer);
}

int run(PV_Init &pv_init) {
   int status = PV_SUCCESS;

   HyPerCol *hypercol = new HyPerCol(&pv_init);
   // HyPerCol::processParams() creates the output directory and its block subdirectories if needed.
   hypercol->processParams("ColumnEnergyOutputterTest.params");

   Communicator *comm = pv_init.getCommunicator();

   deleteOldFiles(hypercol);

   int nx = hypercol->getNxGlobal();
   int ny = hypercol->getNxGlobal();
   printGlobalStats(pv_init);

   status = checkOutputFiles(hypercol);

   delete hypercol;
   if (status == PV_SUCCESS) {
      InfoLog() << "Test passed.\n";
   }
   return status;
}
