#include <checkpointing/Checkpointer.hpp>
#include <columns/Communicator.hpp>
#include <columns/HyPerCol.hpp>
#include <columns/PV_Init.hpp>
#include <include/pv_common.h>
#include <io/FileManager.hpp>
#include <io/FileStreamBuilder.hpp>
#include <io/PVParams.hpp>
#include <probes/ProbeData.hpp>
#include <probes/ProbeDataBuffer.hpp>
#include <probes/StatsProbeOutputter.hpp>
#include <probes/StatsProbeTypes.hpp>
#include <utils/PVLog.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>

#define LOCALBATCHWIDTH 2
#define NUMTIMESTAMPS 1

using namespace PV;

struct OutputData {
   double mTimestamp;
   int mBatchIndex;
   int mNumNeurons;
   double mTotal;
   float mMin;
   float mMax;
   double mAverage;
   double mSigma;
   int mNumNonzero;
};

int checkOutputFiles(HyPerCol *hypercol);
int compare(
      int timestamp,
      int batchIndex,
      LayerStats const &correctStats,
      OutputData const &observedData);
void deleteOldFiles(HyPerCol *hypercol);
void generateData(LayerStats &stats, int timeIndex, int batchWidth, int batchIndex);
void generateData(ProbeDataBuffer<LayerStats> &data, Communicator *comm);
PVParams generateProbeParams(std::string const &probeName, Communicator *comm);
StatsProbeOutputter initStatsProbeOutputter(PV_Init &pv_initObj);
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

   int const globalBatchWidth     = ioMPIBlock->getGlobalBatchDimension() * LOCALBATCHWIDTH;
   int const globalBatchProcIndex = (ioMPIBlock->getStartBatch() + ioMPIBlock->getBatchIndex());

   int const batchStart = globalBatchProcIndex * LOCALBATCHWIDTH;
   int const batchStop  = batchStart + ioMPIBlock->getBatchDimension() * LOCALBATCHWIDTH;

   for (int b = batchStart; b < batchStop; ++b) {
      std::string outputPath("StatsProbeOutputter_batchElement_");
      outputPath.append(std::to_string(b)).append(".txt");
      auto fileStream = FileStreamBuilder(fileManager, outputPath, true, true, false, false).get();
      std::string fullOutputPath = fileStream->getFileName();
      fileStream                 = nullptr; // close the file.
      // We'll reopen it using cstdio's fopen(), in order to read the values using fscanf().
      FILE *fp = fopen(fullOutputPath.c_str(), "r");
      FatalIf(fp == nullptr, "Unable to open %s for reading\n", fullOutputPath.c_str());
      for (int t = 0; t < NUMTIMESTAMPS; ++t) {
         LayerStats correctStats;
         generateData(correctStats, t, globalBatchWidth, b);
         OutputData observedValues;
         int numread = fscanf(
               fp,
               "StatsProbeOutputter:"
               "t==%lf b==%d N==%d Total==%lf Min==%f Avg==%lf Max==%f sigma==%lf nnz==%d\n",
               &observedValues.mTimestamp,
               &observedValues.mBatchIndex,
               &observedValues.mNumNeurons,
               &observedValues.mTotal,
               &observedValues.mMin,
               &observedValues.mAverage,
               &observedValues.mMax,
               &observedValues.mSigma,
               &observedValues.mNumNonzero);
         if (compare(t, b, correctStats, observedValues) != PV_SUCCESS) {
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
      LayerStats const &correctStats,
      OutputData const &observedData) {
   double dNumNeurons    = static_cast<double>(correctStats.mNumNeurons);
   double correctAverage = correctStats.mSum / dNumNeurons;
   double correctVariance =
         correctStats.mSumSquared / dNumNeurons - correctAverage * correctAverage;
   double correctSigma = std::sqrt(correctVariance);

   int status = PV_SUCCESS;

   double timeDbl = static_cast<double>(timestamp);
   if (observedData.mTimestamp != timeDbl) {
      ErrorLog().printf(
            "expected timestamp %f; observed timestamp %f\n", timeDbl, observedData.mTimestamp);
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

   if (observedData.mNumNeurons != correctStats.mNumNeurons) {
      ErrorLog().printf(
            "timestamp %d, batch index %d, expected N = %d; observed N = %d\n",
            timestamp,
            batchIndex,
            correctStats.mNumNeurons,
            observedData.mNumNeurons);
      status = PV_FAILURE;
   }

   if (observedData.mTotal != correctStats.mSum) {
      ErrorLog().printf(
            "timestamp %d, batch index %d, expected Total = %f; observed Total = %f; discrepancy "
            "%g\n",
            timestamp,
            batchIndex,
            correctStats.mSum,
            observedData.mTotal,
            observedData.mTotal - correctStats.mSum);
      status = PV_FAILURE;
   }

   if (observedData.mMin != correctStats.mMin) {
      ErrorLog().printf(
            "timestamp %d, batch index %d, expected Min = %f; observed Min = %f; discrepancy %g\n",
            timestamp,
            batchIndex,
            (double)correctStats.mMin,
            (double)observedData.mMin,
            (double)(observedData.mMin - correctStats.mMin));
      status = PV_FAILURE;
   }

   if (std::abs(observedData.mAverage - correctAverage) > 6.0e-7) {
      ErrorLog().printf(
            "timestamp %d, batch index %d, expected Avg = %f; observed Avg = %f; discrepancy %g\n",
            timestamp,
            batchIndex,
            correctAverage,
            (double)observedData.mAverage,
            observedData.mAverage - correctAverage);
      status = PV_FAILURE;
   }

   if (observedData.mMax != correctStats.mMax) {
      ErrorLog().printf(
            "timestamp %d, batch index %d, expected Max = %f; observed Max = %f; discrepancy %g\n",
            timestamp,
            batchIndex,
            (double)correctStats.mMax,
            (double)observedData.mMax,
            (double)(observedData.mMax - correctStats.mMax));
      status = PV_FAILURE;
   }

   if (std::abs(observedData.mSigma - correctSigma) > 6.0e-7) {
      ErrorLog().printf(
            "timestamp %d, batch index %d, expected Sigma = %f; observed Sigma = %f; discrepancy "
            "%g\n",
            timestamp,
            batchIndex,
            correctSigma,
            (double)observedData.mSigma,
            observedData.mSigma - correctSigma);
      status = PV_FAILURE;
   }

   if (observedData.mNumNonzero != correctStats.mNumNonzero) {
      ErrorLog().printf(
            "timestamp %d, batch index %d, expected nnz = %d; observed nnz = %d\n",
            timestamp,
            batchIndex,
            correctStats.mNumNonzero,
            observedData.mNumNonzero);
      status = PV_FAILURE;
   }

   return status;
}

void deleteOldFiles(HyPerCol *hypercol) {
   // Delete old StatsProbeOutputter output files, if they exist. That way we know, at the end of
   // the run, that if the files exist, they were produced by this run.
   // It would be better to do this without using the blunt hammer of calling system(rm -rf ...).
   auto const ioMPIBlock = hypercol->getPV_InitObj()->getCommunicator()->getIOMPIBlock();
   FileManager fileManager(ioMPIBlock, std::string(hypercol->getOutputPath()));
   if (ioMPIBlock->getRank() == 0) {
      std::string outputBlockDir = fileManager.makeBlockFilename(".");
      std::string rmrfcommand("rm -rf ");
      rmrfcommand.append(outputBlockDir).append("/");
      rmrfcommand.append("StatsProbeOutputter_batchElement_*.txt");
      int rmrfstatus = system(rmrfcommand.c_str());
      FatalIf(
            rmrfstatus,
            "Global rank command \"system(%s)\") returned %d\n",
            rmrfcommand.c_str(),
            rmrfstatus);
   }
}

// Create some fake data. Don't worry if there's a real dataset that could
// produce these numbers. All we're testing is whether data gets relayed
// to the IOMPIBlock's root process and printed correctly.
void generateData(LayerStats &stats, int timeIndex, int batchWidth, int batchIndex) {
   double datapoint  = static_cast<double>(timeIndex * batchWidth + batchIndex);
   stats.mSum        = datapoint;
   stats.mSumSquared = datapoint * datapoint;
   stats.mMin        = static_cast<float>(-(batchIndex + 1));
   stats.mMax        = static_cast<float>(timeIndex + 1);
   stats.mNumNeurons = 256;
   stats.mNumNonzero = 10 + batchWidth - batchIndex - timeIndex;
}

void generateData(ProbeDataBuffer<LayerStats> &data, Communicator *comm) {
   data.clear();
   auto localMPIBlock = comm->getLocalMPIBlock();
   if (localMPIBlock->getRank() != 0) {
      return;
   }
   auto ioMPIBlock                = comm->getIOMPIBlock();
   int const globalBatchWidth     = ioMPIBlock->getGlobalBatchDimension() * LOCALBATCHWIDTH;
   int const globalBatchProcIndex = (ioMPIBlock->getStartBatch() + ioMPIBlock->getBatchIndex());
   for (int t = 0; t < NUMTIMESTAMPS; ++t) {
      double timestamp = static_cast<double>(t);
      ProbeData<LayerStats> stats(timestamp, LOCALBATCHWIDTH);
      for (int b = 0; b < LOCALBATCHWIDTH; ++b) {
         int globalBatchIndex = b + LOCALBATCHWIDTH * globalBatchProcIndex;
         generateData(stats.getValue(b), timestamp, globalBatchWidth, globalBatchIndex);
      }
      data.store(stats);
   }
}

PVParams generateProbeParams(std::string const &probeName, Communicator *comm) {
   std::string paramsString;
   paramsString.append("debugParsing = false;\n");
   paramsString.append("StatsProbeOutputter \"").append(probeName).append("\" = {\n");
   paramsString.append("   textOutputFlag  = true;\n");
   paramsString.append("   probeOutputFile = \"StatsProbeOutputter.txt\";\n");
   paramsString.append("   message         = \"StatsProbeOutputter\";\n");
   paramsString.append("};\n");
   PVParams probeParams(paramsString.c_str(), paramsString.size(), 1UL, comm->globalCommunicator());
   return probeParams;
}

StatsProbeOutputter initStatsProbeOutputter(PV_Init &pv_initObj) {
   Communicator *comm = pv_initObj.getCommunicator();
   std::string probeName("probe");
   PVParams probeParams = generateProbeParams(probeName, comm);
   StatsProbeOutputter statsProbeOutputter(probeName.c_str(), &probeParams, comm);
   statsProbeOutputter.ioParamsFillGroup(PARAMS_IO_READ);

   // create the output files.
   Checkpointer checkpointer("column", comm, pv_initObj.getArguments());
   statsProbeOutputter.initOutputStreams(&checkpointer, LOCALBATCHWIDTH);
   // At this point, the output files should exist and be empty.

   statsProbeOutputter.setConvertToHertzFlag(false);
   // for completeness, should test with this flag set to true as well.

   return statsProbeOutputter;
}

void printGlobalStats(PV_Init &pv_init) {
   ProbeDataBuffer<LayerStats> statsBatchVector;
   generateData(statsBatchVector, pv_init.getCommunicator());

   auto statsProbeOutputter = initStatsProbeOutputter(pv_init);
   statsProbeOutputter.printGlobalStatsBuffer(statsBatchVector);
}

int run(PV_Init &pv_init) {
   int status = PV_SUCCESS;

   HyPerCol *hypercol = new HyPerCol(&pv_init);
   // HyPerCol::processParams() creates the output directory and its block subdirectories if needed.
   hypercol->processParams("StatsProbeOutputterTest.params");

   Communicator *comm = pv_init.getCommunicator();

   deleteOldFiles(hypercol);

   printGlobalStats(pv_init);

   status = checkOutputFiles(hypercol);

   if (status == PV_SUCCESS) {
      InfoLog() << "Test passed.\n";
   }
   return status;
}
