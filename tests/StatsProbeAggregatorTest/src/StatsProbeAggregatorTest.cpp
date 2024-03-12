#include <arch/mpi/mpi.h>
#include <include/PVLayerLoc.hpp>
#include <include/pv_common.h>
#include <probes/ProbeData.hpp>
#include <probes/ProbeDataBuffer.hpp>
#include <probes/StatsProbeAggregator.hpp>
#include <probes/StatsProbeTypes.hpp>
#include <structures/MPIBlock.hpp>
#include <utils/PVLog.hpp>

#include "CheckValue.hpp"

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <ios>
#include <memory>
#include <string>

using PV::checkValue;
using PV::ErrorLog;
using PV::LayerStats;
using PV::MPIBlock;
using PV::ProbeData;
using PV::ProbeDataBuffer;
using PV::StatsProbeAggregator;

int checkAggregatedStats(
      std::string const &messageHead,
      LayerStats const &correct,
      LayerStats const &observed);

LayerStats computeCorrect(int globalBatchIndex, int modifier);
void computeTestData(LayerStats &stats, int index, int modifier);
ProbeData<LayerStats> const initData(
      double timestamp,
      PVLayerLoc const &loc,
      std::shared_ptr<MPIBlock const> mpiBlock,
      int modifier);
PVLayerLoc initLayerLoc(std::shared_ptr<MPIBlock const> mpiBlock);
void initLogFile(std::shared_ptr<MPIBlock const> mpiBlock);
std::shared_ptr<MPIBlock const> initMPIBlock();
int testAggregateStoredValues(std::shared_ptr<MPIBlock const> mpiBlock, PVLayerLoc const &loc);

int main(int argc, char **argv) {
   MPI_Init(&argc, &argv);
   auto mpiBlock = initMPIBlock();
   if (!mpiBlock) {
      MPI_Finalize();
      return EXIT_FAILURE; // error message printed by initMPIBlock()
   }
   initLogFile(mpiBlock);
   auto loc = initLayerLoc(mpiBlock);

   int status = PV_SUCCESS;
   if (testAggregateStoredValues(mpiBlock, loc) != PV_SUCCESS) {
      status = PV_FAILURE;
   }
   if (status == PV_SUCCESS) {
      InfoLog() << "Test passed.\n";
   }
   MPI_Finalize();
   return status == PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

int checkAggregatedStats(
      std::string const &messageHead,
      LayerStats const &correct,
      LayerStats const &observed) {
   int status = PV_SUCCESS;

   try {
      checkValue(messageHead, "Sum", correct.mSum, observed.mSum, 0.0);
   } catch (std::exception const &e) {
      ErrorLog() << e.what();
      status = PV_FAILURE;
   }
   try {
      checkValue(messageHead, "SumSquared", correct.mSumSquared, observed.mSumSquared, 0.0);
   } catch (std::exception const &e) {
      ErrorLog() << e.what();
      status = PV_FAILURE;
   }
   try {
      checkValue(messageHead, "Min", correct.mMin, observed.mMin, 0.0f);
   } catch (std::exception const &e) {
      ErrorLog() << e.what();
      status = PV_FAILURE;
   }
   try {
      checkValue(messageHead, "Max", correct.mMax, observed.mMax, 0.0f);
   } catch (std::exception const &e) {
      ErrorLog() << e.what();
      status = PV_FAILURE;
   }
   try {
      checkValue(messageHead, "NumNeurons", correct.mNumNeurons, observed.mNumNeurons, 0);
   } catch (std::exception const &e) {
      ErrorLog() << e.what();
      status = PV_FAILURE;
   }
   try {
      checkValue(messageHead, "NumNonzero", correct.mNumNonzero, observed.mNumNonzero, 0);
   } catch (std::exception const &e) {
      ErrorLog() << e.what();
      status = PV_FAILURE;
   }
   return status;
}

void computeTestData(LayerStats &stats, int index, int modifier) {
   stats.mSum        = static_cast<float>((100 + modifier) * (1 + index));
   stats.mSumSquared = static_cast<float>((1000 + 2 * modifier) * (1 + index) * (1 + index));

   float yMin = 0.321f + 0.001f * static_cast<float>(modifier);
   float yMax = 1.321f - 0.001f * static_cast<float>(modifier);
   for (int n = 0; n < index; ++n) {
      yMin = 4.0f * yMin * (1.0f - yMin);
      yMax = (12.0f - 4.0f * yMax) * yMax - 7.0f;
   }
   stats.mMin        = yMin;
   stats.mMax        = yMax;
   stats.mNumNeurons = 64;
   stats.mNumNonzero = 16 - index + modifier;
}

ProbeData<LayerStats> const initData(
      double timestamp,
      PVLayerLoc const &loc,
      std::shared_ptr<MPIBlock const> mpiBlock,
      int modifier) {
   ProbeData<LayerStats> result(timestamp, loc.nbatch);
   int rowIndex    = mpiBlock->getStartRow() + mpiBlock->getRowIndex();
   int columnIndex = mpiBlock->getStartColumn() + mpiBlock->getColumnIndex();
   int locIndex    = rowIndex * mpiBlock->getGlobalNumColumns() + columnIndex;
   int maxLoc      = mpiBlock->getGlobalNumRows() * mpiBlock->getGlobalNumColumns();
   int batchIndex  = mpiBlock->getStartBatch() + mpiBlock->getBatchIndex();
   for (int b = 0; b < loc.nbatch; ++b) {
      int index = (batchIndex * loc.nbatch + b) * maxLoc + locIndex;
      computeTestData(result.getValue(b), index, modifier);
   }
   return result;
}

void initLogFile(std::shared_ptr<MPIBlock const> mpiBlock) {
   int globalRank = mpiBlock->getGlobalRank();
   std::string logFileName("StatsProbeAggregatorTest_8_");
   logFileName.append(std::to_string(globalRank)).append(".log");
   PV::setLogFile(logFileName.c_str(), std::ios_base::out);
}

PVLayerLoc initLayerLoc(std::shared_ptr<MPIBlock const> mpiBlock) {
   PVLayerLoc loc;
   loc.nbatch = 2;
   loc.nx     = 4;
   loc.ny     = 4;
   loc.nf     = 1;

   loc.nbatchGlobal = 4;
   loc.nxGlobal     = 8;
   loc.nyGlobal     = 8;
   loc.kb0          = (mpiBlock->getStartBatch() + mpiBlock->getBatchIndex()) * 4;
   loc.kx0          = (mpiBlock->getStartColumn() + mpiBlock->getColumnIndex()) * loc.nx;
   loc.ky0          = (mpiBlock->getStartRow() + mpiBlock->getRowIndex()) * loc.ny;
   loc.kb0          = (mpiBlock->getStartBatch() + mpiBlock->getBatchIndex()) * loc.nbatch;
   loc.halo.lt      = 0;
   loc.halo.rt      = 0;
   loc.halo.dn      = 0;
   loc.halo.up      = 0;
   return loc;
}

std::shared_ptr<MPIBlock const> initMPIBlock() {
   int commSize;
   MPI_Comm_size(MPI_COMM_WORLD, &commSize);
   if (commSize != 8) {
      ErrorLog().printf("StatsProbeAggregatorTest must be called under MPI with 8 processes.\n");
      MPI_Barrier(MPI_COMM_WORLD);
      return nullptr;
   }
   auto result = std::make_shared<MPIBlock>(MPI_COMM_WORLD, 2, 2, 2, 2, 2, 1);
   // MPI configuration is 2 rows, 2 columns, batch dimension 2.
   // MPI blocks are 2 rows, 2 columns, batch dimension 1.
   // This mimics the Communicator's LocalMPIBlock that StatsProbeAggregator typically uses.
   return result;
}

LayerStats computeCorrect(int globalBatchIndex, int modifier) {
   LayerStats result;
   for (int index = 4 * globalBatchIndex; index < 4 * (globalBatchIndex + 1); index++) {
      LayerStats partialStats;
      computeTestData(partialStats, index, modifier);
      result.mSum += partialStats.mSum;
      result.mSumSquared += partialStats.mSumSquared;
      result.mMin = std::min(result.mMin, partialStats.mMin);
      result.mMax = std::max(result.mMax, partialStats.mMax);
      result.mNumNeurons += partialStats.mNumNeurons;
      result.mNumNonzero += partialStats.mNumNonzero;
   }
   return result;
}

int testAggregateStoredValues(std::shared_ptr<MPIBlock const> mpiBlock, PVLayerLoc const &loc) {
   int status = PV_SUCCESS;
   std::string desc("testAggregateStoredValues()");

   // Create partial stats on each process
   ProbeDataBuffer<LayerStats> partialStatsStore;
   for (int k = 0; k < 4; ++k) {
      double timestamp = static_cast<double>(k + 1);
      partialStatsStore.store(initData(timestamp, loc, mpiBlock, (k + 1) /*modifier*/));
   }

   // Aggregate the values. Note that StatsProbeAggregator does not read any params
   StatsProbeAggregator statsProbeAggregator("StatsProbeAggregator", nullptr /*params*/, mpiBlock);
   statsProbeAggregator.aggregateStoredValues(partialStatsStore);

   // Check that the values are correct
   auto const &aggregateStore = statsProbeAggregator.getStoredValues();

   int aggregateSize         = static_cast<int>(aggregateStore.size());
   int partialStatsStoreSize = static_cast<int>(partialStatsStore.size());
   try {
      checkValue(desc, "size", aggregateSize, partialStatsStoreSize, 0);
   } catch (std::exception const &e) {
      ErrorLog() << e.what();
      status = PV_FAILURE;
   }

   for (int n = 0; n < aggregateSize; ++n) {
      for (int b = 0; b < aggregateStore.getData(n).size(); ++b) {
         int bGlobal = (mpiBlock->getStartBatch() + mpiBlock->getBatchIndex()) * loc.nbatch + b;
         LayerStats correctStats           = computeCorrect(bGlobal, (n + 1) /*modifier*/);
         LayerStats const &aggregatedStats = aggregateStore.getData(n).getValue(b);
         std::string messageHead(desc + " batch element " + std::to_string(bGlobal));
         if (checkAggregatedStats(messageHead, correctStats, aggregatedStats) != PV_SUCCESS) {
            status = PV_FAILURE;
         }
      }
   }
   return status;
}
