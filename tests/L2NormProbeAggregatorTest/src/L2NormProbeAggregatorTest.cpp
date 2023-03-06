#include <arch/mpi/mpi.h>
#include <columns/Random.hpp>
#include <columns/RandomSeed.hpp>
#include <include/pv_common.h>
#include <io/PVParams.hpp>
#include <probes/L2NormProbeAggregator.hpp>
#include <probes/ProbeData.hpp>
#include <probes/ProbeDataBuffer.hpp>
#include <structures/MPIBlock.hpp>
#include <utils/PVLog.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ios>
#include <memory>
#include <string>
#include <vector>

using PV::L2NormProbeAggregator;
using PV::MPIBlock;
using PV::ProbeData;
using PV::ProbeDataBuffer;
using PV::PVParams;

int const gSeed = 1234567890U;

int checkAggregatedNorms(
      std::shared_ptr<MPIBlock const> mpiBlock,
      std::vector<double> const &allPartialNorms,
      ProbeDataBuffer<double> const &aggregatedStore,
      double exponent);
PVParams generateProbeParams(std::string const &probeName, MPI_Comm comm, double exponent);
void initLogFile(std::shared_ptr<MPIBlock const> mpiBlock);
std::shared_ptr<MPIBlock const> initMPIBlock();
std::vector<double>
randomVals(std::shared_ptr<MPIBlock const> mpiBlock, unsigned int seed, int numValues);
int testAggregateStoredValues(std::shared_ptr<MPIBlock const> mpiBlock, int nbatch);

int main(int argc, char **argv) {
   MPI_Init(&argc, &argv);
   auto mpiBlock = initMPIBlock();
   if (!mpiBlock) {
      MPI_Finalize();
      return EXIT_FAILURE; // error message printed by initMPIBlock()
   }
   initLogFile(mpiBlock);

   int status = PV_SUCCESS;
   if (testAggregateStoredValues(mpiBlock, 4 /*nbatch*/) != PV_SUCCESS) {
      status = PV_FAILURE;
   }
   if (status == PV_SUCCESS) {
      InfoLog() << "Test passed.\n";
   }
   MPI_Finalize();
   return status == PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

int checkAggregatedNorms(
      std::shared_ptr<MPIBlock const> mpiBlock,
      std::vector<double> const &allPartialNorms,
      ProbeDataBuffer<double> const &aggregatedStore,
      double exponent) {
   // Make sure all elements of aggregatedStore have the same size, this size is the local nbatch.
   auto batchSizeMin = static_cast<std::vector<double>::size_type>(-1);
   auto batchSizeMax = static_cast<std::vector<double>::size_type>(0);

   int numTimestamps = static_cast<int>(aggregatedStore.size());
   for (int t = 0; t < numTimestamps; ++t) {
      auto thisSize = aggregatedStore.getData(t).size();
      batchSizeMin  = std::min(batchSizeMin, thisSize);
      batchSizeMax  = std::max(batchSizeMax, thisSize);
   }
   FatalIf(
         batchSizeMin != batchSizeMax,
         "Elements of aggregatedStore differ in size (min = %lu, max = %lu\n",
         (unsigned long int)batchSizeMin,
         (unsigned long int)batchSizeMax);

   int nbatch = static_cast<int>(batchSizeMin);

   // Get the number of processes in the global communicator.
   int globalNumRows        = mpiBlock->getGlobalNumRows();
   int globalNumColumns     = mpiBlock->getGlobalNumColumns();
   int globalBatchDimension = mpiBlock->getGlobalBatchDimension();
   int globalSize           = globalNumRows * globalNumColumns * globalBatchDimension;

   // Make sure that the size of allPartialNorms is consistent with the dimensions of the
   // ProbeDataBuffer object and the MPI configuration.
   int correctNumAllPartialNorms = nbatch * numTimestamps * globalSize;

   FatalIf(
         correctNumAllPartialNorms != static_cast<int>(allPartialNorms.size()),
         "inconsistent arguments to checkAggregatedStats(): "
         "expected allPartialNorms to have size %d; instead it is %d\n",
         correctNumAllPartialNorms,
         static_cast<int>(allPartialNorms.size()));

   // Now check that the partial norms agree with the correct value in aggregatedStore.
   int status = PV_SUCCESS;
   for (int t = 1; t <= numTimestamps; ++t) {
      ProbeData<double> const &normsBatch = aggregatedStore.getData(t - 1);
      double correctTimestamp             = static_cast<double>(t);
      double observedTimestamp            = normsBatch.getTimestamp();
      if (normsBatch.getTimestamp() != correctTimestamp) {
         ErrorLog().printf(
               "aggregatedStore index %d: expected timestamp %f, observed timestamp %f\n",
               t,
               correctTimestamp,
               observedTimestamp);
         status = PV_FAILURE;
      }
      for (int b = 0; b < nbatch; ++b) {
         double observedValue = normsBatch.getValue(b);
         // To get the correct value, we have to sum the partial norms across all rows
         // and columns over the MPI configuration, but staying within the same batch index.
         int numRows    = mpiBlock->getNumRows();
         int numColumns = mpiBlock->getNumColumns();
         int batchIndex = mpiBlock->getBatchIndex();
         double sum     = 0.0;
         for (int r = 0; r < numRows; ++r) {
            for (int c = 0; c < numColumns; ++c) {
               int rank         = mpiBlock->calcRankFromRowColBatch(r, c, batchIndex);
               int globalOffset = mpiBlock->getStartBatch() + mpiBlock->getBatchIndex();
               int globalRank   = rank + globalNumRows * globalNumColumns * globalOffset;
               int index        = b + nbatch * (globalRank + globalSize * (t - 1));
               double v         = allPartialNorms.at(index);
               sum += v;
            }
         }
         double correctValue = std::pow(sum, exponent / 2.0);
         if (observedValue != correctValue) {
            ErrorLog().printf(
                  "timestamp %f, batch element %d: expected %f, observed %f\n",
                  correctTimestamp,
                  b,
                  correctValue,
                  observedValue);
            status = PV_FAILURE;
         }
      }
   }
   return status;
}

PVParams generateProbeParams(std::string const &probeName, MPI_Comm comm, double exponent) {
   std::string paramsString;
   paramsString.append("debugParsing = false;\n");
   paramsString.append("NormProbeOutputter \"").append(probeName).append("\" = {\n");
   paramsString.append("   exponent = ").append(std::to_string(exponent)).append(";");
   paramsString.append("   probeOutputFile = \"NormProbeOutputter.txt\";\n");
   paramsString.append("   message         = \"NormProbeOutputter\";\n");
   paramsString.append("};\n");
   PVParams probeParams(paramsString.c_str(), paramsString.size(), 1UL, comm);
   return probeParams;
}

void initLogFile(std::shared_ptr<MPIBlock const> mpiBlock) {
   int globalRank = mpiBlock->getGlobalRank();
   std::string logFileName("L2NormProbeAggregatorTest_8_");
   logFileName.append(std::to_string(globalRank)).append(".log");
   PV::setLogFile(logFileName.c_str(), std::ios_base::out);
}

std::shared_ptr<MPIBlock const> initMPIBlock() {
   int commSize;
   MPI_Comm_size(MPI_COMM_WORLD, &commSize);
   if (commSize != 8) {
      ErrorLog().printf("L2NormProbeAggregatorTest must be called under MPI with 8 processes.\n");
      MPI_Barrier(MPI_COMM_WORLD);
      return nullptr;
   }
   auto result = std::make_shared<MPIBlock>(MPI_COMM_WORLD, 2, 2, 2, 2, 2, 1);
   // MPI configuration is 2 rows, 2 columns, batch dimension 2.
   // MPI blocks are 2 rows, 2 columns, batch dimension 1.
   // This mimics the Communicator's LocalMPIBlock that L2NormProbeAggregator typically uses.
   return result;
}

std::vector<double>
randomVals(std::shared_ptr<MPIBlock const> mpiBlock, unsigned int seed, int numValues) {
   std::vector<double> result(numValues);
   int rootProc = 0;
   if (mpiBlock->getGlobalRank() == rootProc) {
      PV::RandomSeed::instance()->initialize(seed);
      PV::Random r(1);
      // random values in the range 0.5, 1.5, 2.5, ..., 19.5
      for (auto &x : result) {
         x = static_cast<double>(std::floor(r.uniformRandom(0, 0.0f, 19.999999f)) + 0.5f);
      }
   }
   MPI_Bcast(&result.at(0), numValues, MPI_DOUBLE, rootProc, mpiBlock->getGlobalComm());
   return result;
}

int testAggregateStoredValues(std::shared_ptr<MPIBlock const> mpiBlock, int nbatch) {
   int globalNumRows        = mpiBlock->getGlobalNumRows();
   int globalNumColumns     = mpiBlock->getGlobalNumColumns();
   int globalBatchDimension = mpiBlock->getGlobalBatchDimension();
   int globalSize           = globalNumRows * globalNumColumns * globalBatchDimension;

   int numTimestamps                   = 10;
   int numAllPartialNorms              = globalSize * nbatch * numTimestamps;
   std::vector<double> allPartialNorms = randomVals(mpiBlock, gSeed, numAllPartialNorms);

   ProbeDataBuffer<double> partialStore;
   for (int t = 1; t <= numTimestamps; ++t) {
      double timestamp = static_cast<double>(t);
      ProbeData<double> partialBatch(timestamp, nbatch);
      for (int b = 0; b < nbatch; ++b) {
         int index                = b + nbatch * (mpiBlock->getGlobalRank() + globalSize * (t - 1));
         partialBatch.getValue(b) = allPartialNorms.at(index);
      }
      partialStore.store(partialBatch);
   }

   // Aggregate the values. Note that L2NormProbeAggregator does not read any params
   char const *probeName = "L2NormProbeAggregator";
   MPI_Comm comm         = mpiBlock->getGlobalComm();
   double exponent       = 1.0;
   PVParams params       = generateProbeParams(std::string(probeName), comm, exponent);
   L2NormProbeAggregator normAggregator(probeName, &params, mpiBlock);
   normAggregator.aggregateStoredValues(partialStore);
   ProbeDataBuffer<double> aggregatedStore = normAggregator.getStoredValues();

   int status = checkAggregatedNorms(mpiBlock, allPartialNorms, aggregatedStore, exponent);

   normAggregator.clearStoredValues();
   int clearedSize = static_cast<int>(normAggregator.getStoredValues().size());
   FatalIf(
         clearedSize != 0,
         "L2NormProbeAggregator::clearStoredValues() failed to clear the stored values.\n");

   return status;
}
