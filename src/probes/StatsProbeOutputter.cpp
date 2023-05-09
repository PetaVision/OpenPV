#include "cMakeHeader.h"

#include "StatsProbeOutputter.hpp"

#include "arch/mpi/mpi.h"
#include "probes/ProbeData.hpp"
#include "probes/ProbeDataBuffer.hpp"
#include "probes/StatsProbeTypes.hpp"
#include "utils/PVAssert.hpp"
#include <vector>

namespace PV {

StatsProbeOutputter::StatsProbeOutputter(
      char const *objName,
      PVParams *params,
      Communicator const *comm) {
   initialize(objName, params, comm);
}

void StatsProbeOutputter::initialize(
      char const *objName,
      PVParams *params,
      Communicator const *comm) {
   BaseProbeOutputter::initialize(objName, params, comm);
}

void StatsProbeOutputter::printGlobalStatsBuffer(ProbeDataBuffer<LayerStats> const &storedValues) {
   if (getTextOutputFlag()) {
      if (getProbeOutputFilename() and getProbeOutputFilename()[0]) {
         printToFiles(storedValues);
      }
      else {
         printToLog(storedValues);
      }
      flushOutputStreams();
   }
}

void StatsProbeOutputter::printLayerStats(
      std::shared_ptr<PrintStream> printStreamPtr,
      LayerStats const &stats,
      double timestamp,
      int batchIndex) {
   char const *averageNote = "";
   double multiplier       = 1.0;
   if (mConvertToHertz) {
      // Convert spikes per millisecond to hertz.
      averageNote = " Hz (/dt ms)";
      multiplier  = 1000.0;
   }
   double average, sigma;
   stats.derivedStats(average, sigma);
   printStreamPtr->printf(
         "%st==%6.1f b==%d N==%d Total==%f Min==%f Avg==%f%s Max==%f sigma==%f nnz==%d\n",
         getMessage().c_str(),
         timestamp,
         batchIndex,
         stats.mNumNeurons,
         stats.mSum,
         (double)stats.mMin,
         average * multiplier,
         averageNote,
         (double)stats.mMax,
         sigma,
         stats.mNumNonzero);
}

void StatsProbeOutputter::printToFiles(ProbeDataBuffer<LayerStats> const &storedValues) {
   pvAssert(getProbeOutputFilename() != nullptr and getProbeOutputFilename()[0] != '\0');
#ifdef PV_USE_MPI
   if (getCommunicator()->commRank() != 0) {
      return;
   }
   MPI_Comm ioComm = getIOMPIBlock()->getComm();
   int rootProc    = 0;

   auto packedSizePerElement = ProbeData<LayerStats>::calcPackedSize(storedValues.getBatchWidth());
   auto packedSizeU = 2U * sizeof(unsigned int) + packedSizePerElement * storedValues.size();
   int packedSize   = static_cast<int>(packedSizeU);

   int const mpiTag = 202;
   if (getIOMPIBlock()->getRank() == rootProc) {
      int blockBatchDimension = getIOMPIBlock()->getBatchDimension();
      std::vector<char> recvData(packedSize);
      std::vector<ProbeDataBuffer<LayerStats>> recvStats;
      recvStats.reserve(blockBatchDimension - 1);
      for (int blockBatchIndex = 1; blockBatchIndex < blockBatchDimension; ++blockBatchIndex) {
         int ioRank = getIOMPIBlock()->calcRankFromRowColBatch(0, 0, blockBatchIndex);
         MPI_Recv(recvData.data(), packedSize, MPI_CHAR, ioRank, mpiTag, ioComm, MPI_STATUS_IGNORE);
         recvStats.push_back(ProbeDataBuffer<LayerStats>::unpack(recvData));
      }
      int globalBatchStart = calcGlobalBatchOffset();

      // We can expect each process's ProbeAggregator to consist of several batch elements and
      // several timestamps. We need to print the stats for all batch elements for one timestamp
      // before moving on to the next timestamp.
      unsigned int numTimestamps = storedValues.size();
      for (unsigned int n = 0; n < numTimestamps; ++n) {
         int numBatchElementsInBlock = getLocalNBatch() * blockBatchDimension;
         for (int b = 0; b < numBatchElementsInBlock; ++b) {
            int bLocal = b % getLocalNBatch();
            int bBlock = (b - bLocal) / getLocalNBatch();
            ProbeData<LayerStats> const &layerStatsBatch =
                  bBlock ? recvStats[bBlock - 1].getData(n) : storedValues.getData(n);
            auto &stats      = layerStatsBatch.getValue(bLocal);
            double timestamp = layerStatsBatch.getTimestamp();
            printLayerStats(returnOutputStream(b), stats, timestamp, globalBatchStart + b);
         }
      }
   }
   else {
      // this process has the aggregate stats for some batch element(s), but is not a process
      // that does I/O. Therefore, we need to send the aggregate stats over MPI.
      std::vector<char> packedStore = storedValues.pack();
      pvAssert(packedStore.size() == packedSizeU);
      char const *sendData = packedStore.data();
      MPI_Send(sendData, packedSize, MPI_CHAR, rootProc, mpiTag, ioComm);
   }
#else // PV_USE_MPI
   unsigned int numTimestamps = storedValues.size();
   for (unsigned int n = 0; n < numTimestamps; ++n) {
      for (int b = 0; b < getLocalNBatch(); ++b) {
         ProbeData<LayerStats> const &layerStatsBatch = storedValues.getData(n);
         auto &stats                                  = layerStatsBatch.getValue(b);
         double timestamp                             = layerStatsBatch.getTimestamp();
         printLayerStats(returnOutputStream(b), stats, timestamp, b);
      }
   }
#endif // PV_USE_MPI
}

void StatsProbeOutputter::printToLog(ProbeDataBuffer<LayerStats> const &storedValues) {
   pvAssert(getProbeOutputFilename() == nullptr or getProbeOutputFilename()[0] == '\0');
   int rank = getCommunicator()->commRank();
   if (rank == 0) {
      int globalBatchStart = calcGlobalBatchOffset();
      int aggregatorSize   = storedValues.size();
      for (int n = 0; n < aggregatorSize; ++n) {
         ProbeData<LayerStats> const &statsBatch = storedValues.getData(n);
         int batchSize                           = statsBatch.size();
         for (int b = 0; b < batchSize; ++b) {
            auto printStreamPtr = returnOutputStream(b);
            printLayerStats(
                  printStreamPtr,
                  statsBatch.getValue(b),
                  statsBatch.getTimestamp(),
                  globalBatchStart + b);
         }
      }
   }
}

void StatsProbeOutputter::setConvertToHertzFlag(bool convertToHertzFlag) {
   mConvertToHertz = convertToHertzFlag;
}

} // namespace PV
