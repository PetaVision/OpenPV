#include "cMakeHeader.h"

#include "AdaptiveTimeScaleProbeOutputter.hpp"

#include "arch/mpi/mpi.h"
#include "probes/ProbeData.hpp"
#include "utils/PVAssert.hpp"
#include <vector>

namespace PV {

AdaptiveTimeScaleProbeOutputter::AdaptiveTimeScaleProbeOutputter(
      char const *objName,
      PVParams *params,
      Communicator const *comm) {
   initialize(objName, params, comm);
}

void AdaptiveTimeScaleProbeOutputter::initialize(
      char const *objName,
      PVParams *params,
      Communicator const *comm) {
   BaseProbeOutputter::initialize(objName, params, comm);
}

void AdaptiveTimeScaleProbeOutputter::ioParam_writeTimeScaleFieldnames(enum ParamsIOFlag ioFlag) {
   pvAssert(!getParams()->presentAndNotBeenRead(getName_c(), "textOutputFlag"));
   if (getTextOutputFlag()) {
      getParams()->ioParamValue(
            ioFlag,
            getName_c(),
            "writeTimeScaleFieldnames",
            &mWriteTimeScaleFieldnames,
            mWriteTimeScaleFieldnames);
   }
}

void AdaptiveTimeScaleProbeOutputter::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   BaseProbeOutputter::ioParamsFillGroup(ioFlag);
   ioParam_writeTimeScaleFieldnames(ioFlag);
}

void AdaptiveTimeScaleProbeOutputter::printTimeScaleBuffer(
      ProbeDataBuffer<TimeScaleData> const &storedValues) {
   if (getTextOutputFlag()) {
      if (getProbeOutputFilename() and getProbeOutputFilename()[0]) {
         printToFiles(storedValues);
      }
      else {
         printToLog(storedValues);
      }
   }
}

void AdaptiveTimeScaleProbeOutputter::printTimeScaleData(
      std::shared_ptr<PrintStream> printStreamPtr,
      double timestamp,
      int batchIndex,
      TimeScaleData const &timeScaleData) {
   char const *formatStr;
   if (mWriteTimeScaleFieldnames) {
      formatStr = "sim_time = %f\n"
            "\tbatch = %d, timeScale = %10.8f, timeScaleTrue = %10.8f, timeScaleMax = %10.8f\n";
   }
   else {
      formatStr = "%f, %d, %10.8f, %10.8f, %10.8f\n";
   }
   printStreamPtr->printf(
         formatStr,
         timestamp,
         batchIndex,
         timeScaleData.mTimeScale,
         timeScaleData.mTimeScaleTrue,
         timeScaleData.mTimeScaleMax);
   printStreamPtr->flush();
}

void AdaptiveTimeScaleProbeOutputter::printToFiles(ProbeDataBuffer<TimeScaleData> const &storedValues) {
   pvAssert(getProbeOutputFilename() != nullptr and getProbeOutputFilename()[0] != '\0');
#ifdef PV_USE_MPI
   if (getCommunicator()->commRank() != 0) {
      return;
   }
   MPI_Comm ioComm            = getIOMPIBlock()->getComm();
   unsigned int numTimestamps = storedValues.size();
   unsigned int batchWidth    = storedValues.getBatchWidth();
   unsigned int transferSize  = ProbeDataBuffer<TimeScaleData>::calcPackedSize(numTimestamps, batchWidth);
   int const mpiTag           = 204;
   if (getIOMPIBlock()->getRank() == 0) {
      int blockBatchDimension = getIOMPIBlock()->getBatchDimension();
      std::vector<char> recvData(transferSize);
      std::vector<ProbeDataBuffer<TimeScaleData>> recvNorms;
      recvNorms.reserve(blockBatchDimension - 1);
      for (int blockBatchIndex = 1; blockBatchIndex < blockBatchDimension; ++blockBatchIndex) {
         int ioRank = getIOMPIBlock()->calcRankFromRowColBatch(0, 0, blockBatchIndex);
         MPI_Recv(
               recvData.data(), transferSize, MPI_BYTE, ioRank, mpiTag, ioComm, MPI_STATUS_IGNORE);
         recvNorms.push_back(ProbeDataBuffer<TimeScaleData>::unpack(recvData));
      }
      int globalBatchStart = calcGlobalBatchOffset();

      // We can expect each process's ProbeAggregator to consist of several batch elements and
      // several timestamps. We need to print the stats for all batch elements for one timestamp
      // before moving on to the next timestamp.
      for (unsigned int n = 0; n < numTimestamps; ++n) {
         int numBatchElementsInBlock = getLocalNBatch() * blockBatchDimension;
         for (int b = 0; b < numBatchElementsInBlock; ++b) {
            int bLocal = b % getLocalNBatch();
            int bBlock = (b - bLocal) / getLocalNBatch();
            ProbeData<TimeScaleData> const &timeScaleBatch =
                  bBlock ? recvNorms[bBlock - 1].getData(n) : storedValues.getData(n);
            auto &stats      = timeScaleBatch.getValue(bLocal);
            double timestamp = timeScaleBatch.getTimestamp();
            printTimeScaleData(returnOutputStream(b), timestamp, globalBatchStart + b, stats);
         }
      }
   }
   else {
      // this process has the aggregate stats for some batch element(s), but is not a process
      // that does I/O. Therefore, we need to send the aggregate stats over MPI.
      std::vector<char> packedStore = storedValues.pack();
      char const *sendData          = packedStore.data();
      MPI_Send(sendData, transferSize, MPI_BYTE, 0 /*root rank*/, mpiTag, ioComm);
   }
#else // PV_USE_MPI
   unsigned int numTimestamps = storedValues.size();
   for (unsigned int n = 0; n < numTimestamps; ++n) {
      for (int b = 0; b < getLocalNBatch(); ++b) {
         ProbeData<TimeScaleData> const &timeScaleBatch = storedValues.getData(n);
         auto &norm                          = timeScaleBatch.getValue(b);
         double timestamp                    = timeScaleBatch.getTimestamp();
         printTimeScaleData(returnOutputStream(b), timestamp, b, stats);
      }
   }
#endif // PV_USE_MPI
}

void AdaptiveTimeScaleProbeOutputter::printToLog(ProbeDataBuffer<TimeScaleData> const &storedValues) {
   pvAssert(getProbeOutputFilename() == nullptr or getProbeOutputFilename()[0] == '\0');
   int rank = getCommunicator()->commRank();
   if (rank == 0) {
      int globalBatchStart = calcGlobalBatchOffset();
      int aggregatorSize   = storedValues.size();
      for (int n = 0; n < aggregatorSize; ++n) {
         ProbeData<TimeScaleData> const &timeScaleBatch = storedValues.getData(n);
         int batchSize                                  = static_cast<int>(timeScaleBatch.size());
         for (int b = 0; b < batchSize; ++b) {
            auto printStreamPtr = returnOutputStream(b);
            printTimeScaleData(
                  printStreamPtr,
                  storedValues.getTimestamp(n),
                  globalBatchStart + b,
                  storedValues.getValue(n, b));
         }
      }
   }
}

} // namespace PV
