#include "ColumnEnergyOutputter.hpp"

namespace PV {

ColumnEnergyOutputter::ColumnEnergyOutputter(
      char const *objName,
      PVParams *params,
      Communicator const *comm) {
   initialize(objName, params, comm);
}

void ColumnEnergyOutputter::initialize(
      char const *objName,
      PVParams *params,
      Communicator const *comm) {
   BaseProbeOutputter::initialize(objName, params, comm);
}

void ColumnEnergyOutputter::printColumnEnergiesBuffer(ProbeDataBuffer<double> const &storedValues) {
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

void ColumnEnergyOutputter::printEnergy(
      std::shared_ptr<PrintStream> printStreamPtr,
      double timestamp,
      int globalBatchIndex,
      double energy) {
   if (!isWritingToFile()) {
      printStreamPtr->printf("\"%s\%,", getName_c()); // lack of \n is deliberate
   }
   printStreamPtr->printf("%10f, %d, %10.9f\n", timestamp, globalBatchIndex, energy);
}

void ColumnEnergyOutputter::printHeader() {
   if (isWritingToFile()) {
      printStringToAll("time,index,energy");
   }
   else {
      printStringToAll("Probe_name,time,index,energy");
   }
}

void ColumnEnergyOutputter::printToFiles(ProbeDataBuffer<double> const &storedValues) {
   pvAssert(getProbeOutputFilename() != nullptr and getProbeOutputFilename()[0] != '\0');
#ifdef PV_USE_MPI
   if (getCommunicator()->commRank() != 0) {
      return;
   }
   MPI_Comm ioComm            = getIOMPIBlock()->getComm();
   unsigned int numTimestamps = storedValues.size();
   unsigned int batchWidth    = storedValues.getBatchWidth();
   unsigned int transferSize  = ProbeDataBuffer<double>::calcPackedSize(numTimestamps, batchWidth);
   int const mpiTag           = 204;
   if (getIOMPIBlock()->getRank() == 0) {
      int blockBatchDimension = getIOMPIBlock()->getBatchDimension();
      std::vector<char> recvData(transferSize);
      std::vector<ProbeDataBuffer<double>> recvNorms;
      recvNorms.reserve(blockBatchDimension - 1);
      for (int blockBatchIndex = 1; blockBatchIndex < blockBatchDimension; ++blockBatchIndex) {
         int ioRank = getIOMPIBlock()->calcRankFromRowColBatch(0, 0, blockBatchIndex);
         MPI_Recv(
               recvData.data(), transferSize, MPI_BYTE, ioRank, mpiTag, ioComm, MPI_STATUS_IGNORE);
         recvNorms.push_back(ProbeDataBuffer<double>::unpack(recvData));
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
            ProbeData<double> const &normsBatch =
                  bBlock ? recvNorms[bBlock - 1].getData(n) : storedValues.getData(n);
            double energy    = normsBatch.getValue(bLocal);
            double timestamp = normsBatch.getTimestamp();
            printEnergy(returnOutputStream(b), timestamp, globalBatchStart + b, energy);
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
         ProbeData<double> const &normsBatch = storedValues.getData(n);
         double energy                       = normsBatch.getValue(b);
         double timestamp                    = normsBatch.getTimestamp();
         printEnergy(returnOutputStream(b), timestamp, b, energy);
      }
   }
#endif // PV_USE_MPI
}

void ColumnEnergyOutputter::printToLog(ProbeDataBuffer<double> const &storedValues) {
   pvAssert(getProbeOutputFilename() == nullptr or getProbeOutputFilename()[0] == '\0');
   int rank = getCommunicator()->commRank();
   if (rank == 0) {
      int globalBatchStart = calcGlobalBatchOffset();
      int aggregatorSize   = storedValues.size();
      for (int n = 0; n < aggregatorSize; ++n) {
         ProbeData<double> const &normsBatch = storedValues.getData(n);
         int batchSize                       = normsBatch.size();
         for (int b = 0; b < batchSize; ++b) {
            auto printStreamPtr = returnOutputStream(b);
            printEnergy(
                  printStreamPtr,
                  storedValues.getTimestamp(n),
                  globalBatchStart + b,
                  storedValues.getValue(n, b));
         }
      }
   }
}

} // namespace PV
