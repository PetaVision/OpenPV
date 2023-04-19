#include "cMakeHeader.h"

#include "ResetStateOnTriggerTestProbeOutputter.hpp"

#include "arch/mpi/mpi.h"
#include "probes/ProbeData.hpp"
#include "utils/PVAssert.hpp"
#include <vector>

ResetStateOnTriggerTestProbeOutputter::ResetStateOnTriggerTestProbeOutputter(
      char const *objName,
      PV::PVParams *params,
      PV::Communicator const *comm) {
   initialize(objName, params, comm);
}

void ResetStateOnTriggerTestProbeOutputter::initialize(
      char const *objName,
      PV::PVParams *params,
      PV::Communicator const *comm) {
   BaseProbeOutputter::initialize(objName, params, comm);
}

void ResetStateOnTriggerTestProbeOutputter::printGlobalStatsBuffer(
      PV::ProbeData<int> const &globalDiscrepancies) {
   pvAssert(getProbeOutputFilename() != nullptr and getProbeOutputFilename()[0] != '\0');
#ifdef PV_USE_MPI
   if (getCommunicator()->commRank() != 0) {
      return;
   }
   MPI_Comm ioComm = getIOMPIBlock()->getComm();
   int rootProc    = 0;

   int localNBatch = static_cast<int>(globalDiscrepancies.size());

   int const mpiTag = 202;
   if (getIOMPIBlock()->getRank() == rootProc) {
      int blockBatchDimension = getIOMPIBlock()->getBatchDimension();
      std::vector<std::vector<int>> recvDataBatch;
      recvDataBatch.resize(blockBatchDimension - 1);
      for (auto &r : recvDataBatch) {
         r.resize(localNBatch);
      }
      for (int blockBatchIndex = 1; blockBatchIndex < blockBatchDimension; ++blockBatchIndex) {
         int ioRank     = getIOMPIBlock()->calcRankFromRowColBatch(0, 0, blockBatchIndex);
         auto &recvData = recvDataBatch[blockBatchIndex - 1];
         MPI_Recv(recvData.data(), localNBatch, MPI_INT, ioRank, mpiTag, ioComm, MPI_STATUS_IGNORE);
      }
      int globalBatchStart = calcGlobalBatchOffset();

      int numBatchElementsInBlock = getLocalNBatch() * blockBatchDimension;
      for (int b = 0; b < numBatchElementsInBlock; ++b) {
         int bLocal = b % getLocalNBatch();
         int bBlock = (b - bLocal) / getLocalNBatch();
         std::vector<int> const &numDiscrepsVector =
               bBlock ? recvDataBatch[bBlock - 1] : globalDiscrepancies.getValues();
         auto &numDiscrepancies = numDiscrepsVector[bLocal];
         double timestamp       = globalDiscrepancies.getTimestamp();
         printDiscrepancies(
               returnOutputStream(b), numDiscrepancies, timestamp, globalBatchStart + b);
      }
   }
   else {
      // this process has the discrepancies for some batch element(s), but is not a process
      // that does I/O. Therefore, we need to send the discrepancy count over MPI.
      auto const &discrepanciesVector = globalDiscrepancies.getValues();
      MPI_Send(discrepanciesVector.data(), localNBatch, MPI_INT, rootProc, mpiTag, ioComm);
   }
#else // PV_USE_MPI
   for (int b = 0; b < getLocalNBatch(); ++b) {
      auto &numDiscrepancies = globalDiscrepancies.getValue(b);
      double timestamp       = globalDiscrepancies.getTimestamp();
      printDiscrepancies(returnOutputStream(b), numDiscrepancies, timestamp, b);
   }
#endif // PV_USE_MPI

#ifdef ATIK
   if (getCommunicator()->commRank() != 0) {
      return;
   }
   if (mProbeStatus != PV_SUCCESS) {
      pvAssert((std::size_t)nBatch == mOutputStreams.size());
      int globalBatchSize = nBatch * getCommunicator()->getIOMPIBlock()->getGlobalBatchDimension();
      for (int localBatchIndex = 0; localBatchIndex < nBatch; localBatchIndex++) {
         int nnz = (int)nearbyint(numDiscreps[localBatchIndex]);
         if (globalBatchSize == 1) {
            pvAssert(localBatchIndex == 0);
            output(localBatchIndex)
                  .printf(
                        "%s: t=%f, %d neuron%s the wrong value.\n",
                        getDescription_c(),
                        simTime,
                        nnz,
                        nnz == 1 ? " has" : "s have");
         }
         else {
            output(localBatchIndex)
                  .printf(
                        "%s: t=%f, batch element %d, %d neuron%s the wrong value.\n",
                        getDescription_c(),
                        simTime,
                        localBatchIndex,
                        nnz,
                        nnz == 1 ? " has" : "s have");
         }
      }
   }
#endif // ATIK
}

void ResetStateOnTriggerTestProbeOutputter::printDiscrepancies(
      std::shared_ptr<PV::PrintStream> printStreamPtr,
      int numDiscrepancies,
      double timestamp,
      int batchIndex) {
   if (numDiscrepancies) {
      printStreamPtr->printf(
            "Probe %s: t=%f, batch element %d, %d neuron%s the wrong value.\n",
            getName_c(),
            timestamp,
            batchIndex,
            numDiscrepancies,
            numDiscrepancies == 1 ? " has" : "s have");
      if (!mDiscrepanciesFound) {
         mDiscrepanciesFound = true;
         mFirstFailureTime   = timestamp;
      }
   }
}
