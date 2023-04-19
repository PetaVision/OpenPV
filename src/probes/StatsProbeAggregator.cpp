#include "cMakeHeader.h"

#include "StatsProbeAggregator.hpp"

#include "arch/mpi/mpi.h"
#include "probes/ProbeData.hpp"
#include "probes/StatsProbeTypes.hpp"
#include "utils/PVAssert.hpp"
#include "utils/PVLog.hpp"

#include <algorithm>
#include <vector>

namespace PV {

StatsProbeAggregator::StatsProbeAggregator(
      char const *objName,
      PVParams *params,
      std::shared_ptr<MPIBlock const> mpiBlock) {
   initialize(objName, params, mpiBlock);
}

void StatsProbeAggregator::aggregateStoredValues(ProbeDataBuffer<LayerStats> const &partialStore) {
#ifdef PV_USE_MPI
   auto numTimestamps = static_cast<unsigned int>(partialStore.size());
   auto batchWidth    = partialStore.getBatchWidth();
   auto packedSizeU   = ProbeDataBuffer<LayerStats>::calcPackedSize(numTimestamps, batchWidth);
   int packedSize     = static_cast<int>(packedSizeU);

   unsigned int oldStoreSize = static_cast<unsigned int>(mStoredValues.size());
   for (unsigned int k = 0; k < numTimestamps; ++k) {
      mStoredValues.store(partialStore.getData(k));
   }

   MPI_Comm comm = mMPIBlock->getComm();
   int rank      = mMPIBlock->getRank();
   int rootProc  = 0;
   if (rank == rootProc) {
      std::vector<char> receivedPacked(packedSize);
      int numProcs = mMPIBlock->getSize();
      for (int r = 0; r < numProcs; ++r) {
         if (r == rootProc) {
            continue;
         }
         char *receivedPtr = receivedPacked.data();
         MPI_Recv(receivedPtr, packedSize, MPI_CHAR, r, 201 /*tag*/, comm, MPI_STATUS_IGNORE);
         auto receivedData = ProbeDataBuffer<LayerStats>::unpack(receivedPacked);
         FatalIf(
               receivedData.size() != numTimestamps,
               "Differing buffer sizes on ranks %d and %d (%u versus %u)\n",
               rootProc,
               r,
               (unsigned int)numTimestamps,
               (unsigned int)partialStore.size());
         for (unsigned int k = 0; k < static_cast<unsigned int>(numTimestamps); ++k) {
            ProbeData<LayerStats> &aggregate     = mStoredValues.getData(oldStoreSize + k);
            ProbeData<LayerStats> const &partial = receivedData.getData(k);
            double timestamp                     = aggregate.getTimestamp();
            FatalIf(
                  partial.getTimestamp() != timestamp,
                  "Differing timestamps on ranks %d and %d (%f versus %f)\n",
                  rootProc,
                  r,
                  aggregate.getTimestamp(),
                  partial.getTimestamp());
            FatalIf(
                  partial.size() != aggregate.size(),
                  "Differing batch sizes on ranks %d and %d at time %f (%u versus %u)\n",
                  rootProc,
                  r,
                  timestamp,
                  (unsigned int)aggregate.size(),
                  (unsigned int)partial.size());

            int nbatch = static_cast<int>(partial.size());
            pvAssert(nbatch == static_cast<int>(aggregate.size()));
            for (int b = 0; b < nbatch; ++b) {
               LayerStats const &recvdElement = partial.getValue(b);
               LayerStats &aggregateElement   = aggregate.getValue(b);
               aggregateElement.mSum += recvdElement.mSum;
               aggregateElement.mSumSquared += recvdElement.mSumSquared;
               aggregateElement.mMin = std::min(aggregateElement.mMin, recvdElement.mMin);
               aggregateElement.mMax = std::max(aggregateElement.mMax, recvdElement.mMax);
               aggregateElement.mNumNeurons += recvdElement.mNumNeurons;
               aggregateElement.mNumNonzero += recvdElement.mNumNonzero;
            }
         }
      }
   }
   else {
      auto packedPartialStore = partialStore.pack();
      MPI_Send(packedPartialStore.data(), packedSize, MPI_CHAR, rootProc, 201 /*tag*/, comm);
   }
   broadcastStoredValues();
#else // PV_USE_MPI
   // Push each value of partialStore onto mStoredValues
   int storeSize = static_cast<int>(partialStore.size());
   for (int n = 0; n < storeSize; ++n) {
      ProbeData<LayerStats> const &partialStats = partialStore.getData(n);
      mStoredValues.store(partialStats);
   }
#endif // PV_USE_MPI
}

void StatsProbeAggregator::broadcastStoredValues() {
   MPI_Comm comm           = mMPIBlock->getComm();
   int rank                = mMPIBlock->getRank();
   int rootProc            = 0;
   unsigned int packedSize = ProbeDataBuffer<LayerStats>::calcPackedSize(
         mStoredValues.size(), mStoredValues.getBatchWidth());
   std::vector<char> packedStore;
   if (rank == rootProc) {
      packedStore = mStoredValues.pack();
      MPI_Bcast(packedStore.data(), (int)packedSize, MPI_CHAR, rootProc, comm);
   }
   else {
      packedStore.resize(packedSize);
      MPI_Bcast(packedStore.data(), (int)packedSize, MPI_CHAR, rootProc, comm);
      mStoredValues = ProbeDataBuffer<LayerStats>::unpack(packedStore);
   }
}

void StatsProbeAggregator::clearStoredValues() { mStoredValues.clear(); }

void StatsProbeAggregator::initialize(
      char const *objName,
      PVParams *params,
      std::shared_ptr<MPIBlock const> mpiBlock) {
   ProbeComponent::initialize(objName, params);
   mMPIBlock = mpiBlock;
}

void StatsProbeAggregator::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {}

} // namespace PV
