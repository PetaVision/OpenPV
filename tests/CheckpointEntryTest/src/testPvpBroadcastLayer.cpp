#include "testPvpBroadcastLayer.hpp"
#include "checkpointing/CheckpointEntryLayerBuffer.hpp"
#include "include/PVLayerLoc.hpp"
#include "utils/PVLog.hpp"
#include "utils/conversions.hpp"
#include <algorithm> // std::copy
#include <vector>

using namespace PV;

PVLayerLoc initLocPvpBroadcastLayer(std::shared_ptr<PV::MPIBlock const> mpiBlock);

void testPvpBroadcastLayer(std::shared_ptr<PV::FileManager const> fileManager) {
   auto mpiBlock = fileManager->getMPIBlock();
   PVLayerLoc loc = initLocPvpBroadcastLayer(mpiBlock);

   std::vector<float> correctData(loc.nf * loc.nbatch);
   for (int b = 0; b < loc.nbatch; ++b) {
      for (int f = 0; f < loc.nf; ++f) {
         int globalBatchIndex = b + loc.nbatch * mpiBlock->getBatchIndex();
         int k                = globalBatchIndex * loc.nf + f;
         correctData.at(k)    = static_cast<float>(k);
      }
   }

   // Initialize checkpointData as a vector with the same size as correctData.
   // Need to make sure that checkpointData.data() never gets relocated, since the
   // CheckpointEntryLayerBuffer's mDataPointer doesn't change with it.
   std::vector<float> checkpointData(correctData.size());
   CheckpointEntryLayerBuffer<float> checkpointEntryPvp{
         "checkpointEntryPvpBroadcastLayer",
         checkpointData.data(),
         &loc,
         true /*broadcastFlag*/,
         false /*extendedFlag*/};

   double const simTime = 10.0;
   // Copy correct data into checkpoint data.
   std::copy(correctData.begin(), correctData.end(), checkpointData.begin());
   checkpointEntryPvp.write(fileManager, simTime, false /*not verifying writes*/);

   // Data has now been checkpointed. Change the vector to make sure that checkpointRead is really
   // modifying the data.
   for (auto &a : checkpointData) {
      a = -1.0f;
   }

   // Read the data back
   double readTime = (double)(simTime == 0.0);
   pvAssert(simTime != readTime);
   checkpointEntryPvp.read(fileManager, &readTime);

   // Verify the read
   FatalIf(readTime != simTime, "Read timestamp %f; expected %f.\n", readTime, simTime);
   for (int k = 0; k < loc.nf * loc.nbatch; k++) {
      FatalIf(
            checkpointData.at(k) != correctData.at(k),
            "testPvpBroadcastLayer failed: data at rank %d, index %d is %f, but should be %f\n",
            fileManager->getMPIBlock()->getGlobalRank(),
            k,
            (double)checkpointData.at(k),
            (double)correctData.at(k));
   }
   MPI_Barrier(fileManager->getMPIBlock()->getComm());
   InfoLog() << "testDataPvpBroadcastLayer passed.\n";
}

PVLayerLoc initLocPvpBroadcastLayer(std::shared_ptr<PV::MPIBlock const> mpiBlock) {
   PVLayerLoc loc;
   loc.nbatch       = 1;
   loc.nx           = 1;
   loc.ny           = 1;
   loc.nf           = 16;
   loc.nbatchGlobal = mpiBlock->getBatchDimension() * loc.nbatch;
   loc.nxGlobal     = 1;
   loc.nyGlobal     = 1;
   loc.kb0          = mpiBlock->getBatchIndex();
   loc.kx0          = 0;
   loc.ky0          = 0;
   loc.halo.lt      = 0;
   loc.halo.rt      = 0;
   loc.halo.dn      = 0;
   loc.halo.up      = 0;
   return loc;
}
