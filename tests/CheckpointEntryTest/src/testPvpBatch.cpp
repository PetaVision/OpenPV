#include "testPvpBatch.hpp"
#include "checkpointing/CheckpointEntryLayerBuffer.hpp"
#include "structures/PVLayerLoc.hpp"
#include "utils/conversions.hpp"
#include <vector>

using namespace PV;

PVLayerLoc initLocPvpBatch(std::shared_ptr<PV::MPIBlock const> mpiBlock);

void testPvpBatch(std::shared_ptr<PV::FileManager const> fileManager) {
   PVLayerLoc loc = initLocPvpBatch(fileManager->getMPIBlock());

   long const localSize = (long)loc.nbatch * (long)loc.nx * (long)loc.ny * (long)loc.nf;
   std::vector<float> correctData(localSize);
   for (long k = 0; k < localSize; k++) {
      int kf           = featureIndex(k, loc.nx, loc.ny, loc.nf);
      int kxGlobal     = kxPos(k, loc.nx, loc.ny, loc.nf) + loc.kx0;
      int kyGlobal     = kyPos(k, loc.nx, loc.ny, loc.nf) + loc.ky0;
      int kbatchGlobal = batchIndex(k, loc.nbatch, loc.nx, loc.ny, loc.nf) + loc.kb0;
      long kGlobal     = kIndexBatch(
            kbatchGlobal,
            kxGlobal,
            kyGlobal,
            kf,
            loc.nbatchGlobal,
            loc.nxGlobal,
            loc.nyGlobal,
            loc.nf);
      correctData.at(k) = (float)kGlobal;
   }

   // Initialize checkpointData as a vector with the same size as correctData.
   // Need to make sure that checkpointData.data() never gets relocated, since the
   // CheckpointEntryLayerBuffer's mDataPointer doesn't change with it.
   std::vector<float> checkpointData(correctData.size());
   CheckpointEntryLayerBuffer<float> checkpointEntryPvp{
         std::string("checkpointEntryPvpBatch"),
         checkpointData.data(),
         &loc,
         false /*broadcastFlag*/,
         false /*extendedFlag*/};

   double const simTime = 10.0;
   // Copy correct data into checkpoint data.
   for (int k = 0; k < localSize; k++) {
      checkpointData.at(k) = correctData.at(k);
   }
   checkpointEntryPvp.write(fileManager, simTime, false /*not verifying writes*/);

   // Data has now been checkpointed. Change the vector to make sure that checkpointRead is really
   // modifying the data.
   for (auto &a : checkpointData) {
      a = -1.0f;
   }

   // Read the data back
   double readTime = (double)(simTime == 0);
   pvAssert(simTime != readTime);
   checkpointEntryPvp.read(fileManager, &readTime);

   // Verify the read
   FatalIf(readTime != simTime, "Read timestamp %f; expected %f.\n", readTime, simTime);
   for (int k = 0; k < localSize; k++) {
      FatalIf(
            checkpointData.at(k) != correctData.at(k),
            "testPvpBatch failed: data at rank %d, index %d is %f, but should be %f\n",
            fileManager->getMPIBlock()->getGlobalRank(),
            k,
            (double)checkpointData.at(k),
            (double)correctData.at(k));
   }
   MPI_Barrier(fileManager->getMPIBlock()->getComm());
   InfoLog() << "testPvpBatch passed.\n";
}

PVLayerLoc initLocPvpBatch(std::shared_ptr<PV::MPIBlock const> mpiBlock) {
   PVLayerLoc loc;
   loc.nbatchGlobal = 4;
   loc.nxGlobal     = 16;
   loc.nyGlobal     = 4;
   loc.nf           = 1;
   loc.halo.lt      = 0;
   loc.halo.rt      = 0;
   loc.halo.dn      = 0;
   loc.halo.up      = 0;
   FatalIf(
         loc.nbatchGlobal % mpiBlock->getBatchDimension(),
         "Global batch size %d is not a multiple of batch width %d\n",
         loc.nbatchGlobal,
         mpiBlock->getBatchDimension());
   loc.nbatch = loc.nbatchGlobal / mpiBlock->getBatchDimension();
   loc.kb0    = loc.nbatchGlobal * mpiBlock->getBatchIndex();
   FatalIf(
         loc.nxGlobal % mpiBlock->getNumColumns(),
         "Global width %d is not a multiple of the number of MPI columns %d\n",
         loc.nxGlobal,
         mpiBlock->getNumColumns());
   loc.nx  = loc.nxGlobal / mpiBlock->getNumColumns();
   loc.kx0 = loc.nx * mpiBlock->getColumnIndex();
   FatalIf(
         loc.nyGlobal % mpiBlock->getNumRows(),
         "Global height %d is not a multiple of the number of MPI rows %d\n",
         loc.nyGlobal,
         mpiBlock->getNumRows());
   loc.ny  = loc.nyGlobal / mpiBlock->getNumRows();
   loc.ky0 = loc.ny * mpiBlock->getRowIndex();
   return loc;
}
