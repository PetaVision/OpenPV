#include "testPvpRestricted.hpp"
#include "checkpointing/CheckpointEntryPvpBuffer.hpp"
#include "include/PVLayerLoc.h"
#include "utils/PVLog.hpp"
#include "utils/conversions.hpp"
#include <vector>

using namespace PV;

PVLayerLoc initLocPvpRestricted(std::shared_ptr<PV::MPIBlock const> mpiBlock);

void testPvpRestricted(std::shared_ptr<PV::FileManager const> fileManager) {
   PVLayerLoc loc = initLocPvpRestricted(fileManager->getMPIBlock());

   int const localSize = loc.nx * loc.ny * loc.nf;
   std::vector<float> correctData(localSize);
   for (int k = 0; k < localSize; k++) {
      int kxGlobal      = kxPos(k, loc.nx, loc.ny, loc.nf) + loc.kx0;
      int kyGlobal      = kyPos(k, loc.nx, loc.ny, loc.nf) + loc.ky0;
      int kf            = featureIndex(k, loc.nx, loc.ny, loc.nf);
      int kGlobal       = kIndex(kxGlobal, kyGlobal, kf, loc.nxGlobal, loc.nyGlobal, loc.nf);
      correctData.at(k) = (float)kGlobal;
   }

   // Initialize checkpointData as a vector with the same size as correctData.
   // Need to make sure that checkpointData.data() never gets relocated, since the
   // CheckpointEntryPvpBuffer's mDataPointer doesn't change with it.
   std::vector<float> checkpointData(correctData.size());
   CheckpointEntryPvpBuffer<float> checkpointEntryPvp{"checkpointEntryPvpRestricted",
                                                      checkpointData.data(),
                                                      &loc,
                                                      false /*not extended*/};

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
            "testDataPvp failed: data at rank %d, index %d is %f, but should be %f\n",
            fileManager->getMPIBlock()->getGlobalRank(),
            k,
            (double)checkpointData.at(k),
            (double)correctData.at(k));
   }
   MPI_Barrier(fileManager->getMPIBlock()->getComm());
   InfoLog() << "testDataPvpRestricted passed.\n";
}

PVLayerLoc initLocPvpRestricted(std::shared_ptr<PV::MPIBlock const> mpiBlock) {
   PVLayerLoc loc;
   loc.nbatchGlobal = mpiBlock->getBatchDimension();
   loc.nxGlobal     = 16;
   loc.nyGlobal     = 4;
   loc.nf           = 4;
   loc.halo.lt      = 0;
   loc.halo.rt      = 0;
   loc.halo.dn      = 0;
   loc.halo.up      = 0;
   loc.nbatch       = 1;
   loc.kb0          = mpiBlock->getBatchIndex();
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
