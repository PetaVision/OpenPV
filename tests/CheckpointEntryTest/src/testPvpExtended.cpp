#include "testPvpExtended.hpp"
#include "checkpointing/CheckpointEntryPvpBuffer.hpp"
#include "include/PVLayerLoc.hpp"
#include "utils/PVLog.hpp"
#include "utils/conversions.hpp"
#include <vector>

using namespace PV;

PVLayerLoc initLocPvpExtended(std::shared_ptr<PV::MPIBlock const> mpiBlock);

void testPvpExtended(std::shared_ptr<PV::FileManager const> fileManager) {
   PVLayerLoc loc = initLocPvpExtended(fileManager->getMPIBlock());

   int const nxLocalExt        = loc.nx + loc.halo.lt + loc.halo.rt;
   int const nyLocalExt        = loc.ny + loc.halo.dn + loc.halo.up;
   int const nxGlobalExt       = loc.nxGlobal + loc.halo.lt + loc.halo.rt;
   int const nyGlobalExt       = loc.nyGlobal + loc.halo.dn + loc.halo.up;
   int const localExtendedSize = nxLocalExt * nyLocalExt * loc.nf;
   std::vector<float> correctData(localExtendedSize);
   for (int k = 0; k < localExtendedSize; k++) {
      int kxGlobalExt   = kxPos(k, nxLocalExt, nyLocalExt, loc.nf) + loc.kx0;
      int kyGlobalExt   = kyPos(k, nxLocalExt, nyLocalExt, loc.nf) + loc.ky0;
      int kf            = featureIndex(k, nxLocalExt, nyLocalExt, loc.nf);
      int kGlobal       = kIndex(kxGlobalExt, kyGlobalExt, kf, nxGlobalExt, nyGlobalExt, loc.nf);
      correctData.at(k) = (float)kGlobal;
   }

   // Initialize checkpointData as a vector with the same size as correctData.
   // Need to make sure that checkpointData.data() never gets relocated, since the
   // CheckpointEntryPvpBuffer's mDataPointer doesn't change with it.
   std::vector<float> checkpointData(correctData.size());
   CheckpointEntryPvpBuffer<float> checkpointEntryPvp{
         "checkpointEntryPvpExtended", checkpointData.data(), &loc, true /*extended*/};

   double const simTime = 10.0;
   // Copy correct data into checkpoint data.
   for (int k = 0; k < localExtendedSize; k++) {
      checkpointData.at(k) = correctData.at(k);
   }
   checkpointEntryPvp.write(fileManager, simTime, false /*not verifying writes*/);

   // Data has now been checkpointed. Change the vector to make sure that checkpointRead is really
   // modifying the data.
   // Note that we're changing the border region as well as the restricted region, even though the
   // border region doesn't get saved.
   for (auto &a : checkpointData) {
      a = -1.0f;
   }

   // Read the data back
   double readTime = (double)(simTime == 0);
   pvAssert(simTime != readTime);
   checkpointEntryPvp.read(fileManager, &readTime);

   // Verify the read, noting that checkpointWrite only saves the restricted portion and
   // checkpointRead sets the the extended portion to zero.
   FatalIf(readTime != simTime, "Read timestamp %f; expected %f.\n", readTime, simTime);
   for (int k = 0; k < localExtendedSize; k++) {
      int kxGlobalExt = kxPos(k, nxLocalExt, nyLocalExt, loc.nf) + loc.kx0;
      int kyGlobalExt = kyPos(k, nxLocalExt, nyLocalExt, loc.nf) + loc.ky0;
      bool inBorder   = kxGlobalExt < loc.halo.lt || kxGlobalExt >= loc.nxGlobal + loc.halo.lt
                      || kyGlobalExt < loc.halo.up || kyGlobalExt >= loc.nyGlobal + loc.halo.up;
      float correctValue = inBorder ? 0.0f : correctData.at(k);
      FatalIf(
            checkpointData.at(k) != correctValue,
            "testDataPvp failed: data at rank %d, index %d is %f, but should be %f\n",
            fileManager->getMPIBlock()->getGlobalRank(),
            k,
            (double)checkpointData.at(k),
            (double)correctValue);
   }
   MPI_Barrier(fileManager->getMPIBlock()->getComm());
   InfoLog() << "testDataPvpExtended passed.\n";
}

PVLayerLoc initLocPvpExtended(std::shared_ptr<PV::MPIBlock const> mpiBlock) {
   PVLayerLoc loc;
   loc.nbatchGlobal = mpiBlock->getBatchDimension();
   loc.nxGlobal     = 16;
   loc.nyGlobal     = 4;
   loc.nf           = 4;
   loc.halo.lt      = 2;
   loc.halo.rt      = 2;
   loc.halo.dn      = 2;
   loc.halo.up      = 2;
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
