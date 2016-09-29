#include "testPvpExtended.hpp"
#include "io/CheckpointEntry.hpp"
#include "utils/conversions.h"
#include "utils/PVLog.hpp"

void testPvpExtended(PV::Communicator * comm, std::string const& directory) {
   PVLayerLoc loc;
   loc.nbatchGlobal = comm->numCommBatches();
   loc.nxGlobal = 16;
   loc.nyGlobal = 4;
   loc.nf = 4;
   loc.halo.lt = 2;
   loc.halo.rt = 2;
   loc.halo.dn = 2;
   loc.halo.up = 2;
   loc.nbatch = 1;
   loc.kb0 = comm->commBatch();
   pvErrorIf(loc.nxGlobal % comm->numCommColumns(), "Global width %d is not a multiple of the number of MPI columns %d\n", loc.nxGlobal, comm->numCommColumns());
   loc.nx = loc.nxGlobal / comm->numCommColumns();
   loc.kx0 = loc.nx * comm->commColumn();
   pvErrorIf(loc.nyGlobal % comm->numCommRows(), "Global height %d is not a multiple of the number of MPI rows %d\n", loc.nyGlobal, comm->numCommRows());
   loc.ny = loc.nyGlobal / comm->numCommRows();
   loc.ky0 = loc.ny * comm->commRow();

   int const nxLocalExt = loc.nx + loc.halo.lt + loc.halo.rt;
   int const nyLocalExt = loc.ny + loc.halo.dn + loc.halo.up;
   int const nxGlobalExt = loc.nxGlobal + loc.halo.lt + loc.halo.rt;
   int const nyGlobalExt = loc.nyGlobal + loc.halo.dn + loc.halo.up;
   int const localExtendedSize = nxLocalExt * nyLocalExt * loc.nf;
   std::vector<float> correctData(localExtendedSize);
   for (int k=0; k<localExtendedSize; k++) {
      int kxGlobalExt = kxPos(k, nxLocalExt, nyLocalExt, loc.nf) + loc.kx0;
      int kyGlobalExt = kyPos(k, nxLocalExt, nyLocalExt, loc.nf) + loc.ky0;
      int kf = featureIndex(k, nxLocalExt, nyLocalExt, loc.nf);
      int kGlobal = kIndex(kxGlobalExt, kyGlobalExt, kf, nxGlobalExt, nyGlobalExt, loc.nf);
      correctData.at(k) = (float) kGlobal;
   }

   // Initialize checkpointData as a vector with the same size as correctData.
   // Need to make sure that checkpointData.data() never gets relocated, since the CheckpointEntryPvp's mDataPointer doesn't change with it.
   std::vector<float> checkpointData(correctData.size());
   PV::CheckpointEntryPvp<float> checkpointEntryPvp{"checkpointEntryPvpExtended", comm,
         checkpointData.data(), checkpointData.size(), PV_FLOAT_TYPE, &loc, true/*extended*/};

   double const simTime = 10.0;
   // Copy correct data into checkpoint data.
   for (int k=0; k<localExtendedSize; k++) {
      checkpointData.at(k) = correctData.at(k);
   }
   checkpointEntryPvp.write(directory, simTime, false/*not verifying writes*/);

   // Data has now been checkpointed. Change the vector to make sure that checkpointRead is really modifying the data.
   // Note that we're changing the border region as well as the restricted region, even though the border region doesn't get saved.
   for (auto& a : checkpointData) {
      a = -1.0f;
   }

   // Read the data back
   double readTime = (double) (simTime==0);
   pvAssert(simTime != readTime);
   checkpointEntryPvp.read(directory, &readTime);

   // Verify the read, noting that checkpointWrite only saves the restricted portion and checkpointRead does not modify the extended portion.
   pvErrorIf(readTime != simTime, "Read timestamp %f; expected %f.\n", readTime, simTime);
   for (int k=0; k<localExtendedSize; k++) {
      int kxGlobalExt = kxPos(k, nxLocalExt, nyLocalExt, loc.nf) + loc.kx0;
      int kyGlobalExt = kyPos(k, nxLocalExt, nyLocalExt, loc.nf) + loc.ky0;
      bool inBorder = kxGlobalExt < loc.halo.lt || kxGlobalExt >= loc.nxGlobal + loc.halo.lt || kyGlobalExt < loc.halo.up || kyGlobalExt >= loc.nyGlobal + loc.halo.up;
      float correctValue = inBorder ? -1.0f : correctData.at(k);
      pvErrorIf(checkpointData.at(k)!=correctValue, "testDataPvp failed: data at rank %d, index %d is %f, but should be %f\n",
            comm->commRank(), k, (double) checkpointData.at(k), (double) correctValue);
   }
   MPI_Barrier(comm->communicator());
   pvInfo() << "testDataPvpRestricted passed.\n";
}
