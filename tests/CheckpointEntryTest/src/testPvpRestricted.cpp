#include "testPvpRestricted.hpp"
#include "io/CheckpointEntry.hpp"
#include "utils/conversions.h"

void testPvpRestricted(PV::Communicator * comm, std::string const& directory) {
   PVLayerLoc loc;
   loc.nbatchGlobal = comm->numCommBatches();
   loc.nxGlobal = 16;
   loc.nyGlobal = 4;
   loc.nf = 4;
   loc.halo.lt = 0;
   loc.halo.rt = 0;
   loc.halo.dn = 0;
   loc.halo.up = 0;
   loc.nbatch = 1;
   loc.kb0 = comm->commBatch();
   pvErrorIf(loc.nxGlobal % comm->numCommColumns(), "Global width %d is not a multiple of the number of MPI columns %d\n", loc.nxGlobal, comm->numCommColumns());
   loc.nx = loc.nxGlobal / comm->numCommColumns();
   loc.kx0 = loc.nx * comm->commColumn();
   pvErrorIf(loc.nyGlobal % comm->numCommRows(), "Global height %d is not a multiple of the number of MPI rows %d\n", loc.nyGlobal, comm->numCommRows());
   loc.ny = loc.nyGlobal / comm->numCommRows();
   loc.ky0 = loc.ny * comm->commRow();

   int const localSize = loc.nx * loc.ny * loc.nf;
   std::vector<float> correctData(localSize);
   for (int k=0; k<localSize; k++) {
      int kxGlobal = kxPos(k, loc.nx, loc.ny, loc.nf) + loc.kx0;
      int kyGlobal = kyPos(k, loc.nx, loc.ny, loc.nf) + loc.ky0;
      int kf = featureIndex(k, loc.nx, loc.ny, loc.nf);
      int kGlobal = kIndex(kxGlobal, kyGlobal, kf, loc.nxGlobal, loc.nyGlobal, loc.nf);
      correctData.at(k) = (float) kGlobal;
   }

   // Initialize checkpointData as a vector with the same size as correctData.
   // Need to make sure that checkpointData.data() never gets relocated, since the CheckpointEntryPvp's mDataPointer doesn't change with it.
   std::vector<float> checkpointData(correctData.size());
   PV::CheckpointEntryPvp<float> checkpointEntryPvp{"checkpointEntryPvpRestricted", false/*not verifying writes*/, comm,
         checkpointData.data(), checkpointData.size(), PV_FLOAT_TYPE, &loc, false/*not extended*/};

   double const simTime = 10.0;
   // Copy correct data into checkpoint data.
   for (int k=0; k<localSize; k++) {
      checkpointData.at(k) = correctData.at(k);
   }
   checkpointEntryPvp.write(directory, simTime);

   // Data has now been checkpointed. Change the vector to make sure that checkpointRead is really modifying the data.
   for (auto& a : checkpointData) {
      a = -1.0f;
   }

   // Read the data back
   double readTime = (double) (simTime==0);
   pvAssert(simTime != readTime);
   checkpointEntryPvp.read(directory, &readTime);

   // Verify the read
   pvErrorIf(readTime != simTime, "Read timestamp %f; expected %f.\n", readTime, simTime);
   for (int k=0; k<localSize; k++) {
      pvErrorIf(checkpointData.at(k)!=correctData.at(k), "testDataPvp failed: data at rank %d, index %d is %f, but should be %f\n",
            comm->commRank(), k, (double) checkpointData.at(k), (double) correctData.at(k));
   }
   MPI_Barrier(comm->communicator());
   pvInfo() << "testDataPvpRestricted passed.\n";
}