#include "testDataNoBroadcast.hpp"
#include "checkpointing/CheckpointEntryData.hpp"
#include "utils/PVLog.hpp"

void testDataNoBroadcast(PV::Communicator *comm, std::string const &directory) {
   int const vectorLength = 32;
   std::vector<float> checkpointData(vectorLength, 0);
   int const rank = comm->commRank();
   if (rank == 0) {
      for (int i = 0; i < vectorLength; i++) {
         checkpointData.at(i) = (float)i;
      }
   }
   PV::CheckpointEntryData<float> checkpointEntryNoBroadcast{"checkpointEntryNoBroadcast",
                                                             comm,
                                                             checkpointData.data(),
                                                             checkpointData.size(),
                                                             false /*no broadcast*/};
   checkpointEntryNoBroadcast.write(
         directory, 0.0 /*simTime, not used*/, false /*not verifying writes*/);

   // Data has now been checkpointed. Copy it to compare after CheckpointEntry::read,
   // and change the vector to make sure that checkpointRead is really modifying the data.
   std::vector<float> dataCopy = checkpointData;
   for (auto &x : checkpointData) {
      x = 5.0f;
   }

   // Then read it back from checkpoint.
   double dummyTime = 0.0; // in checkpointWrite API, but not used by CheckpointEntryData.
   checkpointEntryNoBroadcast.read(directory, &dummyTime);

   // The root process should have the original data; any nonroot processes should still have all
   // 5.0.
   if (rank == 0) {
      for (int i = 0; i < vectorLength; i++) {
         pvErrorIf(
               checkpointData.at(i) != dataCopy.at(i),
               "testDataNoBroadcast failed: data at index %d is %f, but should be %f.\n",
               i,
               (double)checkpointData.at(i),
               (double)dataCopy.at(i));
      }
   }
   else {
      for (int i = 0; i < vectorLength; i++) {
         pvErrorIf(
               checkpointData.at(i) != 5.0f,
               "testDataNoBroadcast failed for rank %d process: data at index %d is %f, but should "
               "be %f.\n",
               rank,
               i,
               (double)checkpointData.at(i),
               (double)5.0f);
      }
   }
   pvInfo() << "testDataNoBroadcast passed.\n";
}
