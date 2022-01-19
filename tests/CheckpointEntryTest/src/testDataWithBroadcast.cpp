#include "testDataWithBroadcast.hpp"
#include "checkpointing/CheckpointEntryData.hpp"
#include "utils/PVLog.hpp"
#include <vector>

void testDataWithBroadcast(std::shared_ptr<PV::FileManager const> fileManager) {
// void testDataWithBroadcast(std::shared_ptr<PV::MPIBlock const> mpiBlock, std::string const &directory)
   int const vectorLength = 32;
   std::vector<float> correctData(vectorLength, 0);
   for (int i = 0; i < vectorLength; i++) {
      correctData.at(i) = (float)i;
   }
   std::vector<float> checkpointData;
   if (fileManager->isRoot()) {
      checkpointData = correctData;
   }
   else {
      checkpointData = std::vector<float>(vectorLength, 0);
   }
   FatalIf(
         (int)checkpointData.size() != vectorLength,
         "checkpointData has length %zu instead of %d\n",
         (size_t)checkpointData.size(),
         vectorLength);
   PV::CheckpointEntryData<float> checkpointEntryWithBroadcast{
         "checkpointEntryWithBroadcast",
         checkpointData.data(),
         checkpointData.size(),
         true /*broadcasting read to all processes*/};
   checkpointEntryWithBroadcast.write(
         fileManager, 0.0 /*simTime, not used*/, false /*not verifying writes*/);

   // Data has now been checkpointed. Copy it to compare after CheckpointEntry::read,
   // and change the vector to make sure that checkpointRead is really modifying the data.
   std::vector<float> dataCopy = checkpointData;
   for (auto &x : checkpointData) {
      x = 5.0f;
   }

   // Then read it back from checkpoint.
   double dummyTime = 0.0; // in checkpointWrite API, but not used by CheckpointEntryData.
   checkpointEntryWithBroadcast.read(fileManager, &dummyTime);

   // All processes should have the original data
   for (int i = 0; i < vectorLength; i++) {
      FatalIf(
            checkpointData.at(i) != correctData.at(i),
            "testDataWithBroadcast failed: data at index %d is %f, but should be %f.\n",
            i,
            (double)checkpointData.at(i),
            (double)correctData.at(i));
   }
   MPI_Barrier(fileManager->getMPIBlock()->getComm());
   InfoLog() << "testDataWithBroadcast passed.\n";
}
