#include "testDataNoBroadcast.hpp"
#include "checkpointing/CheckpointEntryData.hpp"
#include "utils/PVLog.hpp"
#include <memory>
#include <vector>

void testDataNoBroadcast(std::shared_ptr<PV::FileManager const> fileManager) {
   int const vectorLength = 32;
   std::vector<float> checkpointData(vectorLength, 0);
   if (fileManager->isRoot()) {
      for (int i = 0; i < vectorLength; i++) {
         checkpointData.at(i) = (float)i;
      }
   }
   PV::CheckpointEntryData<float> checkpointEntryNoBroadcast{"checkpointEntryNoBroadcast",
                                                             checkpointData.data(),
                                                             checkpointData.size(),
                                                             false /*no broadcast*/};
   checkpointEntryNoBroadcast.write(
         fileManager, 0.0 /*simTime, not used*/, false /*not verifying writes*/);

   // Data has now been checkpointed. Copy it to compare after CheckpointEntry::read,
   // and change the vector to make sure that checkpointRead is really modifying the data.
   std::vector<float> dataCopy = checkpointData;
   for (auto &x : checkpointData) {
      x = 5.0f;
   }

   // Then read it back from checkpoint.
   double dummyTime = 0.0; // in checkpointWrite API, but not used by CheckpointEntryData.
   checkpointEntryNoBroadcast.read(fileManager, &dummyTime);

   // The root process should have the original data; any nonroot processes should still have all
   // 5.0.
   if (fileManager->isRoot()) {
      for (int i = 0; i < vectorLength; i++) {
         FatalIf(
               checkpointData.at(i) != dataCopy.at(i),
               "testDataNoBroadcast failed: data at index %d is %f, but should be %f.\n",
               i,
               (double)checkpointData.at(i),
               (double)dataCopy.at(i));
      }
   }
   else {
      for (int i = 0; i < vectorLength; i++) {
         FatalIf(
               checkpointData.at(i) != 5.0f,
               "testDataNoBroadcast failed for rank %d process: data at index %d is %f, but should "
               "be %f.\n",
               fileManager->getMPIBlock()->getGlobalRank(),
               i,
               (double)checkpointData.at(i),
               (double)5.0f);
      }
   }
   InfoLog() << "testDataNoBroadcast passed.\n";
}
