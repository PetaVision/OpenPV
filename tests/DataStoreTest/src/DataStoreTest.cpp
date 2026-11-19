/*
 * DataStoreTest.cpp
 *
 */

#include <structures/DataStore.hpp>
#include <utils/PVLog.hpp>

const int NUM_BUFFERS = 2;
const int NUM_LEVELS  = 4;
const int NUM_ITEMS   = 10;

float correctData(int bufferIndex, int itemIndex, int levelIndex);
double correctTime(int levelIndex);

int main(int argc, char *argv[]) {
   PV::DataStore store(
         NUM_BUFFERS,
         NUM_LEVELS,
         NUM_ITEMS /*width*/,
         1 /*height*/,
         1 /*features*/,
         false /*sparseFlag*/);

   // Many DataStore methods are overloaded with versions with and without the
   // level index argument. Verify that they are consistent.
   FatalIf(
         store.getBuffer(0) != store.getBuffer(0, 0),
         "getBuffer(0,0) should be equal to getBuffer(0) but is not.\n");
   FatalIf(
         store.getBuffer(1) != store.getBuffer(1, 0),
         "getBuffer(1,0) should be equal to getBuffer(1) but is not.\n");

   // Initialize level 0.
   for (int itemIndex = 0; itemIndex < NUM_ITEMS; itemIndex++) {
      store.getBuffer(0)[itemIndex] = correctData(0, itemIndex, 0 /*levelIndex*/);
      store.getBuffer(1)[itemIndex] = correctData(1, itemIndex, 0 /*levelIndex*/);
   }
   store.setLastUpdateTime(correctTime(0));

   // Initialize the remaining levels, using the levelIndex argument
   for (int levelIndex = 1; levelIndex < NUM_LEVELS; levelIndex++) {
      for (int itemIndex = 0; itemIndex < NUM_ITEMS; itemIndex++) {
         store.getBuffer(0, levelIndex)[itemIndex] = correctData(0, itemIndex, levelIndex);
         store.getBuffer(1, levelIndex)[itemIndex] = correctData(1, itemIndex, levelIndex);
      }
      store.setLastUpdateTime(levelIndex, correctTime(levelIndex));
   }

   // Verify that the values are correct, using nextLevelIndex to advance the levels.
   // We go to two times the number of levels to verify wrap-around.
   for (int levelIndex = 0; levelIndex < 2 * NUM_LEVELS; levelIndex++) {
      // newLevelIndex decrements mCurrentLevel
      int correctLevel = (2 * NUM_LEVELS - levelIndex) % NUM_LEVELS;
      for (int itemIndex = 0; itemIndex < NUM_ITEMS; itemIndex++) {
         FatalIf(
               store.getBuffer(0)[itemIndex] != correctData(0, itemIndex, correctLevel),
               "Verification failed for buffer 0, item %d after %d rotations.\n",
               itemIndex,
               levelIndex);
         FatalIf(
               store.getBuffer(1)[itemIndex] != correctData(1, itemIndex, correctLevel),
               "Verification failed for buffer 1, item %d after %d rotations.\n",
               itemIndex,
               levelIndex);
      }
      FatalIf(
            store.getLastUpdateTime() != correctTime(correctLevel),
            "Verification failed for buffer 0, LastUpdateTime after %d rotations.\n",
            levelIndex);
      store.newLevelIndex(); // Rotate.
   }

   return EXIT_SUCCESS;
}

float correctData(int bufferIndex, int itemIndex, int levelIndex) {
   return (float)(bufferIndex + NUM_BUFFERS * (itemIndex + levelIndex * NUM_ITEMS));
}

double correctTime(int levelIndex) {
   return (double)(200 + NUM_BUFFERS * (levelIndex * NUM_ITEMS));
}
