/*
 * DataStoreTest.cpp
 *
 */

#include <columns/DataStore.hpp>
#include <utils/PVLog.hpp>

const int NUM_BUFFERS = 2;
const int NUM_ITEMS   = 10;
const int NUM_LEVELS  = 4;

float correctData(int bufferIndex, int itemIndex, int levelIndex);
double correctTime(int bufferIndex, int levelIndex);

int main(int argc, char *argv[]) {
   PV::DataStore store(NUM_BUFFERS, NUM_ITEMS, NUM_LEVELS, false /*store is not sparse*/);

   // Many DataStore methods are overloaded with versions with and without the
   // level index argument. Verify that they are consistent.
   FatalIf(
         store.buffer(0) != store.buffer(0, 0),
         "buffer(0,0) should be equal to buffer(0) but is not.\n");
   FatalIf(
         store.buffer(1) != store.buffer(1, 0),
         "buffer(1,0) should be equal to buffer(1) but is not.\n");

   // Initialize level 0.
   for (int itemIndex = 0; itemIndex < NUM_ITEMS; itemIndex++) {
      store.buffer(0)[itemIndex] = correctData(0, itemIndex, 0 /*levelIndex*/);
      store.buffer(1)[itemIndex] = correctData(1, itemIndex, 0 /*levelIndex*/);
   }
   store.setLastUpdateTime(0, correctTime(0, 0));
   store.setLastUpdateTime(1, correctTime(1, 0));

   // Initialize the remaining levels, using the levelIndex argument
   for (int levelIndex = 1; levelIndex < NUM_LEVELS; levelIndex++) {
      for (int itemIndex = 0; itemIndex < NUM_ITEMS; itemIndex++) {
         store.buffer(0, levelIndex)[itemIndex] = correctData(0, itemIndex, levelIndex);
         store.buffer(1, levelIndex)[itemIndex] = correctData(1, itemIndex, levelIndex);
      }
      store.setLastUpdateTime(0, levelIndex, correctTime(0, levelIndex));
      store.setLastUpdateTime(1, levelIndex, correctTime(1, levelIndex));
   }

   // Verify that the values are correct, using nextLevelIndex to advance the levels.
   // We go to two times the number of levels to verify wrap-around.
   for (int levelIndex = 0; levelIndex < 2 * NUM_LEVELS; levelIndex++) {
      // newLevelIndex decrements mCurrentLevel
      int correctLevel = (2 * NUM_LEVELS - levelIndex) % NUM_LEVELS;
      for (int itemIndex = 0; itemIndex < NUM_ITEMS; itemIndex++) {
         FatalIf(
               store.buffer(0)[itemIndex] != correctData(0, itemIndex, correctLevel),
               "Verification failed for buffer 0, item %d after %d rotations.\n",
               itemIndex,
               levelIndex);
         FatalIf(
               store.buffer(1)[itemIndex] != correctData(1, itemIndex, correctLevel),
               "Verification failed for buffer 1, item %d after %d rotations.\n",
               itemIndex,
               levelIndex);
      }
      FatalIf(
            store.getLastUpdateTime(0) != correctTime(0, correctLevel),
            "Verification failed for buffer 0, LastUpdateTime after %d rotations.\n",
            levelIndex);
      FatalIf(
            store.getLastUpdateTime(1) != correctTime(1, correctLevel),
            "Verification failed for buffer 1, LastUpdateTime after %d rotations.\n",
            levelIndex);
      store.newLevelIndex(); // Rotate.
   }

   return EXIT_SUCCESS;
}

float correctData(int bufferIndex, int itemIndex, int levelIndex) {
   return (float)(bufferIndex + NUM_BUFFERS * (itemIndex + levelIndex * NUM_ITEMS));
}

double correctTime(int bufferIndex, int levelIndex) {
   return (double)(200 + bufferIndex + NUM_BUFFERS * (levelIndex * NUM_ITEMS));
}
