#include "BatchIndexer.hpp"
#include "utils/PVLog.hpp"
#include <algorithm>
#include <random>

namespace PV {

// This takes in the global batch index of local batch 0 for the second argument.
// Should this be the value of commBatch() instead?
BatchIndexer::BatchIndexer(
      std::string const &objName,
      int globalBatchCount,
      int batchOffset,
      int batchWidth,
      int fileCount,
      enum BatchMethod batchMethod,
      bool initializeFromCheckpointFlag) {
   mObjName                      = objName;
   mGlobalBatchCount             = globalBatchCount;
   mBatchMethod                  = batchMethod;
   mFileCount                    = fileCount ? fileCount : 1;
   mBatchWidth                   = batchWidth;
   mBatchOffset                  = batchOffset;
   mInitializeFromCheckpointFlag = initializeFromCheckpointFlag;
   mIndices.resize(mBatchWidth, 0);
   mStartIndices.resize(mBatchWidth, 0);
   mSkipAmounts.resize(mBatchWidth, 0);
   shuffleLookupTable();
}

int BatchIndexer::nextIndex(int localBatchIndex) {
   int result   = getIndex(localBatchIndex);
   int newIndex = mIndices.at(localBatchIndex) + mSkipAmounts.at(localBatchIndex);
   if (newIndex >= mFileCount) {
      shuffleLookupTable();
      if (mWrapToStartIndex) {
         newIndex = mStartIndices.at(localBatchIndex);
      }
      else {
         newIndex %= mFileCount;
      }
   }
   mIndices.at(localBatchIndex) = newIndex;
   return newIndex;
}

int BatchIndexer::getIndex(int localBatchIndex) {
   if (mBatchMethod != RANDOM) {
      return mIndices.at(localBatchIndex);
   }
   return mIndexLookupTable.at(mIndices.at(localBatchIndex));
}

void BatchIndexer::specifyBatching(int localBatchIndex, int startIndex, int skipAmount) {
   mStartIndices.at(localBatchIndex) = startIndex % mFileCount;
   mSkipAmounts.at(localBatchIndex)  = skipAmount < 1 ? 1 : skipAmount;
}

void BatchIndexer::initializeBatch(int localBatchIndex) {
   int globalBatchIndex = mBatchOffset + localBatchIndex;
   switch (mBatchMethod) {
      case RANDOM:
      case BYFILE:
         specifyBatching(
               localBatchIndex,
               mStartIndices.at(localBatchIndex) + globalBatchIndex,
               mGlobalBatchCount);
         break;
      case BYLIST:
         specifyBatching(
               localBatchIndex,
               mStartIndices.at(localBatchIndex)
                     + globalBatchIndex * (mFileCount / mGlobalBatchCount),
               1);
         break;
      case BYSPECIFIED:
         FatalIf(
               mSkipAmounts.at(localBatchIndex) < 1,
               "BatchIndexer batchMethod was set to BYSPECIFIED, but no values were specified.\n");
         break;
   }
   mIndices.at(localBatchIndex) = mStartIndices.at(localBatchIndex);
}

void BatchIndexer::setRandomSeed(unsigned int seed) {
   mRandomSeed = seed;
   shuffleLookupTable();
}

// This clears the current file index lookup table and fills it with
// randomly ordered ints from 0 to mFileCount. The random seed is
// incremented so the next time this is called it  will result in a new order.
// Two objects with BatchIndexers with the same seed will randomize the order
// in the same manner.
void BatchIndexer::shuffleLookupTable() {
   if (mBatchMethod != RANDOM) {
      return;
   }
   mIndexLookupTable.clear();
   mIndexLookupTable.resize(mFileCount);
   for (int i = 0; i < mFileCount; ++i) {
      mIndexLookupTable.at(i) = i;
   }
   std::mt19937 rng(mRandomSeed++);
   std::shuffle(mIndexLookupTable.begin(), mIndexLookupTable.end(), rng);
}

Response::Status BatchIndexer::registerData(Checkpointer *checkpointer) {
   auto status = CheckpointerDataInterface::registerData(checkpointer);
   if (!Response::completed(status)) {
      return status;
   }
   checkpointer->registerCheckpointData<int>(
         mObjName,
         std::string("FrameNumbers"),
         mIndices.data(),
         mIndices.size(),
         false /*do not broadcast*/,
         false /*not constant*/);
   if (mBatchMethod == RANDOM) {
      checkpointer->registerCheckpointData<unsigned int>(
            mObjName,
            std::string("RandomSeed"),
            &mRandomSeed,
            1,
            false /*do not broadcast*/,
            false /*not constant*/);
   }
   return Response::SUCCESS;
}

Response::Status BatchIndexer::processCheckpointRead() {
   checkIndices();
   return Response::SUCCESS;
}

Response::Status BatchIndexer::readStateFromCheckpoint(Checkpointer *checkpointer) {
   if (mInitializeFromCheckpointFlag) {
      checkpointer->readNamedCheckpointEntry(mObjName, "FrameNumbers", false /*not constant*/);
      checkIndices();
      return Response::SUCCESS;
   }
   else {
      return Response::NO_ACTION;
   }
}

void BatchIndexer::checkIndices() {
   for (int k = 0; k < mBatchWidth; k++) {
      int n = getIndex(k);
      FatalIf(
            n >= mFileCount,
            "BatchIndexer \"%s\" has index %d=%d, but fileCount is only %d.\n",
            mObjName.c_str(),
            k,
            n,
            mFileCount);
      FatalIf(
            n < 0,
            "BatchIndexer \"%s\" has index %d=%d. Indices cannot be negative.\n",
            mObjName.c_str(),
            k,
            n);
   }
}

} // end namespace PV
