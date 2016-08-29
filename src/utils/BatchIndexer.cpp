#include "BatchIndexer.hpp"
#include "PVLog.hpp"

namespace PV {
   
   // This takes in the global batch index of local batch 0 for the second argument. Should this be the value of commBatch() instead?
   BatchIndexer::BatchIndexer(int globalBatchCount, int globalBatchIndex, int batchWidth, int fileCount, enum BatchMethod batchMethod) {
      mGlobalBatchCount = globalBatchCount;
      mBatchMethod = batchMethod;
      mFileCount = fileCount;
      mBatchWidth = batchWidth > 0 ? batchWidth : 1;
      mBatchWidthIndex = globalBatchIndex / mBatchWidth;
      mIndices.resize(mGlobalBatchCount / mBatchWidth, 0);
      mStartIndices.resize(mGlobalBatchCount / mBatchWidth, 0);
      mSkipAmounts.resize(mGlobalBatchCount / mBatchWidth, 0);
   }

   int BatchIndexer::nextIndex(int localBatchIndex) {
      int currentIndex = mIndices.at(localBatchIndex);
      int newIndex = currentIndex + mSkipAmounts.at(localBatchIndex);
      if(newIndex >= mFileCount) {
         if(mWrapToStartIndex) {
            newIndex = mStartIndices.at(localBatchIndex);
         }
         else {
            newIndex %= mFileCount;
         }
      }
      mIndices.at(localBatchIndex) = newIndex;
      return currentIndex;
   }

   void BatchIndexer::specifyBatching(int localBatchIndex, int startIndex, int skipAmount) {
      mStartIndices.at(localBatchIndex) = startIndex;
      mSkipAmounts.at(localBatchIndex) = skipAmount < 1 ? 1 : skipAmount;
   }
    
   void BatchIndexer::initializeBatch(int localBatchIndex) {
      int globalBatchIndex = mBatchWidthIndex * mBatchWidth + localBatchIndex;
      switch (mBatchMethod) {
         case BYFILE:
            specifyBatching(localBatchIndex, mStartIndices.at(localBatchIndex) + globalBatchIndex, mGlobalBatchCount);
            break;
         case BYLIST:
            specifyBatching(localBatchIndex, mStartIndices.at(localBatchIndex) + globalBatchIndex * (mFileCount / mGlobalBatchCount), 1); 
            break;
         case BYSPECIFIED:
            pvErrorIf(mSkipAmounts.at(localBatchIndex) < 1, "BatchIndexer batchMethod was set to BYSPECIFIED, but no values were specified.\n");
            break;
      }
      mIndices.at(localBatchIndex) = mStartIndices.at(localBatchIndex);
   }
}
