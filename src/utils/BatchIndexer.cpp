#include "BatchIndexer.hpp"
#include "PVLog.hpp"

namespace PV {
   
   BatchIndexer::BatchIndexer(int globalBatchCount, int globalBatchIndex, int batchWidth, int fileCount, enum BatchMethod batchMethod) {
      mGlobalBatchCount = globalBatchCount;
      mBatchMethod = batchMethod;
      mFileCount = fileCount;
      mBatchWidth = batchWidth;
      mBatchWidthIndex = globalBatchIndex / batchWidth;

      mIndices.resize(mGlobalBatchCount);
      mStartIndices.resize(mGlobalBatchCount);
      mSkipAmounts.resize(mGlobalBatchCount);
   }

   int BatchIndexer::nextIndex(int localBatchIndex) {
      int currentIndex = mIndices.at(localBatchIndex);
      int newIndex = currentIndex + mSkipAmounts.at(localBatchIndex);

      pvDebug() << "CURRENT: " << currentIndex << ", NEW: " << newIndex << "\n";

      if(newIndex >= mFileCount) {
         pvDebug() << " REWINDING, EXCEEDED FILECOUNT = " << mFileCount << "\n";
         newIndex %= mFileCount;
         if(mWrapToStartIndex) {
            newIndex += mStartIndices.at(localBatchIndex);
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
      switch (mBatchMethod) {
         case BYFILE:
            specifyBatching(localBatchIndex, mStartIndices.at(localBatchIndex) + mBatchWidth * mBatchWidthIndex + localBatchIndex, mGlobalBatchCount);
            break;
         case BYLIST:
            specifyBatching(localBatchIndex, mStartIndices.at(localBatchIndex) + (mBatchWidth * mBatchWidthIndex + localBatchIndex) * (mFileCount / mGlobalBatchCount), 1); 
            break;
         case BYSPECIFIED:
            pvErrorIf(mSkipAmounts.at(localBatchIndex) < 1, "BatchIndexer batchMethod was set to BYSPECIFIED, but no values were specified.\n");
            break;
      }
      mIndices.at(localBatchIndex) = mStartIndices.at(localBatchIndex);
   }
}
