#pragma once

#include <vector>

namespace PV {

   class BatchIndexer {
      
      public:
         enum BatchMethod {
            BYFILE,
            BYLIST,
            BYSPECIFIED
         };

         BatchIndexer(int globalBatchCount, int globalBatchIndex, int batchWidth, int fileCount, enum BatchMethod batchMethod);
         int nextIndex(int localBatchIndex);
         void specifyBatching(int localBatchIndex, int startIndex, int skipAmount);
         void initializeBatch(int localBatchIndex);
         void setIndices(const std::vector<int> &indices) { mIndices = indices; }
         void setWrapToStartIndex(bool value) { mWrapToStartIndex = value; }
         bool getWrapToStartIndex() { return mWrapToStartIndex; }
         std::vector<int> getIndices() { return mIndices; }

      private:
         int mGlobalBatchCount = 0;
         int mFileCount = 0;
         bool mWrapToStartIndex = true;
         int mBatchWidth = 0;
         int mBatchWidthIndex = 0;
         std::vector<int> mIndices;
         std::vector<int> mStartIndices;
         std::vector<int> mSkipAmounts;
         BatchMethod mBatchMethod;
   };

}
