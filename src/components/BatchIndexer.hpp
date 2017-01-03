#pragma once

#include "checkpointing/Checkpointer.hpp"
#include <vector>

namespace PV {

class BatchIndexer : public CheckpointerDataInterface {

  public:
   enum BatchMethod { BYFILE, BYLIST, BYSPECIFIED, RANDOM };

   BatchIndexer(
         int globalBatchCount,
         int globalBatchIndex,
         int batchWidth,
         int fileCount,
         enum BatchMethod batchMethod);
   int nextIndex(int localBatchIndex);
   int getIndex(int localBatchIndex);
   void specifyBatching(int localBatchIndex, int startIndex, int skipAmount);
   void initializeBatch(int localBatchIndex);
   void shuffleLookupTable();
   void setRandomSeed(int seed);
   virtual int registerData(Checkpointer *checkpointer, std::string const &objName) override;
   void setIndices(const std::vector<int> &indices) { mIndices = indices; }
   void setWrapToStartIndex(bool value) { mWrapToStartIndex = value; }
   bool getWrapToStartIndex() { return mWrapToStartIndex; }
   std::vector<int> getIndices() { return mIndices; }

  private:
   int mGlobalBatchCount  = 0;
   int mFileCount         = 0;
   int mBatchWidth        = 0;
   int mBatchWidthIndex   = 0;
   int mRandomSeed        = 123456789;
   bool mWrapToStartIndex = true;
   std::vector<int> mIndexLookupTable;
   std::vector<int> mIndices;
   std::vector<int> mStartIndices;
   std::vector<int> mSkipAmounts;
   BatchMethod mBatchMethod;
};
}
