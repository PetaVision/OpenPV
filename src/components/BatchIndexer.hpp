#ifndef BATCHINDEXER_HPP_
#define BATCHINDEXER_HPP_

#include "checkpointing/CheckpointerDataInterface.hpp"
#include <vector>

namespace PV {

class BatchIndexer : public CheckpointerDataInterface {

  public:
   enum BatchMethod { BYFILE, BYLIST, BYSPECIFIED, RANDOM };

   BatchIndexer(
         std::string const &objName,
         int globalBatchCount,
         int globalBatchIndex,
         int batchWidth,
         int fileCount,
         enum BatchMethod batchMethod,
         bool initializeFromCheckpointFlag);
   int nextIndex(int localBatchIndex);
   int getIndex(int localBatchIndex);
   void specifyBatching(int localBatchIndex, int startIndex, int skipAmount);
   void initializeBatch(int localBatchIndex);
   void shuffleLookupTable();
   void setRandomSeed(unsigned int seed);
   void setIndices(const std::vector<int> &indices) { mIndices = indices; }
   void setWrapToStartIndex(bool value) { mWrapToStartIndex = value; }
   bool getWrapToStartIndex() { return mWrapToStartIndex; }
   std::vector<int> getIndices() { return mIndices; }

   virtual Response::Status registerData(Checkpointer *checkpointer) override;

  protected:
   virtual Response::Status processCheckpointRead() override;
   virtual Response::Status readStateFromCheckpoint(Checkpointer *checkpointer) override;

   /** Exits with error if any of index is negative or >= fileCount.
    *  Called when reading or initializing from checkpoint.
    */
   void checkIndices();

  private:
   std::string mObjName;
   int mGlobalBatchCount    = 0;
   int mFileCount           = 0;
   int mBatchWidth          = 0;
   int mBatchOffset         = 0;
   unsigned int mRandomSeed = 123456789;
   bool mWrapToStartIndex   = true;
   std::vector<int> mIndexLookupTable;
   std::vector<int> mIndices;
   std::vector<int> mStartIndices;
   std::vector<int> mSkipAmounts;
   BatchMethod mBatchMethod;
   bool mInitializeFromCheckpointFlag = false;
   // mInitializeFromCheckpointFlag is a hack.
   // BatchIndexer should load the indices from checkpoint when the InputLayer's
   // initializeFromCheckpointFlag is true, and not when it's false.
   // The problem is that BatchIndexer can't see the InputLayer, where the
   // initializeFromCheckpointFlag is read.
};
}

#endif // BATCHINDEXER_HPP_
