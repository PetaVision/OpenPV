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
   int getStartIndex(int b) const { return mStartIndices.at(b); }
   int getSkipAmount(int b) const { return mSkipAmounts.at(b); }
   std::vector<int> getIndices() { return mIndices; }

   virtual Response::Status
   registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) override;

  protected:
   virtual Response::Status processCheckpointRead(double simTime) override;
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
};
} // namespace PV

#endif // BATCHINDEXER_HPP_
