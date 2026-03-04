#ifndef BATCHINDEXER_HPP_
#define BATCHINDEXER_HPP_

#include "utils/cl_random.h"

#include <string>
#include <vector>

namespace PV {

class BatchIndexer {

  public:

   BatchIndexer(
         std::string const &objName,
         int globalBatchCount,
         int batchWidth,
         int fileCount,
         std::vector<int> const &start_indices,
         std::vector<int> const &skip_indices);
   BatchIndexer(
         std::string const &objName,
         int globalBatchCount,
         int batchWidth,
         int fileCount,
         int batchOffset, // The global batch index of the zero-th batch element in this IO block
         taus_uint4 const &rng);
   void advanceIndices();
   int getIndex(int localBatchIndex);
   taus_uint4 const &getRandomState() const { return mOldRNG; }
   taus_uint4 &getRandomState() { return mOldRNG; }
   void setRandomState(taus_uint4 const &rng);
   void setIndices(const std::vector<int> &indices);
   void setWrapToStartIndex(bool value) { mWrapToStartIndex = value; }
   bool getWrapToStartIndex() { return mWrapToStartIndex; }
   int getStartIndex(int b) const { return mStartIndices.at(b); }
   int getSkipAmount(int b) const { return mSkipAmounts.at(b); }
   std::vector<int> const &getIndices() const { return mIndices; }

  protected:
   void generateIndexLookupTable();

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
   taus_uint4 mRNG;
   taus_uint4 mOldRNG;
   // OldRNG is what gets checkpointed, because generating the lookup table changes the RNG's state.
   bool mWrapToStartIndex = true;
   int mGlobalBatchStartIndex; // Used if batchMethod=Random to manage reshuffling

   // A vector whose length is the number of images, used with batchMethod=random.
   // A permutation of the integers {1,...,numImages}, reshuffled every time the end
   // of the images is reached
   std::vector<int> mIndexLookupTable;

   // A vector whose length is the local batch width, used with batchMethod=random.
   // It is generally a slice of mIndexLookupTable, advancing through the table when
   // advanceIndices() is called. If the glboal batch count does not evenly divide the
   // file count, this vector may have slices from two different shuffles.
   std::vector<int> mIndexLookups;

   // A vector whose length is mBatchWidth, used with batchMethod=random.
   // It holds the section of mIndexLookupTable corresponding to the current set of images.
   std::vector<int> mFrameLookup;
   std::vector<int> mIndices;
   std::vector<int> mStartIndices;
   std::vector<int> mSkipAmounts;

   // If true, batch method is random and the object was created using the taus_uint4 constructor.
   // The StartIndices and SkipAmounts vectors are determined from the global batch size and input
   // file count and cannot otherwise be changed.
   // If false, batch method is not random and the object was created using the start/skip vectors
   // constructor; the taus_uint4 RNG data member is not used.
   bool mRandomFlag;
};
} // namespace PV

#endif // BATCHINDEXER_HPP_
