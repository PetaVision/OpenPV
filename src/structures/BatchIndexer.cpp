#include "BatchIndexer.hpp"
#include "utils/PVAssert.hpp"
#include "utils/TauswortheURBG.hpp"
#include <algorithm>

namespace PV {

BatchIndexer::BatchIndexer(
      std::string const &objName,
      int globalBatchCount,
      int batchOffset,
      int batchWidth,
      int fileCount,
      std::vector<int> const &start_indices,
      std::vector<int> const &skip_amounts) {
   FatalIf(
         start_indices.size() != std::vector<int>::size_type(batchWidth),
         "BatchIndexer for \"%s\": start_indices vector size %zu does not match batchWidth %d\n",
         std::size_t(start_indices.size()),
         batchWidth);
   FatalIf(
         skip_amounts.size() != std::vector<int>::size_type(batchWidth),
         "BatchIndexer for \"%s\": skip_indices vector size %zu does not match batchWidth %d\n",
         std::size_t(skip_amounts.size()),
         batchWidth);
   mRandomFlag                   = false;
   mObjName                      = objName;
   mGlobalBatchCount             = globalBatchCount;
   mFileCount                    = fileCount ? fileCount : 1;
   mBatchWidth                   = batchWidth;
   mBatchOffset                  = batchOffset;
   mIndices                      = start_indices;
   mStartIndices                 = start_indices;
   mSkipAmounts                  = skip_amounts;
}

BatchIndexer::BatchIndexer(
      std::string const &objName,
      int globalBatchCount,
      int batchOffset,
      int batchWidth,
      int fileCount,
      taus_uint4 const &rng) {
   mRandomFlag                   = true;
   mObjName                      = objName;
   mGlobalBatchCount             = globalBatchCount;
   mFileCount                    = fileCount ? fileCount : 1;
   mBatchWidth                   = batchWidth;
   mBatchOffset                  = batchOffset;
   mStartIndices.resize(batchWidth);
   mSkipAmounts.resize(batchWidth);
   for (int b =  0; b < batchWidth; ++b) {
      mStartIndices[b] = (batchOffset + b) % mFileCount;
      mSkipAmounts[b] = globalBatchCount;
   }
   mIndices = mStartIndices;
   mIndexLookupTable.resize(mFileCount);
   setRandomState(rng);
}

void BatchIndexer::setRandomState(taus_uint4 const &rng) {
   mRNG = rng;
   generateIndexLookupTable();
}

void BatchIndexer::generateIndexLookupTable() {
   mOldRNG = mRNG;
   int count = static_cast<int>(mIndexLookupTable.size());
   for (int i = 0; i < count; ++i) {
      mIndexLookupTable[i] = i;
   }
   TauswortheURBG urbg(mRNG);
   std::shuffle(mIndexLookupTable.begin(), mIndexLookupTable.end(), urbg);
   mRNG = urbg.getState();
}

void BatchIndexer::advanceIndices() {
   if (!mRandomFlag) {
      for (int b = 0; b < mBatchWidth; ++b) {
         int newIndex = mIndices.at(b) + mSkipAmounts.at(b);
         if (newIndex >= mFileCount) {
            if (mWrapToStartIndex) {
               newIndex = mStartIndices.at(b);
            }
            else {
               newIndex %= mFileCount;
            }
         }
         mIndices.at(b) = newIndex;
      }
   }
   else {
      // mRandomFlag == true


      for (int b = 0; b < mBatchWidth; ++b) {
         int newIndex = mIndices.at(b) + mSkipAmounts.at(b);
         if (newIndex >= mFileCount) {
            generateIndexLookupTable();
            if (mWrapToStartIndex) {
               newIndex = mStartIndices.at(b);
            }
            else {
               newIndex %= mFileCount;
            }
         }
         mIndices.at(b) = newIndex;
      }


   }
}

int BatchIndexer::getIndex(int localBatchIndex) {
   if (mRandomFlag) {
      return mIndexLookupTable.at(mIndices.at(localBatchIndex));
   }
   return mIndices.at(localBatchIndex);
}

void BatchIndexer::specifyBatching(int localBatchIndex, int startIndex, int skipAmount) {
   mStartIndices.at(localBatchIndex) = startIndex % mFileCount;
   mSkipAmounts.at(localBatchIndex)  = skipAmount < 1 ? 1 : skipAmount;
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

void BatchIndexer::setIndices(const std::vector<int> &indices) {
   mIndices = indices;
   checkIndices();
}

} // end namespace PV
