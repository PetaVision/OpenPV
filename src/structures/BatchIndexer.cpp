#include "BatchIndexer.hpp"
#include "utils/PVAssert.hpp"
#include "utils/TauswortheURBG.hpp"
#include <algorithm>

namespace PV {

BatchIndexer::BatchIndexer(
      std::string const &objName,
      int globalBatchCount,
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
   mRandomFlag       = false;
   mObjName          = objName;
   mGlobalBatchCount = globalBatchCount;
   mFileCount        = fileCount ? fileCount : 1;
   mBatchWidth       = batchWidth;
   mIndices          = start_indices;
   mStartIndices     = start_indices;
   mSkipAmounts      = skip_amounts;
}

BatchIndexer::BatchIndexer(
      std::string const &objName,
      int globalBatchCount,
      int batchWidth,
      int fileCount,
      int batchOffset,
      taus_uint4 const &rng) {
   FatalIf(
         fileCount < globalBatchCount,
         "BatchIndexer with batchMethod=random requires FileCount >= GlobalBatchCount "
         "(FileCount=%d, GlobalBatchCount=%d).\n",
         fileCount, globalBatchCount);
   mRandomFlag       = true;
   mObjName          = objName;
   mGlobalBatchCount = globalBatchCount;
   mFileCount        = fileCount ? fileCount : 1;
   mBatchWidth       = batchWidth;
   mBatchOffset      = batchOffset;
   mStartIndices.resize(batchWidth);
   for (int b =  0; b < batchWidth; ++b) {
      mStartIndices[b] = (batchOffset + b) % mFileCount;
   }
   mGlobalBatchStartIndex = 0;
   mIndices = mStartIndices;
   mIndexLookupTable.resize(mFileCount);
   mIndexLookups.resize(mBatchWidth);
   setRandomState(rng);
   for (int b = 0; b < mBatchWidth; ++b) {
      mIndexLookups[b] = mIndexLookupTable.at(batchOffset + b);
   }
}

void BatchIndexer::setRandomState(taus_uint4 const &rng) {
   mRNG = rng;
   generateIndexLookupTable();
   mGlobalBatchStartIndex = (mIndices[0] - mBatchOffset) % mFileCount;
   if (mGlobalBatchStartIndex < 0) {
      mGlobalBatchStartIndex += mFileCount;
   }
   for (int b = 0; b < mBatchWidth; ++b) {
      mIndexLookups[b] = mIndexLookupTable.at(mIndices[b]);
   }
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

      // Use global condition for reshuffling, to keep all processes in sync
      mGlobalBatchStartIndex += mGlobalBatchCount;
      if (mGlobalBatchStartIndex >= mFileCount) {
         generateIndexLookupTable();
         if (mWrapToStartIndex) {
            mGlobalBatchStartIndex = 0;
         }
         else {
            mGlobalBatchStartIndex %= mFileCount;
         }
      }

      for (int b = 0; b < mBatchWidth; ++b) {
         int newIndex = mIndices.at(b) + mGlobalBatchCount;
         if (newIndex >= mFileCount) {
            if (mWrapToStartIndex) {
               newIndex = mStartIndices.at(b);
            }
            else {
               newIndex %= mFileCount;
            }
         }
         mIndexLookups[b] = mIndexLookupTable[newIndex];
         mIndices.at(b) = newIndex;
      }
   }
}

int BatchIndexer::getIndex(int localBatchIndex) {
   if (mRandomFlag) {
      return mIndexLookups.at(localBatchIndex);
   }
   return mIndices.at(localBatchIndex);
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
