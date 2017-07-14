/*
 * CheckpointEntry.cpp
 *
 *  Created on Sep 27, 2016
 *      Author: Pete Schultz
 */

#include "CheckpointEntryWeightPvp.hpp"
#include "io/fileio.hpp"
#include "structures/Buffer.hpp"
#include "utils/BufferUtilsMPI.hpp"
#include "utils/BufferUtilsPvp.hpp"
#include <limits>

namespace PV {

void CheckpointEntryWeightPvp::initialize(
      int numArbors,
      bool sharedWeights,
      PVPatch const *const *const *patchGeometry,
      float **weightData,
      int numPatchesX,
      int numPatchesY,
      int numPatchesF,
      int nxp,
      int nyp,
      int nfp,
      PVLayerLoc const *preLoc,
      PVLayerLoc const *postLoc,
      bool compressFlag) {
   mNumArbors     = numArbors;
   mSharedWeights = sharedWeights;
   mPatchGeometry = patchGeometry;
   mWeightData    = weightData;
   mNumPatchesX = numPatchesX, mNumPatchesY = numPatchesY, mNumPatchesF = numPatchesF,
   mWeightDataSize = numPatchesX * numPatchesY * numPatchesF;
   mPatchSizeX     = nxp;
   mPatchSizeY     = nyp;
   mPatchSizeF     = nfp;
   mPreLoc         = preLoc;
   mPostLoc        = postLoc;
   mCompressFlag   = compressFlag;
}

void CheckpointEntryWeightPvp::calcMinMaxWeights(float *minWeightPtr, float *maxWeightPtr) const {
   float minWeight = std::numeric_limits<float>::infinity();
   float maxWeight = -std::numeric_limits<float>::infinity();
   for (int arbor = 0; arbor < mNumArbors; arbor++) {
      float const *arborStart = mWeightData[arbor];
      int const numValues     = mPatchSizeX * mPatchSizeY * mPatchSizeF * mWeightDataSize;
      for (int k = 0; k < numValues; k++) {
         float weight = arborStart[k];
         if (weight < minWeight) {
            minWeight = weight;
         }
         if (weight > maxWeight) {
            maxWeight = weight;
         }
      }
   }
   *minWeightPtr = minWeight;
   *maxWeightPtr = maxWeight;
   if (!mSharedWeights) {
      float extrema[2];
      extrema[0] = minWeight;
      extrema[1] = -maxWeight;
      MPI_Allreduce(MPI_IN_PLACE, extrema, 2, MPI_FLOAT, MPI_MIN, getMPIBlock()->getComm());
      minWeight = extrema[0];
      maxWeight = -extrema[1];
   }
}

void CheckpointEntryWeightPvp::write(
      std::string const &checkpointDirectory,
      double simTime,
      bool verifyWritesFlag) const {
   std::string path(checkpointDirectory);
   path.append("/").append(getName()).append(".pvp");
   float minWeight, maxWeight;
   calcMinMaxWeights(&minWeight, &maxWeight);
   if (mSharedWeights) {
      if (getMPIBlock()->getRank() == 0) {
         FileStream fileStream(path.c_str(), std::ios_base::out, verifyWritesFlag);
         writeSharedWeights(
               simTime,
               &fileStream,
               getMPIBlock(),
               mPreLoc,
               mPatchSizeX,
               mPatchSizeY,
               mPatchSizeF,
               mNumArbors,
               mWeightData,
               mCompressFlag,
               minWeight,
               maxWeight,
               mNumPatchesX,
               mNumPatchesY,
               mNumPatchesF);
      }
   }
   else {
      FileStream *fileStream = nullptr;
      if (getMPIBlock()->getRank() == 0) {
         fileStream = new FileStream(path.c_str(), std::ios_base::out, verifyWritesFlag);
      }
      writeNonsharedWeights(
            simTime,
            fileStream,
            getMPIBlock(),
            mPreLoc,
            mPatchSizeX,
            mPatchSizeY,
            mPatchSizeF,
            mNumArbors,
            mWeightData,
            mCompressFlag,
            true /*extended*/,
            mPostLoc,
            mPatchGeometry);
      delete fileStream;
   }
}

void CheckpointEntryWeightPvp::read(std::string const &checkpointDirectory, double *simTimePtr)
      const {
   // Need to clear weights before reading because reading weights is increment-add, not assignment.
   for (int a = 0; a < mNumArbors; a++) {
      int const numWeightsInArbor = mWeightDataSize * mPatchSizeX * mPatchSizeY * mPatchSizeF;
      memset(mWeightData[a], 0, sizeof(mWeightData[a][0]) * (std::size_t)numWeightsInArbor);
   }

   std::string path(checkpointDirectory);
   path.append("/").append(getName()).append(".pvp");
   FileStream *fileStream = nullptr;
   if (getMPIBlock()->getRank() == 0) {
      fileStream = new FileStream(path.c_str(), std::ios_base::in, false);
   }

   if (mSharedWeights) {
      readSharedWeights(
            fileStream,
            0 /*frameNumber*/,
            getMPIBlock(),
            mPreLoc,
            mPatchSizeX,
            mPatchSizeY,
            mPatchSizeF,
            mNumArbors,
            mWeightData,
            mNumPatchesX,
            mNumPatchesY,
            mNumPatchesF);
   }
   else {
      readNonsharedWeights(
            fileStream,
            0 /*frameNumber*/,
            getMPIBlock(),
            mPreLoc,
            mPatchSizeX,
            mPatchSizeY,
            mPatchSizeF,
            mNumArbors,
            mWeightData,
            true /*extended*/,
            mPostLoc,
            0 /*offsetX*/,
            0 /*offsetY*/);
   }
}

void CheckpointEntryWeightPvp::remove(std::string const &checkpointDirectory) const {
   deleteFile(checkpointDirectory, "pvp");
}

} // end namespace PV
