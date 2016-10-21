/*
 * CheckpointEntry.cpp
 *
 *  Created on Sep 27, 2016
 *      Author: Pete Schultz
 */

#include "CheckpointEntryWeightPvp.hpp"
#include "structures/Buffer.hpp"
#include "utils/BufferUtilsMPI.hpp"
#include "utils/BufferUtilsPvp.hpp"
#include <limits>

namespace PV {

void CheckpointEntryWeightPvp::initialize(
      int numArbors,
      bool sharedWeights,
      PVPatch ***patchData,
      int patchDataSize,
      pvdata_t **weightData,
      int weightDataSize,
      int nxp,
      int nyp,
      int nfp,
      PVLayerLoc const *preLoc,
      PVLayerLoc const *postLoc,
      bool compressFlag) {
   mNumArbors      = numArbors;
   mSharedWeights  = sharedWeights;
   mPatchData      = patchData;
   mPatchDataSize  = patchDataSize;
   mWeightData     = weightData;
   mWeightDataSize = weightDataSize;
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
      int const numValues     = mPatchSizeX * mPatchSizeY * mPatchSizeF;
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
      MPI_Allreduce(
            MPI_IN_PLACE, extrema, 2, MPI_FLOAT, MPI_MIN, getCommunicator()->communicator());
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
   int fileType;
   PVPatch ***patchesArgument;
   if (mSharedWeights) {
      fileType        = PVP_KERNEL_FILE_TYPE;
      patchesArgument = nullptr;
   }
   else {
      fileType        = PVP_WGT_FILE_TYPE;
      patchesArgument = mPatchData;
   }
   writeWeights(
         path.c_str(),
         getCommunicator(),
         simTime,
         false /*do not append*/,
         mPreLoc,
         mPostLoc,
         mPatchSizeX,
         mPatchSizeY,
         mPatchSizeF,
         minWeight,
         maxWeight,
         patchesArgument,
         mWeightData,
         mWeightDataSize,
         mNumArbors,
         mCompressFlag,
         fileType);
}

void CheckpointEntryWeightPvp::read(std::string const &checkpointDirectory, double *simTimePtr)
      const {
   std::string path(checkpointDirectory);
   path.append("/").append(getName()).append(".pvp");
   int fileType;
   PVPatch ***patchesArgument;
   if (mSharedWeights) {
      fileType        = PVP_KERNEL_FILE_TYPE;
      patchesArgument = nullptr;
   }
   else {
      fileType        = PVP_WGT_FILE_TYPE;
      patchesArgument = mPatchData;
   }

   // Need to clear weights before reading because readWeights is increment-add, not assignment.
   for (int a = 0; a < mNumArbors; a++) {
      int const numWeightsInArbor = mWeightDataSize * mPatchSizeX * mPatchSizeY * mPatchSizeF;
      memset(mWeightData[a], 0, sizeof(mWeightData[a][0]) * (std::size_t)numWeightsInArbor);
   }
   readWeights(
         patchesArgument,
         mWeightData,
         mNumArbors,
         mWeightDataSize,
         mPatchSizeX,
         mPatchSizeY,
         mPatchSizeF,
         path.c_str(),
         getCommunicator(),
         simTimePtr,
         mPreLoc);
}

void CheckpointEntryWeightPvp::remove(std::string const &checkpointDirectory) const {
   deleteFile(checkpointDirectory, "pvp");
}

} // end namespace PV
