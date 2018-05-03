/*
 * Weights.cpp
 *
 *  Created on: Jul 21, 2017
 *      Author: Pete Schultz
 */

#include "Weights.hpp"
#include "checkpointing/CheckpointEntryWeightPvp.hpp"
#include "utils/PVAssert.hpp"
#include "utils/conversions.h"
#include <cstring>
#include <stdexcept>

namespace PV {

Weights::Weights(std::string const &name) { setName(name); }

Weights::Weights(
      std::string const &name,
      int patchSizeX,
      int patchSizeY,
      int patchSizeF,
      PVLayerLoc const *preLoc,
      PVLayerLoc const *postLoc,
      int numArbors,
      bool sharedWeights,
      double timestamp) {
   setName(name);
   initialize(
         patchSizeX, patchSizeY, patchSizeF, preLoc, postLoc, numArbors, sharedWeights, timestamp);
}

void Weights::initialize(
      std::shared_ptr<PatchGeometry> geometry,
      int numArbors,
      bool sharedWeights,
      double timestamp) {
   FatalIf(
         mGeometry != nullptr,
         "Weights object \"%s\" has already been initialized.\n",
         getName().c_str());
   mGeometry   = geometry;
   mNumArbors  = numArbors;
   mSharedFlag = sharedWeights;
   mTimestamp  = timestamp;

   initNumDataPatches();

#ifdef PV_USE_CUDA
   mTimestampGPU = timestamp;
#endif // PV_USE_CUDA
}

void Weights::initialize(Weights const *baseWeights) {
   auto geometry = baseWeights->getGeometry();
   initialize(
         geometry,
         baseWeights->getNumArbors(),
         baseWeights->getSharedFlag(),
         baseWeights->getTimestamp());
}

void Weights::initialize(
      int patchSizeX,
      int patchSizeY,
      int patchSizeF,
      PVLayerLoc const *preLoc,
      PVLayerLoc const *postLoc,
      int numArbors,
      bool sharedWeights,
      double timestamp) {
   auto geometry = std::make_shared<PatchGeometry>(
         mName.c_str(), patchSizeX, patchSizeY, patchSizeF, preLoc, postLoc);
   initialize(geometry, numArbors, sharedWeights, timestamp);
}

void Weights::setMargins(PVHalo const &preHalo, PVHalo const &postHalo) {
   mGeometry->setMargins(preHalo, postHalo);
   initNumDataPatches();
}

void Weights::allocateDataStructures() {
   if (!mData.empty()) {
      return;
   }
   FatalIf(mGeometry == nullptr, "%s has not been initialized.\n", mName.c_str());
   mGeometry->allocateDataStructures();

   int numDataPatches = mNumDataPatchesX * mNumDataPatchesY * mNumDataPatchesF;
   if (numDataPatches != 0) {
      int numItemsPerPatch = getPatchSizeOverall();
      mData.resize(mNumArbors);
      for (int arbor = 0; arbor < mNumArbors; arbor++) {
         mData[arbor].resize(numDataPatches * numItemsPerPatch);
      }
   }
   if (mSharedFlag and getNumDataPatches() > 0) {
      int const numPatches = mGeometry->getNumPatches();
      dataIndexLookupTable.resize(numPatches);
      for (int p = 0; p < numPatches; p++) {
         dataIndexLookupTable[p] = calcDataIndexFromPatchIndex(p);
      }
   }
#ifdef PV_USE_CUDA
   if (mUsingGPUFlag) {
      allocateCudaBuffers();
   }
#endif // PV_USE_CUDA
}

#ifdef PV_USE_CUDA
void Weights::allocateCudaBuffers() {
   FatalIf(
         mCudaDevice == nullptr,
         "Weights::allocateCudaBuffers() called for weights \"%s\" without having set "
         "CudaDevice.\n",
         getName().c_str());
   pvAssert(mDeviceData == nullptr); // Should only be called once, by allocateDataStructures();
#ifdef PV_USE_CUDNN
   pvAssert(mCUDNNData == nullptr); // Should only be called once, by allocateDataStructures();
#endif // PV_USE_CUDNN
   std::string description(mName);
   int numPatches = getGeometry()->getNumPatchesX() * getGeometry()->getNumPatchesY()
                    * getGeometry()->getNumPatchesF();
   std::size_t size;

   // Apr 10, 2018: mDevicePatches and mDeviceGSynPatchStart have been moved to
   // PresynapticPerspectiveGPUDelivery, since they are needed for the presynaptic perspective,
   // but not the postsynaptic perspective.

   if (getNumDataPatches() > 0) {
      std::vector<int> hostPatchToDataLookupVector(numPatches);
      if (mSharedFlag) {
         for (int patchIndex = 0; patchIndex < numPatches; patchIndex++) {
            hostPatchToDataLookupVector[patchIndex] = dataIndexLookupTable[patchIndex];
         }
      }
      else {
         for (int patchIndex = 0; patchIndex < numPatches; patchIndex++) {
            hostPatchToDataLookupVector[patchIndex] = patchIndex;
         }
      }
      size = hostPatchToDataLookupVector.size() * sizeof(hostPatchToDataLookupVector[0]);
      mDevicePatchToDataLookup = mCudaDevice->createBuffer(size, &description);
      // Copy PatchToDataLookup array onto CUDA device because it never changes.
      mDevicePatchToDataLookup->copyToDevice(hostPatchToDataLookupVector.data());

      size = (std::size_t)getNumArbors() * (std::size_t)getNumDataPatches()
             * (std::size_t)getPatchSizeOverall() * sizeof(float);
      mDeviceData = mCudaDevice->createBuffer(size, &description);
      pvAssert(mDeviceData);
#ifdef PV_USE_CUDNN
      mCUDNNData = mCudaDevice->createBuffer(size, &description);
#endif
      // no point in copying weights to device yet; they aren't set until the InitializeState stage.
   }
}
#endif // PV_USE_CUDA

void Weights::checkpointWeightPvp(
      Checkpointer *checkpointer,
      char const *bufferName,
      bool compressFlag) {
   auto checkpointEntry = std::make_shared<CheckpointEntryWeightPvp>(
         getName(), bufferName, checkpointer->getMPIBlock(), this, compressFlag);
   bool registerSucceeded =
         checkpointer->registerCheckpointEntry(checkpointEntry, !mWeightsArePlastic);
   FatalIf(
         !registerSucceeded,
         "%s failed to register %s for checkpointing.\n",
         getName().c_str(),
         bufferName);
}

void Weights::initNumDataPatches() {
   if (mSharedFlag) {
      setNumDataPatches(
            mGeometry->getNumKernelsX(), mGeometry->getNumKernelsY(), mGeometry->getNumKernelsF());
   }
   else {
      setNumDataPatches(
            mGeometry->getNumPatchesX(), mGeometry->getNumPatchesY(), mGeometry->getNumPatchesF());
   }
}

void Weights::setNumDataPatches(int numDataPatchesX, int numDataPatchesY, int numDataPatchesF) {
   mNumDataPatchesX = numDataPatchesX;
   mNumDataPatchesY = numDataPatchesY;
   mNumDataPatchesF = numDataPatchesF;
}

Patch const &Weights::getPatch(int patchIndex) const { return mGeometry->getPatch(patchIndex); }

float *Weights::getData(int arbor) { return &mData[arbor][0]; }

float const *Weights::getDataReadOnly(int arbor) const { return &mData[arbor][0]; }

float *Weights::getDataFromDataIndex(int arbor, int dataIndex) {
   int numItemsPerPatch = getPatchSizeX() * getPatchSizeY() * getPatchSizeF();
   return &mData[arbor][dataIndex * numItemsPerPatch];
}

float *Weights::getDataFromPatchIndex(int arbor, int patchIndex) {
   int dataIndex = mSharedFlag ? dataIndexLookupTable[patchIndex] : patchIndex;
   return getDataFromDataIndex(arbor, dataIndex);
}

int Weights::calcDataIndexFromPatchIndex(int patchIndex) const {
   if (getSharedFlag()) {
      int numPatchesX = mGeometry->getNumPatchesX();
      int numPatchesY = mGeometry->getNumPatchesY();
      int numPatchesF = mGeometry->getNumPatchesF();
      int xIndex      = kxPos(patchIndex, numPatchesX, numPatchesY, numPatchesF);
      xIndex          = (xIndex - mGeometry->getPreLoc().halo.lt) % mNumDataPatchesX;
      if (xIndex < 0) {
         xIndex += mNumDataPatchesX;
      }

      int yIndex = kyPos(patchIndex, numPatchesX, numPatchesY, numPatchesF);
      yIndex     = (yIndex - mGeometry->getPreLoc().halo.up) % mNumDataPatchesY;
      if (yIndex < 0) {
         yIndex += mNumDataPatchesY;
      }

      int fIndex = featureIndex(patchIndex, numPatchesX, numPatchesY, numPatchesF);

      int dataIndex =
            kIndex(xIndex, yIndex, fIndex, mNumDataPatchesX, mNumDataPatchesY, mNumDataPatchesF);
      return dataIndex;
   }
   else {
      return patchIndex;
   }
}

float Weights::calcMinWeight() {
   float minWeight = FLT_MAX;
   for (int a = 0; a < mNumArbors; a++) {
      float arborMin = calcMinWeight(a);
      if (arborMin < minWeight) {
         minWeight = arborMin;
      }
   }
   return minWeight;
}

float Weights::calcMinWeight(int arbor) {
   float arborMin = FLT_MAX;
   if (getSharedFlag()) {
      for (auto &w : mData[arbor]) {
         if (w < arborMin) {
            arborMin = w;
         }
      }
   }
   else {
      pvAssert(getNumDataPatches() == getGeometry()->getNumPatches());
      for (int p = 0; p < getNumDataPatches(); p++) {
         Patch const &patch = getPatch(p);
         int const nk       = patch.nx * getPatchSizeF();
         int const sy       = getPatchSizeX() * getPatchSizeF();
         for (int y = 0; y < patch.ny; y++) {
            for (int k = 0; k < nk; k++) {
               float w = getDataFromDataIndex(arbor, p)[patch.offset + y * sy + k];
               if (w < arborMin) {
                  arborMin = w;
               }
            }
         }
      }
   }
   return arborMin;
}

float Weights::calcMaxWeight() {
   float maxWeight = -FLT_MAX;
   for (int a = 0; a < mNumArbors; a++) {
      float arborMax = calcMaxWeight(a);
      if (arborMax > maxWeight) {
         maxWeight = arborMax;
      }
   }
   return maxWeight;
}

float Weights::calcMaxWeight(int arbor) {
   float arborMax = -FLT_MAX;
   if (getSharedFlag()) {
      for (auto &w : mData[arbor]) {
         if (w > arborMax) {
            arborMax = w;
         }
      }
   }
   else {
      pvAssert(getNumDataPatches() == getGeometry()->getNumPatches());
      for (int p = 0; p < getNumDataPatches(); p++) {
         Patch const &patch = getPatch(p);
         int const nk       = patch.nx * getPatchSizeF();
         int const sy       = getPatchSizeX() * getPatchSizeF();
         for (int y = 0; y < patch.ny; y++) {
            for (int k = 0; k < nk; k++) {
               float w = getDataFromDataIndex(arbor, p)[patch.offset + y * sy + k];
               if (w > arborMax) {
                  arborMax = w;
               }
            }
         }
      }
   }
   return arborMax;
}

#ifdef PV_USE_CUDA
void Weights::copyToGPU() {
   if (!(mUsingGPUFlag and mTimestampGPU < mTimestamp)) {
      return;
   }
   pvAssert(mDeviceData);

   int const numDataPatches    = mNumDataPatchesX * mNumDataPatchesY * mNumDataPatchesF;
   std::size_t const arborSize = (std::size_t)numDataPatches * (std::size_t)getPatchSizeOverall();
   std::size_t const numArbors = (std::size_t)mNumArbors;
   for (std::size_t a = 0; a < numArbors; a++) {
      mDeviceData->copyToDevice(mData[a].data(), arborSize * sizeof(mData[a][0]), a * arborSize);
   }
#ifdef PV_USE_CUDNN
   mCUDNNData->permuteWeightsPVToCudnn(
         mDeviceData->getPointer(),
         mNumArbors,
         numDataPatches,
         getPatchSizeX(),
         getPatchSizeY(),
         getPatchSizeF());
#endif // PV_USE_CUDNN
   mTimestampGPU = mTimestamp;
}
#endif // PV_USE_CUDA

} // end namespace PV
