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
      int numItemsPerPatch = getPatchSizeX() * getPatchSizeY() * getPatchSizeF();
      mData.resize(mNumArbors);
      for (int arbor = 0; arbor < mNumArbors; arbor++) {
         mData[arbor].resize(numDataPatches * numItemsPerPatch);
      }
   }
}

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
   int dataIndex = calcDataIndexFromPatchIndex(patchIndex);
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

} // end namespace PV
