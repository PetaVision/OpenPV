/*
 * Weights.cpp
 *
 *  Created on: Jul 21, 2017
 *      Author: Pete Schultz
 */

#include "Weights.hpp"
#include "utils/PVAssert.hpp"
#include "utils/conversions.h"
#include <cstring>
#include <stdexcept>

namespace PV {

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
   auto geometry =
         std::make_shared<PatchGeometry>(name, patchSizeX, patchSizeY, patchSizeF, preLoc, postLoc);
   initialize(name, geometry, numArbors, sharedWeights, timestamp);
}

Weights::Weights(
      std::string const &name,
      std::shared_ptr<PatchGeometry> geometry,
      int numArbors,
      bool sharedWeights,
      double timestamp) {
   initialize(name, geometry, numArbors, sharedWeights, timestamp);
}

Weights::Weights(std::string const &name, Weights const *baseWeights) {
   auto geometry = baseWeights->getGeometry();
   initialize(
         name,
         geometry,
         baseWeights->getNumArbors(),
         baseWeights->getSharedFlag(),
         baseWeights->getTimestamp());
}

void Weights::initialize(
      std::string const &name,
      std::shared_ptr<PatchGeometry> geometry,
      int numArbors,
      bool sharedWeights,
      double timestamp) {
   mName       = name;
   mGeometry   = geometry;
   mNumArbors  = numArbors;
   mSharedFlag = sharedWeights;
   mTimestamp  = timestamp;

   initNumDataPatches();
}

void Weights::allocateDataStructures() {
   if (!mData.empty()) {
      return;
   }
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

void Weights::initNumDataPatches() {
   if (mSharedFlag) {
      if (mGeometry->getPreLoc().nx <= mGeometry->getPostLoc().nx) {
         mNumDataPatchesX = 1;
      }
      else {
         mNumDataPatchesX = mGeometry->getPreLoc().nx / mGeometry->getPostLoc().nx;
         pvAssert(mNumDataPatchesX * mGeometry->getPostLoc().nx == mGeometry->getPreLoc().nx);
      }
      if (mGeometry->getPreLoc().ny <= mGeometry->getPostLoc().ny) {
         mNumDataPatchesY = 1;
      }
      else {
         mNumDataPatchesY = mGeometry->getPreLoc().ny / mGeometry->getPostLoc().ny;
         pvAssert(mNumDataPatchesY * mGeometry->getPostLoc().ny == mGeometry->getPreLoc().ny);
      }
   }
   else {
      mNumDataPatchesX = mGeometry->getNumPatchesX();
      mNumDataPatchesY = mGeometry->getNumPatchesY();
   }
   mNumDataPatchesF = mGeometry->getPreLoc().nf;
}

void Weights::setNumDataPatches(int numDataPatchesX, int numDataPatchesY, int numDataPatchesF) {
   mNumDataPatchesX = numDataPatchesX;
   mNumDataPatchesY = numDataPatchesY;
   mNumDataPatchesF = numDataPatchesF;
}

Patch const &Weights::getPatch(int patchIndex) const { return mGeometry->getPatch(patchIndex); }

float *Weights::getData(int arbor) { return &mData.at(arbor)[0]; }

float *Weights::getDataFromDataIndex(int arbor, int dataIndex) {
   int numItemsPerPatch = getPatchSizeX() * getPatchSizeY() * getPatchSizeF();
   return &mData.at(arbor).at(dataIndex * numItemsPerPatch);
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
      int arborMin = calcMinWeight(a);
      if (arborMin < minWeight) {
         minWeight = arborMin;
      }
   }
   return minWeight;
}

float Weights::calcMinWeight(int arbor) {
   float arborMin = FLT_MAX;
   if (getSharedFlag()) {
      for (auto &w : mData.at(arbor)) {
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
      int arborMax = calcMaxWeight(a);
      if (arborMax > maxWeight) {
         maxWeight = arborMax;
      }
   }
   return maxWeight;
}

float Weights::calcMaxWeight(int arbor) {
   float arborMax = -FLT_MAX;
   if (getSharedFlag()) {
      for (auto &w : mData.at(arbor)) {
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
