/*
 * PatchGeometry.cpp
 *
 *  Created on: Jul 21, 2017
 *      Author: Pete Schultz
 */

#include "PatchGeometry.hpp"
#include "utils/PVAssert.hpp"
#include "utils/conversions.h"
#include <cmath>
#include <cstring>
#include <sstream>
#include <stdexcept>

namespace PV {

PatchGeometry::PatchGeometry(
      std::string const &name,
      int patchSizeX,
      int patchSizeY,
      int patchSizeF,
      PVLayerLoc const *preLoc,
      PVLayerLoc const *postLoc) {
   initialize(name, patchSizeX, patchSizeY, patchSizeF, preLoc, postLoc);
}

void PatchGeometry::initialize(
      std::string const &name,
      int patchSizeX,
      int patchSizeY,
      int patchSizeF,
      PVLayerLoc const *preLoc,
      PVLayerLoc const *postLoc) {
   mPatchSizeX = patchSizeX;
   mPatchSizeY = patchSizeY;
   mPatchSizeF = patchSizeF;
   std::memcpy(&mPreLoc, preLoc, sizeof(*preLoc));
   std::memcpy(&mPostLoc, postLoc, sizeof(*postLoc));
   mSelfConnectionFlag = preLoc == postLoc;
   mNumPatchesX        = preLoc->nx + preLoc->halo.lt + preLoc->halo.rt;
   mNumPatchesY        = preLoc->ny + preLoc->halo.dn + preLoc->halo.up;
   mNumPatchesF        = preLoc->nf;

   mPatchStrideX       = patchSizeF;
   mPatchStrideY       = patchSizeX * mPatchSizeF;
   mPatchStrideF       = 1;

   try {
      verifyPatchSize();
   } catch (const std::exception &e) {
      throw std::runtime_error(name + std::string(": ") + e.what());
   }

   mPatchVector.clear();
   mGSynPatchStart.clear();
   mAPostOffset.clear();
}

void PatchGeometry::allocateDataStructures() {
   if (!mPatchVector.empty()) {
      return;
   }
   int numPatches = mNumPatchesX * mNumPatchesY * mNumPatchesF;
   mPatchVector.resize(numPatches);
   mGSynPatchStart.resize(numPatches);
   mAPostOffset.resize(numPatches);
   setPatchGeometry();
}

int PatchGeometry::verifyPatchSize(int numPreRestricted, int numPostRestricted, int patchSize) {
   int log2ScaleDiff;
   std::stringstream errMsgStream;
   if (numPreRestricted > numPostRestricted) {
      int stride = numPreRestricted / numPostRestricted;
      if (stride * numPostRestricted != numPreRestricted) {
         errMsgStream << "presynaptic ?-dimension (" << numPreRestricted << ") "
                      << "is greater than but not a multiple of "
                      << "presynaptic ?-dimension (" << numPostRestricted << ")";
      }
      else {
         log2ScaleDiff = (int)std::nearbyint(std::log2(stride));
         if (2 << (log2ScaleDiff - 1) != stride) {
            errMsgStream << "presynaptic ?-dimension (" << numPreRestricted << ") is a multiple "
                         << "of postsynaptic ?-dimension (" << numPostRestricted << ") "
                         << "but the quotient " << stride << " is not a power of 2";
         }
      }
   }
   else if (numPreRestricted < numPostRestricted) {
      int tstride = numPostRestricted / numPreRestricted;
      if (tstride * numPreRestricted != numPostRestricted) {
         errMsgStream << "postsynaptic ?-dimension (" << numPostRestricted << ") "
                      << "is greater than but not an even multiple of "
                      << "presynaptic ?-dimension (" << numPreRestricted << ")";
      }
      else if (patchSize % tstride != 0) {
         errMsgStream << "postsynaptic ?-dimension (" << numPostRestricted << ") "
                      << "is greater than presynaptic ?-dimension (" << numPreRestricted << ") "
                      << "but patch size " << patchSize << " is not a multiple of the quotient "
                      << tstride;
      }
      else {
         int negLog2ScaleDiff = (int)std::nearbyint(std::log2(tstride));
         if (2 << (negLog2ScaleDiff - 1) != tstride) {
            errMsgStream << "postsynaptic ?-dimension (" << numPostRestricted << ") is a multiple "
                         << "of presynaptic ?-dimension (" << numPreRestricted << ") "
                         << "but the quotient " << tstride << " is not a power of 2";
         }
         log2ScaleDiff = -negLog2ScaleDiff;
      }
   }
   else {
      pvAssert(numPreRestricted == numPostRestricted);
      if (patchSize % 2 != 1) {
         errMsgStream << "presynaptic and postsynaptic ?-dimensions are both equal to "
                      << numPreRestricted << ", but patch size " << patchSize << " is not odd";
      }
      log2ScaleDiff = 0;
   }
   std::string errorMessage(errMsgStream.str());
   if (!errorMessage.empty()) {
      throw std::runtime_error(errorMessage);
   }

   return log2ScaleDiff;
}

void PatchGeometry::verifyPatchSize() {
   std::string errorMessage;
   try {
      mLog2ScaleDiffX = verifyPatchSize(mPreLoc.nx, mPostLoc.nx, mPatchSizeX);
   } catch (std::exception const &e) {
      errorMessage                = e.what();
      std::size_t questionmarkpos = (std::size_t)0;
      while ((questionmarkpos = errorMessage.find("?", questionmarkpos)) != std::string::npos) {
         errorMessage.replace(questionmarkpos, (std::size_t)1, "x");
      }
      throw std::runtime_error(errorMessage);
   }
   try {
      mLog2ScaleDiffY = verifyPatchSize(mPreLoc.ny, mPostLoc.ny, mPatchSizeY);
   } catch (std::exception const &e) {
      errorMessage                = e.what();
      std::size_t questionmarkpos = (std::size_t)0;
      while ((questionmarkpos = errorMessage.find("?", questionmarkpos)) != std::string::npos) {
         errorMessage.replace(questionmarkpos, (std::size_t)1, "y");
      }
      throw std::runtime_error(errorMessage);
   }
   if (mPatchSizeF != mPostLoc.nf) {
      std::stringstream errMsgStream;
      errMsgStream << "number of features in patch (" << mPatchSizeF << ") "
                   << "must equal the number of postsynaptic features (" << mPostLoc.nf << ")";
      std::string errorMessage(errMsgStream.str());
      throw std::runtime_error(errorMessage);
   }
}

void PatchGeometry::setPatchGeometry() {
   int numPatches = (int)mPatchVector.size();
   pvAssert(numPatches == mNumPatchesX * mNumPatchesY * mNumPatchesF);

   std::vector<int> patchStartX(mNumPatchesX);
   std::vector<int> patchDimX(mNumPatchesX);
   std::vector<int> postStartRestrictedX(mNumPatchesX);
   std::vector<int> postStartExtendedX(mNumPatchesX);

   for (int xIndex = 0; xIndex < mNumPatchesX; xIndex++) {
      calcPatchData(
            xIndex,
            mPreLoc.nx,
            mPreLoc.halo.lt,
            mPreLoc.halo.rt,
            mPostLoc.nx,
            mPostLoc.halo.dn,
            mPostLoc.halo.up,
            mPatchSizeX,
            &patchDimX[xIndex],
            &patchStartX[xIndex],
            &postStartRestrictedX[xIndex],
            &postStartExtendedX[xIndex]);
   }

   std::vector<int> patchStartY(mNumPatchesY);
   std::vector<int> patchDimY(mNumPatchesY);
   std::vector<int> postStartRestrictedY(mNumPatchesY);
   std::vector<int> postStartExtendedY(mNumPatchesY);

   for (int yIndex = 0; yIndex < mNumPatchesY; yIndex++) {
      calcPatchData(
            yIndex,
            mPreLoc.ny,
            mPreLoc.halo.dn,
            mPreLoc.halo.up,
            mPostLoc.ny,
            mPostLoc.halo.dn,
            mPostLoc.halo.up,
            mPatchSizeY,
            &patchDimY[yIndex],
            &patchStartY[yIndex],
            &postStartRestrictedY[yIndex],
            &postStartExtendedY[yIndex]);
   }

   for (int patchIndex = 0; patchIndex < numPatches; patchIndex++) {
      Patch &patch = mPatchVector[patchIndex];

      int xIndex = kxPos(patchIndex, mNumPatchesX, mNumPatchesY, mNumPatchesF);
      patch.nx   = patchDimX[xIndex];

      int yIndex = kyPos(patchIndex, mNumPatchesX, mNumPatchesY, mNumPatchesF);
      patch.ny   = patchDimY[yIndex];

      patch.offset = kIndex(
            patchStartX[xIndex], patchStartY[yIndex], 0, mPatchSizeX, mPatchSizeY, mPatchSizeF);

      int startX                  = postStartRestrictedX[xIndex];
      int startY                  = postStartRestrictedY[yIndex];
      int nxPost                  = mPostLoc.nx;
      int nyPost                  = mPostLoc.ny;
      int nfPost                  = mPostLoc.nf;
      mGSynPatchStart[patchIndex] = kIndex(startX, startY, 0, nxPost, nyPost, nfPost);

      int startXExt            = postStartExtendedX[xIndex];
      int startYExt            = postStartExtendedY[yIndex];
      int nxExtPost            = mPostLoc.nx + mPostLoc.halo.lt + mPostLoc.halo.rt;
      int nyExtPost            = mPostLoc.ny + mPostLoc.halo.dn + mPostLoc.halo.up;
      mAPostOffset[patchIndex] = kIndex(startXExt, startYExt, 0, nxExtPost, nyExtPost, nfPost);
   }
}

void PatchGeometry::calcPatchData(
      int index,
      int numPreRestricted,
      int preStartBorder,
      int preEndBorder,
      int numPostRestricted,
      int postStartBorder,
      int postEndBorder,
      int patchSize,
      int *patchDim,
      int *patchStart,
      int *postPatchStartRestricted,
      int *postPatchStartExtended) {
   int lPatchDim   = patchSize;
   int lPatchStart = 0;
   int lPostPatchStartRes;
   int restrictedIndex = index - preStartBorder;

   if (numPreRestricted > numPostRestricted) {
      int stride = numPreRestricted / numPostRestricted;
      pvAssert(stride * numPostRestricted == numPreRestricted);
      pvAssert(stride % 2 == 0);
      int halfstride = stride / 2;

      // Use floating-point division with floor because integer division of a negative number
      // is defined inconveniently in C++.
      float fStride        = (float)stride;
      float fPatchStartPre = (float)(restrictedIndex - halfstride * (patchSize - 1));
      lPostPatchStartRes   = (int)std::floor(fPatchStartPre / fStride);
   }
   else if (numPreRestricted < numPostRestricted) {
      int tstride = numPostRestricted / numPreRestricted;
      pvAssert(tstride * numPreRestricted == numPostRestricted);
      pvAssert(tstride % 2 == 0);
      pvAssert(patchSize % tstride == 0);
      lPostPatchStartRes = restrictedIndex * tstride - (patchSize - tstride) / 2;
   }
   else {
      pvAssert(numPreRestricted == numPostRestricted);
      pvAssert(patchSize % 2 == 1);
      lPostPatchStartRes = restrictedIndex - (patchSize - 1) / 2;
   }
   int lPostPatchEndRes = lPostPatchStartRes + patchSize;

   if (lPostPatchEndRes < 0) {
      int excess = -lPostPatchEndRes;
      lPostPatchStartRes += excess;
      lPostPatchEndRes = 0;
   }

   if (lPostPatchStartRes > numPostRestricted) {
      int excess         = lPostPatchStartRes - numPostRestricted;
      lPostPatchStartRes = numPostRestricted;
      lPostPatchEndRes -= excess;
   }

   if (lPostPatchStartRes < 0) {
      int excess         = -lPostPatchStartRes;
      lPostPatchStartRes = 0;
      lPatchDim -= excess;
      lPatchStart = excess;
   }

   if (lPostPatchEndRes > numPostRestricted) {
      int excess       = lPostPatchEndRes - numPostRestricted;
      lPostPatchEndRes = numPostRestricted;
      lPatchDim -= excess;
   }

   if (lPatchDim < 0) {
      lPatchDim = 0;
   }

   pvAssert(lPatchDim >= 0);
   pvAssert(lPatchStart >= 0);
   pvAssert(lPatchStart + lPatchDim <= patchSize);
   pvAssert(lPostPatchStartRes >= 0);
   pvAssert(lPostPatchStartRes <= lPostPatchEndRes);
   pvAssert(lPostPatchEndRes <= numPostRestricted);

   *patchDim                 = lPatchDim;
   *patchStart               = lPatchStart;
   *postPatchStartRestricted = lPostPatchStartRes;
   *postPatchStartExtended   = lPostPatchStartRes + postStartBorder;
}

} // end namespace PV
