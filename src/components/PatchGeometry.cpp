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

   mPatchStrideX = patchSizeF;
   mPatchStrideY = patchSizeX * mPatchSizeF;
   mPatchStrideF = 1;

   try {
      verifyPatchSize();
   } catch (const std::exception &e) {
      throw std::runtime_error(name + std::string(": ") + e.what());
   }

   mNumKernelsX = preLoc->nx > postLoc->nx ? preLoc->nx / postLoc->nx : 1;
   mNumKernelsY = preLoc->ny > postLoc->ny ? preLoc->ny / postLoc->ny : 1;
   mNumKernelsF = preLoc->nf;

   mPatchVector.clear();
   mGSynPatchStart.clear();
   mAPostOffset.clear();
   mTransposeItemIndex.clear();
}

void PatchGeometry::setMargins(PVHalo const &preHalo, PVHalo const &postHalo) {
   if (!mPatchVector.empty()) {
      // Can't change halo after allocation.
      FatalIf(
            std::memcmp(&preHalo, &mPreLoc.halo, sizeof(PVHalo))
                  or std::memcmp(&preHalo, &mPreLoc.halo, sizeof(PVHalo)),
            "Attempt to change margins of a PatchGeometry object after allocateDataStructures "
            "had been called for the same object.\n");
   }
   else {
      std::memcpy(&mPreLoc.halo, &preHalo, sizeof(PVHalo));
      std::memcpy(&mPostLoc.halo, &postHalo, sizeof(PVHalo));
      mNumPatchesX = mPreLoc.nx + mPreLoc.halo.lt + mPreLoc.halo.rt;
      mNumPatchesY = mPreLoc.ny + mPreLoc.halo.dn + mPreLoc.halo.up;
   }
}

void PatchGeometry::allocateDataStructures() {
   if (!mPatchVector.empty()) {
      return;
   }
   setPatchGeometry();
   setTransposeItemIndices();
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
   int numPatches = mNumPatchesX * mNumPatchesY * mNumPatchesF;
   mPatchVector.resize(numPatches);
   mGSynPatchStart.resize(numPatches);
   mAPostOffset.resize(numPatches);
   mUnshrunkenStart.resize(numPatches);

   std::vector<int> patchStartX(mNumPatchesX);
   std::vector<int> patchDimX(mNumPatchesX);
   std::vector<int> postStartRestrictedX(mNumPatchesX);
   std::vector<int> postStartExtendedX(mNumPatchesX);
   std::vector<int> postUnshrunkenStartX(mNumPatchesX);

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
            &postStartExtendedX[xIndex],
            &postUnshrunkenStartX[xIndex]);
   }

   std::vector<int> patchStartY(mNumPatchesY);
   std::vector<int> patchDimY(mNumPatchesY);
   std::vector<int> postStartRestrictedY(mNumPatchesY);
   std::vector<int> postStartExtendedY(mNumPatchesY);
   std::vector<int> postUnshrunkenStartY(mNumPatchesY);

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
            &postStartExtendedY[yIndex],
            &postUnshrunkenStartY[yIndex]);
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

      int startUnshrunkenX = postUnshrunkenStartX[xIndex];
      int startUnshrunkenY = postUnshrunkenStartY[yIndex];
      mUnshrunkenStart[patchIndex] =
            kIndex(startUnshrunkenX, startUnshrunkenY, 0, nxExtPost, nyExtPost, nfPost);
   }
}

void PatchGeometry::setTransposeItemIndices() {
   int const patchSizeOverall = getPatchSizeOverall();
   int const numKernels       = getNumKernels();
   mTransposeItemIndex.resize(numKernels);
   for (auto &t : mTransposeItemIndex) {
      t.resize(patchSizeOverall);
   }
   int const xStride  = mPreLoc.nx > mPostLoc.nx ? mPreLoc.nx / mPostLoc.nx : 1;
   int const yStride  = mPreLoc.ny > mPostLoc.ny ? mPreLoc.ny / mPostLoc.ny : 1;
   int const xTStride = mPostLoc.nx > mPreLoc.nx ? mPostLoc.nx / mPreLoc.nx : 1;
   int const yTStride = mPostLoc.ny > mPreLoc.ny ? mPostLoc.ny / mPreLoc.ny : 1;
   pvAssert(!(mPreLoc.nx > mPostLoc.nx) or xStride * mPostLoc.nx == mPreLoc.nx);
   pvAssert(!(mPreLoc.ny > mPostLoc.ny) or yStride * mPostLoc.ny == mPreLoc.ny);
   pvAssert(!(mPostLoc.nx > mPreLoc.nx) or xTStride * mPreLoc.nx == mPostLoc.nx);
   pvAssert(!(mPostLoc.ny > mPreLoc.ny) or yTStride * mPreLoc.ny == mPostLoc.ny);
   int const patchSizeXPre = getPatchSizeX();
   int const patchSizeYPre = getPatchSizeY();
   // Either xStride or xTStride is one, and if xTStride>1, xTStride must divide patchSizeXPre.
   // We compute both xStride and xTStride to avoid if/else if/else branching between
   // one-to-one, one-to-many, and many-to-one cases.
   int const patchSizeXPost = patchSizeXPre * xStride / xTStride;
   int const patchSizeYPost = patchSizeYPre * yStride / yTStride;
   for (int kernelIndexPre = 0; kernelIndexPre < numKernels; kernelIndexPre++) {
      for (int itemInPatchPre = 0; itemInPatchPre < patchSizeOverall; itemInPatchPre++) {
         int const kernelIndexXPre =
               kxPos(kernelIndexPre, mNumKernelsX, mNumKernelsY, mNumKernelsF);

         int const itemInPatchXPre  = kxPos(itemInPatchPre, mPatchSizeX, mPatchSizeY, mPatchSizeF);
         int const itemInPatchXConj = patchSizeXPre - 1 - itemInPatchXPre;

         // kernelStartX is nonzero only in many-to-one connections where patchSizeXPre is even.
         // In this case, the start of the patch does not line up with the start of a cell in
         // post-synapic space.
         int const extentOneSideX = (patchSizeXPre - 1) * xStride / 2;
         int kernelStartX         = (kernelIndexXPre - extentOneSideX) % xStride;
         if (kernelStartX < 0) {
            kernelStartX += xStride;
         }
         int const itemInPatchXPost = (xStride * itemInPatchXConj + kernelStartX) / xTStride;

         int const kernelIndexYPre =
               kyPos(kernelIndexPre, mNumKernelsX, mNumKernelsY, mNumKernelsF);

         int const itemInPatchYPre  = kyPos(itemInPatchPre, mPatchSizeX, mPatchSizeY, mPatchSizeF);
         int const itemInPatchYConj = patchSizeYPre - 1 - itemInPatchYPre;

         int const extentOneSideY = (patchSizeYPre - 1) * yStride / 2;
         int kernelStartY         = (kernelIndexYPre - extentOneSideY) % yStride;
         if (kernelStartY < 0) {
            kernelStartY += yStride;
         }
         int const itemInPatchYPost = (yStride * itemInPatchYConj + kernelStartY) / yTStride;

         int const kernelIndexFPre =
               featureIndex(kernelIndexPre, mNumKernelsX, mNumKernelsY, mNumKernelsF);
         int const itemInPatchFPre =
               featureIndex(itemInPatchPre, mPatchSizeX, mPatchSizeY, mPatchSizeF);

         int itemInPatchFPost = kernelIndexFPre;
         int patchSizeFPost   = mNumKernelsF;
         int itemInPatchPost  = kIndex(
               itemInPatchXPost,
               itemInPatchYPost,
               itemInPatchFPost,
               patchSizeXPost,
               patchSizeYPost,
               patchSizeFPost);
         mTransposeItemIndex[kernelIndexPre][itemInPatchPre] = itemInPatchPost;
      }
   }
}

int PatchGeometry::calcPatchStartInPost(
      int indexRestrictedPre,
      int patchSize,
      int numNeuronsPre,
      int numNeuronsPost) {
   int patchStartInPost;
   if (numNeuronsPre == numNeuronsPost) {
      int extentOneSide = (patchSize - 1) / 2;
      FatalIf(
            extentOneSide * 2 + 1 != patchSize,
            "One-to-one connection with patch size %d. One-to-one connections require an odd "
            "patch size.\n",
            patchSize);
      patchStartInPost = indexRestrictedPre - extentOneSide;
   }
   else if (numNeuronsPre < numNeuronsPost) {
      int tstride = numNeuronsPost / numNeuronsPre;
      FatalIf(
            tstride * numNeuronsPre != numNeuronsPost or tstride % 2 != 0,
            "One-to-many connection with numNeuronsPost = %d and numNeuronsPre = %d, "
            "but %d/%d is not an even integer.\n",
            numNeuronsPost,
            numNeuronsPre,
            numNeuronsPost,
            numNeuronsPre);
      FatalIf(
            patchSize % tstride != 0,
            "One-to-many connection with numPost/numPre=%d and patch size %d. One-to-many "
            "connections require the patch size be a multiple of numPost/numPre=%d/%d.\n",
            tstride,
            patchSize,
            numNeuronsPost,
            numNeuronsPre);
      int extentOneSide = (patchSize - tstride) / 2;
      patchStartInPost  = indexRestrictedPre * tstride - extentOneSide;
   }
   else {
      pvAssert(numNeuronsPre > numNeuronsPost);
      int stride = numNeuronsPre / numNeuronsPost;
      FatalIf(
            stride * numNeuronsPost != numNeuronsPre or stride % 2 != 0,
            "Many-to-one connection with numNeuronsPre = %d and numNeuronsPost = %d, "
            "but %d/%d is not an even integer.\n",
            numNeuronsPre,
            numNeuronsPost,
            numNeuronsPre,
            numNeuronsPost);
      int extentOneSide = (stride / 2) * (patchSize - 1);
      // Use floating-point division with floor because integer division of a negative number
      // is defined inconveniently in C++.
      float fStride        = (float)stride;
      float fPatchStartPre = (float)(indexRestrictedPre - extentOneSide);
      patchStartInPost     = (int)std::floor(fPatchStartPre / fStride);
   }
   return patchStartInPost;
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
      int *postPatchStartExtended,
      int *postPatchUnshrunkenStart) {
   int lPatchDim       = patchSize;
   int lPatchStart     = 0;
   int restrictedIndex = index - preStartBorder;
   int lPostPatchStartRes =
         calcPatchStartInPost(restrictedIndex, patchSize, numPreRestricted, numPostRestricted);
   *postPatchUnshrunkenStart = lPostPatchStartRes + postStartBorder;
   int lPostPatchEndRes      = lPostPatchStartRes + patchSize;

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
