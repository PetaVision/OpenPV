/*
 * InitUniformRandomWeights.cpp
 *
 *  Created on: Aug 9, 2011
 *      Author: kpeterson
 */

#include "InitUniformRandomWeights.hpp"

namespace PV {

InitUniformRandomWeights::InitUniformRandomWeights(char const *name, HyPerCol *hc) {
   initialize(name, hc);
}

InitUniformRandomWeights::InitUniformRandomWeights() {}

InitUniformRandomWeights::~InitUniformRandomWeights() {}

int InitUniformRandomWeights::initialize(char const *name, HyPerCol *hc) {
   int status = InitRandomWeights::initialize(name, hc);
   return status;
}

int InitUniformRandomWeights::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = InitRandomWeights::ioParamsFillGroup(ioFlag);
   ioParam_wMinInit(ioFlag);
   ioParam_wMaxInit(ioFlag);
   ioParam_sparseFraction(ioFlag);
   ioParam_minNNZ(ioFlag);
   return status;
}

void InitUniformRandomWeights::ioParam_wMinInit(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "wMinInit", &mWMin, mWMin);
}

void InitUniformRandomWeights::ioParam_wMaxInit(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "wMaxInit", &mWMax, mWMax);
}

void InitUniformRandomWeights::ioParam_sparseFraction(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(
         ioFlag, name, "sparseFraction", &mSparseFraction, mSparseFraction);
}

void InitUniformRandomWeights::ioParam_minNNZ(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "minNNZ", &mMinNNZ, mMinNNZ);
}

/**
 * randomWeights() fills the full-size patch with random numbers, whether or not the patch is
 * shrunken.
 */
void InitUniformRandomWeights::randomWeights(float *patchDataStart, int patchIndex) {
   double p;
   if (mWMax <= mWMin) {
      if (mWMax < mWMin) {
         WarnLog().printf(
               "uniformWeights maximum less than minimum.  Changing max = %f to min value of %f\n",
               (double)mWMax,
               (double)mWMin);
         mWMax = mWMin;
      }
      p = 0.0;
   }
   else {
      p = (double)(mWMax - mWMin) / (1.0 + (double)CL_RANDOM_MAX);
   }
   float sparseFraction = mSparseFraction * (float)(1.0 + (double)CL_RANDOM_MAX);

   // loop over all post-synaptic cells in patch

   const int nxp       = mWeights->getPatchSizeX();
   const int nyp       = mWeights->getPatchSizeY();
   const int nfp       = mWeights->getPatchSizeF();
   const int patchSize = nxp * nyp * nfp;

   // Force a minimum number of nonzero weights
   int zeroesLeft = patchSize - mMinNNZ;

   // Start from a random index so that we don't always run out of zeros in the same place
   int startIndex = 0;

   // This line ensures we create the same weight patches for minNNZ = 0 as we did before
   if (mMinNNZ != 0) {
      startIndex = mRandState->randomUInt(patchIndex) % patchSize;
   }

   for (int n = 0; n < patchSize; n++) {
      float data = (mWMin + (float)(p * (double)mRandState->randomUInt(patchIndex)));
      if (zeroesLeft > 0 && (double)mRandState->randomUInt(patchIndex) < (double)sparseFraction) {
         data = 0.0f;
         --zeroesLeft;
      }
      patchDataStart[(n + startIndex) % patchSize] = data;
   }
}

} /* namespace PV */
