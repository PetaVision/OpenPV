/*
 * InitUniformRandomWeights.cpp
 *
 *  Created on: Aug 9, 2011
 *      Author: kpeterson
 */

#include "InitUniformRandomWeights.hpp"

namespace PV {

InitUniformRandomWeights::InitUniformRandomWeights(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   initialize(paramsIO, comm);
}

InitUniformRandomWeights::InitUniformRandomWeights() {}

InitUniformRandomWeights::~InitUniformRandomWeights() {}

void InitUniformRandomWeights::initialize(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   InitRandomWeights::initialize(paramsIO, comm);
}

int InitUniformRandomWeights::ioParamsFillGroup(ParamsIOSwitch ioSwitch) {
   int status = InitRandomWeights::ioParamsFillGroup(ioSwitch);
   ioParam_wMinInit(ioSwitch);
   ioParam_wMaxInit(ioSwitch);
   ioParam_sparseFraction(ioSwitch);
   ioParam_minNNZ(ioSwitch);
   return status;
}

void InitUniformRandomWeights::ioParam_wMinInit(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "wMinInit", &mWMin);
}

void InitUniformRandomWeights::ioParam_wMaxInit(ParamsIOSwitch ioSwitch) {
   pvAssert(!mParamsIO->presentAndNotBeenRead("wMinInit"));
   mParamsIO->ioParam(ioSwitch, "wMaxInit", &mWMax);
   FatalIf(
         mWMax < mWMin,
         "%s \"%s\" with UniformRandomV has wMaxInit = %f < wMinInit = %f\n",
         mParamsIO->getKeyword(), mParamsIO->getName(), (double)mWMax, (double)mWMin);
}

void InitUniformRandomWeights::ioParam_sparseFraction(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "sparseFraction", &mSparseFraction);
}

void InitUniformRandomWeights::ioParam_minNNZ(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "minNNZ", &mMinNNZ);
}

/**
 * randomWeights() fills the full-size patch with random numbers, whether or not the patch is
 * shrunken.
 */
void InitUniformRandomWeights::randomWeights(float *patchDataStart, int patchIndex) {
   pvAssert(mWMax >= mWMin); // checked when reading params
   double p = (double)(mWMax - mWMin) / (1.0 + (double)CL_RANDOM_MAX);
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
