/*
 * InitGaussianRandomWeights.cpp
 *
 *  Created on: Aug 9, 2011
 *      Author: kpeterson
 */

#include "InitGaussianRandomWeights.hpp"

namespace PV {

InitGaussianRandomWeights::InitGaussianRandomWeights(char const *name, HyPerCol *hc) {
   initialize_base();
   initialize(name, hc);
}

InitGaussianRandomWeights::InitGaussianRandomWeights() { initialize_base(); }

InitGaussianRandomWeights::~InitGaussianRandomWeights() {
   mGaussianRandState = nullptr;
   // Don't delete. base class deletes mRandState,
   // which mGaussianRandState is effectively a dynamic_cast of.
}

int InitGaussianRandomWeights::initialize_base() {
   mGaussianRandState = nullptr;
   return PV_SUCCESS;
}

int InitGaussianRandomWeights::initialize(char const *name, HyPerCol *hc) {
   int status = InitRandomWeights::initialize(name, hc);
   return status;
}

int InitGaussianRandomWeights::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = InitRandomWeights::ioParamsFillGroup(ioFlag);
   ioParam_wGaussMean(ioFlag);
   ioParam_wGaussStdev(ioFlag);
   return status;
}

void InitGaussianRandomWeights::ioParam_wGaussMean(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "wGaussMean", &mWGaussMean, mWGaussMean);
}

void InitGaussianRandomWeights::ioParam_wGaussStdev(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "wGaussStdev", &mWGaussStdev, mWGaussStdev);
}

int InitGaussianRandomWeights::initRNGs(bool isKernel) {
   pvAssert(mRandState == nullptr && mGaussianRandState == nullptr);
   int status = PV_SUCCESS;
   if (isKernel) {
      mGaussianRandState = new GaussianRandom(mCallingConn->getNumDataPatches());
   }
   else {
      mGaussianRandState = new GaussianRandom(
            mCallingConn->preSynapticLayer()->getLayerLoc(), true /*isExtended*/);
   }

   if (mGaussianRandState == nullptr) {
      Fatal().printf(
            "InitRandomWeights error in rank %d process: unable to create object of class "
            "Random.\n",
            parent->columnId());
   }
   mRandState = (Random *)mGaussianRandState;
   return status;
}

/**
 * randomWeights() fills the full-size patch with random numbers, whether or not the patch is
 * shrunken.
 */
void InitGaussianRandomWeights::randomWeights(float *patchDataStart, int patchIndex) {
   const int nxp = mCallingConn->xPatchSize();
   const int nyp = mCallingConn->yPatchSize();
   const int nfp = mCallingConn->fPatchSize();

   const int patchSize = nxp * nyp * nfp;
   for (int n = 0; n < patchSize; n++) {
      patchDataStart[n] = mGaussianRandState->gaussianDist(patchIndex, mWGaussMean, mWGaussStdev);
   }
}

} /* namespace PV */
