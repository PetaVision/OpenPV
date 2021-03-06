/*
 * InitGaussianRandomWeights.cpp
 *
 *  Created on: Aug 9, 2011
 *      Author: kpeterson
 */

#include "InitGaussianRandomWeights.hpp"

namespace PV {

InitGaussianRandomWeights::InitGaussianRandomWeights(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

InitGaussianRandomWeights::InitGaussianRandomWeights() {}

InitGaussianRandomWeights::~InitGaussianRandomWeights() {
   pvAssert(dynamic_cast<Random *>(mGaussianRandState) == mRandState);
   delete mGaussianRandState;
   mRandState = nullptr; // Prevents InitRandomWeights destructor from double-deleting
}

void InitGaussianRandomWeights::initialize(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   InitRandomWeights::initialize(name, params, comm);
}

int InitGaussianRandomWeights::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = InitRandomWeights::ioParamsFillGroup(ioFlag);
   ioParam_wGaussMean(ioFlag);
   ioParam_wGaussStdev(ioFlag);
   return status;
}

void InitGaussianRandomWeights::ioParam_wGaussMean(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "wGaussMean", &mWGaussMean, mWGaussMean);
}

void InitGaussianRandomWeights::ioParam_wGaussStdev(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "wGaussStdev", &mWGaussStdev, mWGaussStdev);
}

int InitGaussianRandomWeights::initRNGs(bool isKernel) {
   pvAssert(mRandState == nullptr && mGaussianRandState == nullptr);
   int status = PV_SUCCESS;
   if (isKernel) {
      mGaussianRandState = new GaussianRandom(mWeights->getNumDataPatches());
   }
   else {
      mGaussianRandState =
            new GaussianRandom(&mWeights->getGeometry()->getPreLoc(), true /*isExtended*/);
   }

   if (mGaussianRandState == nullptr) {
      Fatal().printf(
            "InitRandomWeights error in rank %d process: unable to create object of class "
            "Random.\n",
            mCommunicator->globalCommRank());
   }
   mRandState = (Random *)mGaussianRandState;
   return status;
}

/**
 * randomWeights() fills the full-size patch with random numbers, whether or not the patch is
 * shrunken.
 */
void InitGaussianRandomWeights::randomWeights(float *patchDataStart, int patchIndex) {
   const int patchSize = mWeights->getPatchSizeOverall();
   for (int n = 0; n < patchSize; n++) {
      patchDataStart[n] = mGaussianRandState->gaussianDist(patchIndex, mWGaussMean, mWGaussStdev);
   }
}

} /* namespace PV */
