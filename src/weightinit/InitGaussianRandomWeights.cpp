/*
 * InitGaussianRandomWeights.cpp
 *
 *  Created on: Aug 9, 2011
 *      Author: kpeterson
 */

#include "InitGaussianRandomWeights.hpp"

namespace PV {

InitGaussianRandomWeights::InitGaussianRandomWeights(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
}

InitGaussianRandomWeights::InitGaussianRandomWeights() {}

InitGaussianRandomWeights::~InitGaussianRandomWeights() {
   pvAssert(dynamic_cast<Random *>(mGaussianRandState) == mRandState);
   delete mGaussianRandState;
   mRandState = nullptr; // Prevents InitRandomWeights destructor from double-deleting
}

void InitGaussianRandomWeights::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   InitRandomWeights::initialize(params, defaults, comm);
}

int InitGaussianRandomWeights::ioParamsFillGroup(ParamsIOSwitch ioSwitch) {
   int status = InitRandomWeights::ioParamsFillGroup(ioSwitch);
   ioParam_wGaussMean(ioSwitch);
   ioParam_wGaussStdev(ioSwitch);
   return status;
}

void InitGaussianRandomWeights::ioParam_wGaussMean(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "wGaussMean", &mWGaussMean);
}

void InitGaussianRandomWeights::ioParam_wGaussStdev(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "wGaussStdev", &mWGaussStdev);
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
