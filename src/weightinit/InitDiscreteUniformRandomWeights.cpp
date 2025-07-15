/*
 * InitDiscreteUniformRandomWeights.cpp
 *
 *  Created on: Sep 27, 2022
 *      Author: peteschultz
 */

#include "InitDiscreteUniformRandomWeights.hpp"

namespace PV {

InitDiscreteUniformRandomWeights::InitDiscreteUniformRandomWeights(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   initialize(paramsIO, comm);
}

InitDiscreteUniformRandomWeights::InitDiscreteUniformRandomWeights() {}

InitDiscreteUniformRandomWeights::~InitDiscreteUniformRandomWeights() {}

void InitDiscreteUniformRandomWeights::initialize(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   InitRandomWeights::initialize(paramsIO, comm);
}

int InitDiscreteUniformRandomWeights::ioParamsFillGroup(ParamsIOSwitch ioSwitch) {
   int status = InitRandomWeights::ioParamsFillGroup(ioSwitch);
   ioParam_wMin(ioSwitch);
   ioParam_wMax(ioSwitch);
   ioParam_wNumValues(ioSwitch);
   FatalIf(
         mWMax < mWMin,
         "%s has wMax=%f less than wMin=%f.\n",
         getDescription().c_str(),
         (double)mWMax,
         (double)mWMin);
   return status;
}

void InitDiscreteUniformRandomWeights::ioParam_wMin(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "wMin", &mWMin);
}

void InitDiscreteUniformRandomWeights::ioParam_wMax(ParamsIOSwitch ioSwitch) {
   pvAssert(!mParamsIO->presentAndNotBeenRead("wMin"));
   mParamsIO->ioParam(ioSwitch, "wMax", &mWMax);
   FatalIf(
         mWMax <= mWMin,
         "%s \"%s\" with UniformRandomV has wMax = %f <= wMin = %f\n",
         mParamsIO->getKeyword(), mParamsIO->getName(), (double)mWMax, (double)mWMin);
}

void InitDiscreteUniformRandomWeights::ioParam_wNumValues(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "numValues", &mNumValues);
   FatalIf(
         mNumValues < 2,
         "%s parameter \"numValues\" is %d, but it must be at least 2.\n",
         getDescription().c_str(),
         mNumValues);
}

/**
 * randomWeights() fills the full-size patch with random numbers, whether or not the patch is
 * shrunken.
 */
void InitDiscreteUniformRandomWeights::randomWeights(float *patchDataStart, int patchIndex) {
   pvAssert(mWMax > mWMin and mNumValues >= 2); // checked when reading params

   // loop over all post-synaptic cells in patch
   const int nxp       = mWeights->getPatchSizeX();
   const int nyp       = mWeights->getPatchSizeY();
   const int nfp       = mWeights->getPatchSizeF();
   const int patchSize = nxp * nyp * nfp;

   double wMin = static_cast<double>(mWMin);
   double wMax = static_cast<double>(mWMax);
   double numValues = static_cast<double>(mNumValues);
   double dW = (wMax - wMin) / (numValues - 1.0);
   double p = numValues / (1.0 + static_cast<double>(CL_RANDOM_MAX));
   for (int n = 0; n < patchSize; n++) {
      double data = std::floor(p * static_cast<double>(mRandState->randomUInt(patchIndex)));
      pvAssert(data >= 0.0 and data < numValues and data == std::round(data));
      data = wMin + dW * data;
      patchDataStart[n] = static_cast<float>(data);
   }
}

} /* namespace PV */
