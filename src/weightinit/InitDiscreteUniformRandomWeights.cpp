/*
 * InitDiscreteUniformRandomWeights.cpp
 *
 *  Created on: Sep 27, 2022
 *      Author: peteschultz
 */

#include "InitDiscreteUniformRandomWeights.hpp"

namespace PV {

InitDiscreteUniformRandomWeights::InitDiscreteUniformRandomWeights(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

InitDiscreteUniformRandomWeights::InitDiscreteUniformRandomWeights() {}

InitDiscreteUniformRandomWeights::~InitDiscreteUniformRandomWeights() {}

void InitDiscreteUniformRandomWeights::initialize(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   InitRandomWeights::initialize(name, params, comm);
}

int InitDiscreteUniformRandomWeights::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = InitRandomWeights::ioParamsFillGroup(ioFlag);
   ioParam_wMin(ioFlag);
   ioParam_wMax(ioFlag);
   ioParam_wNumValues(ioFlag);
   FatalIf(
         mWMax < mWMin,
         "%s has wMax=%f less than wMin=%f.\n",
         getDescription().c_str(),
         (double)mWMax,
         (double)mWMin);
   return status;
}

void InitDiscreteUniformRandomWeights::ioParam_wMin(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValueRequired(ioFlag, name, "wMin", &mWMin);
}

void InitDiscreteUniformRandomWeights::ioParam_wMax(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValueRequired(ioFlag, name, "wMax", &mWMax);
}

void InitDiscreteUniformRandomWeights::ioParam_wNumValues(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValueRequired(ioFlag, name, "numValues", &mNumValues);
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
