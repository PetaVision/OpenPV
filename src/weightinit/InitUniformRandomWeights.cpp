/*
 * InitUniformRandomWeights.cpp
 *
 *  Created on: Aug 9, 2011
 *      Author: kpeterson
 */

#include "InitUniformRandomWeights.hpp"
#include "InitUniformRandomWeightsParams.hpp"

namespace PV {

InitUniformRandomWeights::InitUniformRandomWeights(char const *name, HyPerCol *hc) {
   initialize_base();
   initialize(name, hc);
}

InitUniformRandomWeights::InitUniformRandomWeights() { initialize_base(); }

InitUniformRandomWeights::~InitUniformRandomWeights() {}

int InitUniformRandomWeights::initialize_base() { return PV_SUCCESS; }

int InitUniformRandomWeights::initialize(char const *name, HyPerCol *hc) {
   int status = InitRandomWeights::initialize(name, hc);
   return status;
}

InitWeightsParams *InitUniformRandomWeights::createNewWeightParams() {
   InitWeightsParams *tempPtr = new InitUniformRandomWeightsParams(name, parent);
   return tempPtr;
}

/**
 * randomWeights() fills the full-size patch with random numbers, whether or not the patch is
 * shrunken.
 */
int InitUniformRandomWeights::randomWeights(
      float *patchDataStart,
      InitWeightsParams *weightParams,
      int patchIndex) {

   InitUniformRandomWeightsParams *weightParamPtr =
         dynamic_cast<InitUniformRandomWeightsParams *>(weightParams);

   if (weightParamPtr == NULL) {
      Fatal().printf("Failed to recast pointer to weightsParam!  Exiting...");
   }

   double minwgt        = weightParamPtr->getWMin();
   double maxwgt        = weightParamPtr->getWMax();
   float sparseFraction = weightParamPtr->getSparseFraction();

   double p;
   if (maxwgt <= minwgt) {
      if (maxwgt < minwgt) {
         WarnLog().printf(
               "uniformWeights maximum less than minimum.  Changing max = %f to min value of %f\n",
               maxwgt,
               minwgt);
         maxwgt = minwgt;
      }
      p = 0;
   }
   else {
      p = (maxwgt - minwgt) / (1.0 + (double)CL_RANDOM_MAX);
   }
   sparseFraction *= (float)(1.0 + (double)CL_RANDOM_MAX);

   // loop over all post-synaptic cells in patch

   const int nxp       = weightParamPtr->getnxPatch();
   const int nyp       = weightParamPtr->getnyPatch();
   const int nfp       = weightParamPtr->getnfPatch();
   const int patchSize = nxp * nyp * nfp;

   // Never allocate an all zero patch
   int zeroesLeft = patchSize - 1; 
   // Start from a random index so that we don't always run out of zeros in the same place
   int startIndex = randomUInt(patchIndex);

   for (int n = 0; n < patchSize; n++) {
      float data = (float)(minwgt + (p * (double)randState->randomUInt(patchIndex)));
      if (zeroesLeft > 0 && (double)randState->randomUInt(patchIndex) < (double)sparseFraction) {
         data = 0.0f;
         --zeroesLeft;
      }
      patchDataStart[(n + startIndex) % patchSize] = data;
   }

   return PV_SUCCESS;
}

} /* namespace PV */
