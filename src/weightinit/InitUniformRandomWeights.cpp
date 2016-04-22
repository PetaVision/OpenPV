/*
 * InitUniformRandomWeights.cpp
 *
 *  Created on: Aug 9, 2011
 *      Author: kpeterson
 */

#include "InitUniformRandomWeights.hpp"
#include "InitUniformRandomWeightsParams.hpp"

namespace PV {

InitUniformRandomWeights::InitUniformRandomWeights(char const * name, HyPerCol * hc) {
   initialize_base();
   initialize(name, hc);
}

InitUniformRandomWeights::InitUniformRandomWeights() {
   initialize_base();
}

InitUniformRandomWeights::~InitUniformRandomWeights() {
}

int InitUniformRandomWeights::initialize_base() {
   return PV_SUCCESS;
}

int InitUniformRandomWeights::initialize(char const * name, HyPerCol * hc) {
   int status = InitRandomWeights::initialize(name, hc);
   return status;
}

InitWeightsParams * InitUniformRandomWeights::createNewWeightParams() {
   InitWeightsParams * tempPtr = new InitUniformRandomWeightsParams(name, parent);
   return tempPtr;
}

/**
 * randomWeights() fills the full-size patch with random numbers, whether or not the patch is shrunken.
 */
int InitUniformRandomWeights::randomWeights(pvdata_t * patchDataStart, InitWeightsParams *weightParams, int patchIndex) {

   InitUniformRandomWeightsParams *weightParamPtr = dynamic_cast<InitUniformRandomWeightsParams*>(weightParams);

   if(weightParamPtr==NULL) {
      fprintf(stderr, "Failed to recast pointer to weightsParam!  Exiting...");
      exit(1);
   }

   float minwgt = weightParamPtr->getWMin();
   float maxwgt = weightParamPtr->getWMax();
   float sparseFraction = weightParamPtr->getSparseFraction();

   double p;
   if( maxwgt <= minwgt ) {
      if( maxwgt < minwgt ) {
         fprintf(stderr, "Warning: uniformWeights maximum less than minimum.  Changing max = %f to min value of %f\n", maxwgt, minwgt);
         maxwgt = minwgt;
      }
      p = 0;
   }
   else {
       p = (maxwgt - minwgt) / (1.0+(double) CL_RANDOM_MAX);
   }
   sparseFraction *= (1.0+(double) CL_RANDOM_MAX);

   // loop over all post-synaptic cells in patch

   const int nxp = weightParamPtr->getnxPatch();
   const int nyp = weightParamPtr->getnyPatch();
   const int nfp = weightParamPtr->getnfPatch();
   const int patchSize = nxp*nyp*nfp;
   for (int n=0; n<patchSize; n++) {
      pvdata_t data = minwgt + (pvdata_t) (p * (double) randState->randomUInt(patchIndex));
      if ((double) randState->randomUInt(patchIndex) < sparseFraction) data = 0.0;
      patchDataStart[n] = data;
   }

   return PV_SUCCESS;
}

BaseObject * createInitUniformRandomWeights(char const * name, HyPerCol * hc) {
   return hc ? new InitUniformRandomWeights(name, hc) : NULL;
}

} /* namespace PV */
