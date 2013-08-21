/*
 * InitUniformRandomWeights.cpp
 *
 *  Created on: Aug 9, 2011
 *      Author: kpeterson
 */

#include "InitUniformRandomWeights.hpp"
#include "InitUniformRandomWeightsParams.hpp"

namespace PV {

InitUniformRandomWeights::InitUniformRandomWeights() {
   initialize_base();
}

InitUniformRandomWeights::~InitUniformRandomWeights() {
}

int InitUniformRandomWeights::initialize_base() {
   return PV_SUCCESS;
}

InitWeightsParams * InitUniformRandomWeights::createNewWeightParams(HyPerConn * callingConn) {
   InitWeightsParams * tempPtr = new InitUniformRandomWeightsParams(callingConn);
   return tempPtr;
}

/**
 * randomWeights() fills the full-size patch with random numbers, whether or not the patch is shrunken.
 */
int InitUniformRandomWeights::randomWeights(pvdata_t * patchDataStart, InitWeightsParams *weightParams, uint4 * rnd_state) {

   InitUniformRandomWeightsParams *weightParamPtr = dynamic_cast<InitUniformRandomWeightsParams*>(weightParams);

   if(weightParamPtr==NULL) {
      fprintf(stderr, "Failed to recast pointer to weightsParam!  Exiting...");
      exit(1);
   }

   float minwgt = weightParamPtr->getWMin();
   float maxwgt = weightParamPtr->getWMax();
   float sparseFraction = weightParamPtr->getSparseFraction();

   const int nxp = weightParamPtr->getnxPatch_tmp();
   const int nyp = weightParamPtr->getnyPatch_tmp();
   const int nfp = weightParamPtr->getnfPatch_tmp();

   const int sxp = weightParamPtr->getsx_tmp();
   const int syp = weightParamPtr->getsy_tmp();
   const int sfp = weightParamPtr->getsf_tmp();

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
   for (int y = 0; y < nyp; y++) {
      for (int x = 0; x < nxp; x++) {
         for (int f = 0; f < nfp; f++) {
            pvdata_t data = minwgt + (pvdata_t) (p * (double) rand_ul(rnd_state));
            if ((double) rand_ul(rnd_state) < sparseFraction) data = 0.0;
            patchDataStart[x * sxp + y * syp + f * sfp] = data;
         }
      }
   }

   return PV_SUCCESS;
}

unsigned int InitUniformRandomWeights::rand_ul(uint4 * state) {
   // Generates a pseudo-random number in the range 0 to UINT_MAX (usually 2^32-1)
   *state = cl_random_get(*state);
   return state->s0;
}

} /* namespace PV */
