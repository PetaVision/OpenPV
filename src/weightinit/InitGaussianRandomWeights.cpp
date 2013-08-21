/*
 * InitGaussianRandomWeights.cpp
 *
 *  Created on: Aug 9, 2011
 *      Author: kpeterson
 */

#include "InitGaussianRandomWeights.hpp"
#include "InitGaussianRandomWeightsParams.hpp"

namespace PV {

InitGaussianRandomWeights::InitGaussianRandomWeights() {
   initialize_base();
}

InitGaussianRandomWeights::~InitGaussianRandomWeights() {
}

int InitGaussianRandomWeights::initialize_base() {
   return PV_SUCCESS;
}

InitWeightsParams * InitGaussianRandomWeights::createNewWeightParams(HyPerConn * callingConn) {
   InitWeightsParams * tempPtr = new InitGaussianRandomWeightsParams(callingConn);
   return tempPtr;
}

/**
 * randomWeights() fills the full-size patch with random numbers, whether or not the patch is shrunken.
 */
int InitGaussianRandomWeights::randomWeights(pvdata_t * patchDataStart, InitWeightsParams *weightParams, uint4 * rnd_state) {
   InitGaussianRandomWeightsParams *weightParamPtr = dynamic_cast<InitGaussianRandomWeightsParams*>(weightParams);

   if(weightParamPtr==NULL) {
      fprintf(stderr, "Failed to recast pointer to weightsParam!  Exiting...");
      exit(1);
   }

   const float mean = weightParamPtr->getMean();
   const float stdev = weightParamPtr->getStDev();

   const int nxp = weightParamPtr->getnxPatch_tmp();
   const int nyp = weightParamPtr->getnyPatch_tmp();
   const int nfp = weightParamPtr->getnfPatch_tmp();

   const int sxp = weightParamPtr->getsx_tmp();
   const int syp = weightParamPtr->getsy_tmp();
   const int sfp = weightParamPtr->getsf_tmp();

   // loop over all post-synaptic cells in patch
   for (int y = 0; y < nyp; y++) {
      for (int x = 0; x < nxp; x++) {
         for (int f = 0; f < nfp; f++) {
            patchDataStart[x * sxp + y * syp + f * sfp] = cl_box_muller(mean,stdev,rnd_state);
         }
      }
   }

   return 0;
}

} /* namespace PV */
