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
   gaussianRandState = NULL; // Don't delete; base class deletes randState, which gaussianRandState is effectively a dynamic_cast of.
}

int InitGaussianRandomWeights::initialize_base() {
   gaussianRandState = NULL;
   return PV_SUCCESS;
}

InitWeightsParams * InitGaussianRandomWeights::createNewWeightParams(HyPerConn * callingConn) {
   InitWeightsParams * tempPtr = new InitGaussianRandomWeightsParams(callingConn);
   return tempPtr;
}

int InitGaussianRandomWeights::initRNGs(HyPerConn * conn, bool isKernel) {
   assert(randState==NULL && gaussianRandState==NULL);
   int status = PV_SUCCESS;
   if (isKernel) {
      gaussianRandState = new GaussianRandom(conn->getParent(), conn->getNumDataPatches());
   }
   else {
      gaussianRandState = new GaussianRandom(conn->getParent(), conn->preSynapticLayer()->getLayerLoc(), true/*isExtended*/);
   }
   if (randState == NULL) {
      fprintf(stderr, "InitRandomWeights error in rank %d process: unable to create object of class Random.\n", conn->getParent()->columnId());
      exit(EXIT_FAILURE);
   }
   randState = (Random *) gaussianRandState;
   return status;
}


/**
 * randomWeights() fills the full-size patch with random numbers, whether or not the patch is shrunken.
 */
int InitGaussianRandomWeights::randomWeights(pvdata_t * patchDataStart, InitWeightsParams *weightParams, int patchIndex) {
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

   const int patchSize = nxp*nyp*nfp;
   for (int n=0; n<patchSize; n++) {
      patchDataStart[n] = gaussianRandState->gaussianDist(patchIndex, mean, stdev);
   }

   return 0;
}

} /* namespace PV */
