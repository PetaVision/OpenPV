/*
 * InitGaussianRandomWeights.cpp
 *
 *  Created on: Aug 9, 2011
 *      Author: kpeterson
 */

#include "InitGaussianRandomWeights.hpp"
#include "InitGaussianRandomWeightsParams.hpp"

namespace PV {

InitGaussianRandomWeights::InitGaussianRandomWeights(char const * name, HyPerCol * hc) {
   initialize_base();
   initialize(name, hc);
}

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

int InitGaussianRandomWeights::initialize(char const * name, HyPerCol * hc) {
   int status = InitRandomWeights::initialize(name, hc);
   return status;
}

InitWeightsParams * InitGaussianRandomWeights::createNewWeightParams() {
   InitWeightsParams * tempPtr = new InitGaussianRandomWeightsParams(name, parent);
   return tempPtr;
}

int InitGaussianRandomWeights::initRNGs(bool isKernel) {
   assert(randState==NULL && gaussianRandState==NULL);
   int status = PV_SUCCESS;
   if (isKernel) {
      gaussianRandState = new GaussianRandom(callingConn->getParent(), callingConn->getNumDataPatches());
   }
   else {
      gaussianRandState = new GaussianRandom(callingConn->getParent(), callingConn->preSynapticLayer()->getLayerLoc(), true/*isExtended*/);
   }

   if (gaussianRandState == NULL) {
      fprintf(stderr, "InitRandomWeights error in rank %d process: unable to create object of class Random.\n", callingConn->getParent()->columnId());
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

   const int nxp = weightParamPtr->getnxPatch();
   const int nyp = weightParamPtr->getnyPatch();
   const int nfp = weightParamPtr->getnfPatch();

   const int patchSize = nxp*nyp*nfp;
   for (int n=0; n<patchSize; n++) {
      patchDataStart[n] = gaussianRandState->gaussianDist(patchIndex, mean, stdev);
   }

   return 0;
}

BaseObject * createInitGaussianRandomWeights(char const * name, HyPerCol * hc) {
   return hc ? new InitGaussianRandomWeights(name, hc) : NULL;
}

} /* namespace PV */
