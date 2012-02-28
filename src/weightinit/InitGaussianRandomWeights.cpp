/*
 * InitGaussianRandomWeights.cpp
 *
 *  Created on: Aug 9, 2011
 *      Author: kpeterson
 */

#include "InitGaussianRandomWeights.hpp"

#include "InitGaussianRandomWeightsParams.hpp"
#include "../utils/pv_random.h"


namespace PV {

InitGaussianRandomWeights::InitGaussianRandomWeights() {
	   initialize_base();
}
//InitGaussianRandomWeights::InitGaussianRandomWeights(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
//      ChannelType channel) : InitWeights() {
//
//	InitGaussianRandomWeights::initialize_base();
//	InitGaussianRandomWeights::initialize(name, hc, pre, post, channel);
//
//}

InitGaussianRandomWeights::~InitGaussianRandomWeights() {
	// TODO Auto-generated destructor stub
}

int InitGaussianRandomWeights::initialize_base() {
   return PV_SUCCESS;
}
//int InitGaussianRandomWeights::initialize(const char * name, HyPerCol * hc,
//      HyPerLayer * pre, HyPerLayer * post, ChannelType channel) {
//   InitWeights::initialize(name, hc, pre, post, channel);
//   return PV_SUCCESS;
//}

InitWeightsParams * InitGaussianRandomWeights::createNewWeightParams(HyPerConn * callingConn) {
   InitWeightsParams * tempPtr = new InitGaussianRandomWeightsParams(callingConn);
   return tempPtr;
}

int InitGaussianRandomWeights::calcWeights(/* PVPatch * patch */ pvdata_t * dataStart, int patchIndex, int arborId,
      InitWeightsParams *weightParams) {

   InitGaussianRandomWeightsParams *weightParamPtr = dynamic_cast<InitGaussianRandomWeightsParams*>(weightParams);

   if(weightParamPtr==NULL) {
      fprintf(stderr, "Failed to recast pointer to weightsParam!  Exiting...");
      exit(1);
   }

   const float wGaussMean = weightParamPtr->getMean();
   const float wGaussStdev = weightParamPtr->getStDev();

   gaussianWeights(dataStart, wGaussMean, wGaussStdev);
   return PV_SUCCESS; // return 1;
}

/**
 * generate random weights for a patch from a gaussian distribution
 * NOTES:
 *    - the pointer w already points to the patch head in the data structure
 *    - it only sets the weights to "real" neurons, not to neurons in the boundary
 *    layer. For example, if x are boundary neurons and o are real neurons,
 *    x x x x
 *    x o o o
 *    x o o o
 *    x o o o
 *
 *    for a 4x4 connection it sets the weights to the o neurons only.
 *    .
 */
// int HyPerConn::gaussianWeights(PVPatch * wp, float mean, float stdev, int * seed)
int InitGaussianRandomWeights::gaussianWeights(/* PVPatch * wp */ pvdata_t * dataStart, float mean, float stdev) {
   // pvdata_t * w = wp->data;

   const int nxp = parentConn->xPatchSize(); // wp->nx;
   const int nyp = parentConn->yPatchSize(); // wp->ny;
   const int nfp = parentConn->fPatchSize(); //wp->nf;

   const int sxp = parentConn->xPatchStride();
   const int syp = parentConn->yPatchStride();
   const int sfp = parentConn->fPatchStride();

   // loop over all post-synaptic cells in patch
   for (int y = 0; y < nyp; y++) {
      for (int x = 0; x < nxp; x++) {
         for (int f = 0; f < nfp; f++) {
            dataStart[x * sxp + y * syp + f * sfp] = box_muller(mean,stdev);
         }
      }
   }

   return 0;
}

} /* namespace PV */
