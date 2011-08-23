/*
 * InitUniformWeights.cpp
 *
 *  Created on: Aug 23, 2011
 *      Author: kpeterson
 */

#include "InitUniformWeights.hpp"
#include "InitUniformWeightsParams.hpp"

namespace PV {

InitUniformWeights::InitUniformWeights()
{
   initialize_base();
}

InitUniformWeights::~InitUniformWeights()
{
   // TODO Auto-generated destructor stub
}

int InitUniformWeights::initialize_base() {
   return PV_SUCCESS;
}

InitWeightsParams * InitUniformWeights::createNewWeightParams(HyPerConn * callingConn) {
   InitWeightsParams * tempPtr = new InitUniformWeightsParams(callingConn);
   return tempPtr;
}

int InitUniformWeights::calcWeights(PVPatch * patch, int patchIndex,
      InitWeightsParams *weightParams) {
   InitUniformWeightsParams *weightParamPtr = dynamic_cast<InitUniformWeightsParams*>(weightParams);

   if(weightParamPtr==NULL) {
      fprintf(stderr, "Failed to recast pointer to weightsParam!  Exiting...");
      exit(1);
   }

   const float iWeight = weightParamPtr->getInitWeight();


   uniformWeights(patch, iWeight);
   return 1;
}

/**
 * Initializes all weights to iWeight
 *
 */
int InitUniformWeights::uniformWeights(PVPatch * wp, float iWeight) {
      // changed variable names to avoid confusion with data members this->wMin and this->wMax
   pvdata_t * w = wp->data;

   const int nxp = wp->nx;
   const int nyp = wp->ny;
   const int nfp = wp->nf;

   const int sxp = wp->sx;
   const int syp = wp->sy;
   const int sfp = wp->sf;

   // loop over all post-synaptic cells in patch
   for (int y = 0; y < nyp; y++) {
      for (int x = 0; x < nxp; x++) {
         for (int f = 0; f < nfp; f++) {
            w[x * sxp + y * syp + f * sfp] = iWeight;
         }
      }
   }

   return PV_SUCCESS;
}

} /* namespace PV */
