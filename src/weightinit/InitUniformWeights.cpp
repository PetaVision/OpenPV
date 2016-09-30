/*
 * InitUniformWeights.cpp
 *
 *  Created on: Aug 23, 2011
 *      Author: kpeterson
 */

#include "InitUniformWeights.hpp"
#include "InitUniformWeightsParams.hpp"

namespace PV {

InitUniformWeights::InitUniformWeights(char const * name, HyPerCol * hc)
{
   initialize_base();
   initialize(name, hc);
}

InitUniformWeights::InitUniformWeights()
{
   initialize_base();
}

InitUniformWeights::~InitUniformWeights()
{
}

int InitUniformWeights::initialize_base() {
   return PV_SUCCESS;
}

int InitUniformWeights::initialize(char const * name, HyPerCol * hc) {
   int status = InitWeights::initialize(name, hc);
   return status;
}

InitWeightsParams * InitUniformWeights::createNewWeightParams() {
   InitWeightsParams * tempPtr = new InitUniformWeightsParams(name, parent);
   return tempPtr;
}

int InitUniformWeights::calcWeights(/* PVPatch * patch */ pvdata_t * dataStart, int patchIndex, int arborId) {
   InitUniformWeightsParams *weightParamPtr = dynamic_cast<InitUniformWeightsParams*>(weightParams);

   if(weightParamPtr==NULL) {
      pvError().printf("Failed to recast pointer to weightsParam!  Exiting...");
   }

   const float iWeight = weightParamPtr->getInitWeight();
   const bool connectOnlySameFeatures = weightParamPtr->getConnectOnlySameFeatures();

   const int nfp = weightParamPtr->getnfPatch();
   const int kf = patchIndex % nfp;

   uniformWeights(dataStart, iWeight, kf, weightParamPtr, connectOnlySameFeatures);
   return PV_SUCCESS; 
}

/**
 * Initializes all weights to iWeight
 *
 */
  int InitUniformWeights::uniformWeights(pvdata_t * dataStart, float iWeight, int kf, InitUniformWeightsParams *weightParamPtr, bool connectOnlySameFeatures) {
      // changed variable names to avoid confusion with data members this->wMin and this->wMax

   const int nxp = weightParamPtr->getnxPatch(); 
   const int nyp = weightParamPtr->getnyPatch(); 
   const int nfp = weightParamPtr->getnfPatch(); 

   const int sxp = weightParamPtr->getsx(); 
   const int syp = weightParamPtr->getsy(); 
   const int sfp = weightParamPtr->getsf(); 

   // loop over all post-synaptic cells in patch
   for (int y = 0; y < nyp; y++) {
      for (int x = 0; x < nxp; x++) {
         for (int f = 0; f < nfp; f++) {
	   if ((connectOnlySameFeatures) && (kf != f)){
	     dataStart[x * sxp + y * syp + f * sfp] = 0;
	   }
	   else{
	     dataStart[x * sxp + y * syp + f * sfp] = iWeight;
	   }
         }
      }
   }

   return PV_SUCCESS;
}

} /* namespace PV */
