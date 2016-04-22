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
      fprintf(stderr, "Failed to recast pointer to weightsParam!  Exiting...");
      exit(1);
   }

   const float iWeight = weightParamPtr->getInitWeight();
   const bool connectOnlySameFeatures = weightParamPtr->getConnectOnlySameFeatures();

   const int nfp = weightParamPtr->getnfPatch();
   const int kf = patchIndex % nfp;

   uniformWeights(dataStart, iWeight, kf, weightParamPtr, connectOnlySameFeatures);
   return PV_SUCCESS; // return 1;
}

/**
 * Initializes all weights to iWeight
 *
 */
  int InitUniformWeights::uniformWeights(pvdata_t * dataStart, float iWeight, int kf, InitUniformWeightsParams *weightParamPtr, bool connectOnlySameFeatures) {
      // changed variable names to avoid confusion with data members this->wMin and this->wMax
   // pvdata_t * w = wp->data;

   const int nxp = weightParamPtr->getnxPatch(); // wp->nx;
   const int nyp = weightParamPtr->getnyPatch(); // wp->ny;
   const int nfp = weightParamPtr->getnfPatch(); //wp->nf;

   const int sxp = weightParamPtr->getsx(); //wp->sx;
   const int syp = weightParamPtr->getsy(); //wp->sy;
   const int sfp = weightParamPtr->getsf(); //wp->sf;

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

BaseObject * createInitUniformWeights(char const * name, HyPerCol * hc) {
   return hc ? new InitUniformWeights(name, hc) : NULL;
}

} /* namespace PV */
