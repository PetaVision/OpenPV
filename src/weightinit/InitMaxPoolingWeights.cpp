/*
 * InitMaxPoolingWeights.cpp
 *
 *  Created on: Jan 14, 2015
 *      Author: gkenyon
 */

#include "InitMaxPoolingWeights.hpp"
#include "InitMaxPoolingWeightsParams.hpp"

namespace PV {

InitMaxPoolingWeights::InitMaxPoolingWeights(const char * name, HyPerCol * hc)
{
   initialize_base();
   initialize(name, hc);
}

InitMaxPoolingWeights::InitMaxPoolingWeights()
{
   initialize_base();
}

InitMaxPoolingWeights::~InitMaxPoolingWeights()
{
}

int InitMaxPoolingWeights::initialize_base() {
   return PV_SUCCESS;
}

int InitMaxPoolingWeights::initialize(const char * name, HyPerCol * hc) {
   int status = InitWeights::initialize(name, hc);
   return status;
}

InitWeightsParams * InitMaxPoolingWeights::createNewWeightParams() {
   InitWeightsParams * tempPtr = new InitMaxPoolingWeightsParams(name, parent);
   return tempPtr;
}

int InitMaxPoolingWeights::calcWeights(/* PVPatch * patch */ pvdata_t * dataStart, int patchIndex, int arborId) {
   InitMaxPoolingWeightsParams *weightParamPtr = dynamic_cast<InitMaxPoolingWeightsParams*>(weightParams);

   if(weightParamPtr==NULL) {
      pvError().printf("Failed to recast pointer to weightsParam!  Exiting...");
   }

   return PV_SUCCESS; 
}

} /* namespace PV */
