/*
 * InitIdentWeights.cpp
 *
 *  Created on: Aug 14, 2011
 *      Author: kpeterson
 */

#include "InitIdentWeights.hpp"
#include "InitIdentWeightsParams.hpp"

namespace PV {

InitIdentWeights::InitIdentWeights(char const * name, HyPerCol * hc)
{
   initialize_base();
   initialize(name, hc);
}

InitIdentWeights::InitIdentWeights()
{
   initialize_base();
}

InitIdentWeights::~InitIdentWeights()
{
}

int InitIdentWeights::initialize_base() {
   return PV_SUCCESS;
}

int InitIdentWeights::initialize(char const * name, HyPerCol * hc) {
   int status = InitOneToOneWeights::initialize(name, hc);
   return status;
}

InitWeightsParams * InitIdentWeights::createNewWeightParams() {
   InitWeightsParams * tempPtr = new InitIdentWeightsParams(name, parent);
   return tempPtr;
}

int InitIdentWeights::calcWeights(pvdata_t * dataStart, int patchIndex, int arborId) {

   InitIdentWeightsParams *weightParamPtr = dynamic_cast<InitIdentWeightsParams*>(weightParams);


   if(weightParamPtr==NULL) {
      fprintf(stderr, "Failed to recast pointer to weightsParam!  Exiting...");
      exit(PV_FAILURE);
   }


   weightParamPtr->calcOtherParams(patchIndex);


   return createOneToOneConnection(dataStart, patchIndex, 1, weightParamPtr);
}

BaseObject * createInitIdentWeights(char const * name, HyPerCol * hc) {
   return hc ? new InitIdentWeights(name, hc) : NULL;
}

} /* namespace PV */
