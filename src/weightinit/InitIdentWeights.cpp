/*
 * InitIdentWeights.cpp
 *
 *  Created on: Aug 14, 2011
 *      Author: kpeterson
 */

#include "InitIdentWeights.hpp"
#include "InitIdentWeightsParams.hpp"

namespace PV {

InitIdentWeights::InitIdentWeights(HyPerConn * conn)
{
   initialize_base();
   initialize(conn);
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

int InitIdentWeights::initialize(HyPerConn * conn) {
   int status = InitOneToOneWeights::initialize(conn);
   return status;
}

InitWeightsParams * InitIdentWeights::createNewWeightParams() {
   InitWeightsParams * tempPtr = new InitIdentWeightsParams(callingConn);
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

} /* namespace PV */
