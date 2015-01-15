/*
 * InitMaxPoolingWeights.cpp
 *
 *  Created on: Jan 14, 2015
 *      Author: gkenyon
 */

#include "InitMaxPoolingWeights.hpp"
#include "InitMaxPoolingWeightsParams.hpp"

namespace PV {

InitMaxPoolingWeights::InitMaxPoolingWeights(HyPerConn * conn)
{
   initialize_base();
   initialize(conn);
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

int InitMaxPoolingWeights::initialize(HyPerConn * conn) {
   int status = InitWeights::initialize(conn);
   return status;
}

InitWeightsParams * InitMaxPoolingWeights::createNewWeightParams() {
   InitWeightsParams * tempPtr = new InitMaxPoolingWeightsParams(callingConn);
   return tempPtr;
}

int InitMaxPoolingWeights::calcWeights(/* PVPatch * patch */ pvdata_t * dataStart, int patchIndex, int arborId) {
   InitMaxPoolingWeightsParams *weightParamPtr = dynamic_cast<InitMaxPoolingWeightsParams*>(weightParams);

   if(weightParamPtr==NULL) {
      fprintf(stderr, "Failed to recast pointer to weightsParam!  Exiting...");
      exit(1);
   }

   return PV_SUCCESS; // return 1;
}

} /* namespace PV */
